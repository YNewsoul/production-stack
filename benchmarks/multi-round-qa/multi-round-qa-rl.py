import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional
import os
import openai
import uuid
import pandas as pd
import time
import bisect
import numpy as np

from utils import AsyncLoopWrapper, init_logger
from Predictor import Predictor

logger = init_logger(__name__, logging.INFO)

@dataclass
class WorkloadConfig:
    answer_len: int
    num_rounds: int
    qps: int
    model: str
    dataset:str
    use_predict: bool
    round_data: int

@dataclass
class UserConfig:
    user_id: int
    answer_len: int
    num_rounds: int
    use_predict: bool
    dataset: str

    @staticmethod
    def new_user_config(user_id: int, workload_config: WorkloadConfig) -> "UserConfig":
        return UserConfig(
            user_id=user_id,
            answer_len=workload_config.answer_len,
            num_rounds=workload_config.num_rounds,
            use_predict=workload_config.use_predict,
            dataset=workload_config.dataset,
        )
class ChatHistory:

    def __init__(self,):
        self.history = []

    def on_user_query(self, query: str, num_tokens: Optional[int] = None):
        """添加用户查询到聊天历史中"""
        self.history.append({"role": "user", "content": query, "tokens": num_tokens})

    def on_system_response(self, response: str, num_tokens: Optional[int] = None):
        """添加系统响应到聊天历史中"""
        self.history.append({"role": "assistant", "content": response, "tokens": num_tokens})

    def get_messages_for_openai(self):
        """获取符合OpenAI API要求格式的消息列表,list: 包含所有历史消息的列表，每条消息包含role和content字段"""
        return [{"role": item["role"], "content": item["content"]} for item in self.history]

    def estimate_prompt_tokens(self) -> int:
        """获取当前请求的提示 tokens 数"""
        total = 0
        for item in self.history:
            total += item["tokens"]
        return total

@dataclass
class Response:
    body: str
    ttft: float
    generation_time: float
    prompt_tokens: int
    generation_tokens: int
    send_time: float
    end_time: float
    max_tokens: int
    predict_tokens:int
    ttft_slo:float

class RequestExecutor:
    """请求执行器类，负责处理与OpenAI API的交互，包括发送请求和处理响应。"""
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.loop = AsyncLoopWrapper.GetOrStartLoop()
        self.request_history = []

    async def _async_launch_request(self, request_data=None, extra_headers=None):
        """异步发起请求到OpenAI API并处理响应"""
        start_time = time.time()
        first_token_time = None
        words = ""

        extra_body = {}
        extra_body.update({
            "min_p": 0.02,
            "top_p": 1,
            "top_k": -1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "repetition_penalty": 1,
            "temperature": 0,
            "extra_data":{
                "ttft_slo":request_data["ttft_slo"],
                "arrival_time":time.monotonic(),},
        })
        
        # 发送异步请求到OpenAI API，启用流式响应
        response = await self.client.chat.completions.create(
            messages=request_data["messages"],
            model=self.model,
            temperature=0, # 设置为0使输出更加确定性
            stream=True,
            max_tokens=request_data["max_tokens"],
            stream_options={"include_usage": True}, # 包含token使用统计
            extra_headers=extra_headers,
            extra_body=extra_body,
        )

        # 处理流式响应
        async for tok in response:
            if not tok.choices:
                continue
            chunk_message = tok.choices[0].delta.content
            if chunk_message is not None:
                if first_token_time is None and chunk_message != "":
                    first_token_time = time.time()
                # 累积生成的文本
                words += chunk_message
        # 提取token使用统计信息
        tokens_out = tok.usage.completion_tokens # 生成的token数量
        tokens_prefill = tok.usage.prompt_tokens # 提示的token数量

        # 构造并返回响应对象，包含性能指标
        return Response(
            body=words,                                         # 生成的完整文本
            ttft=first_token_time - start_time,                 # ttft
            generation_time=time.time() - first_token_time,     # 整体生成时间
            prompt_tokens=tokens_prefill,                       # 提示的token数量
            generation_tokens=tokens_out,                       # 生成的token数量
            send_time=start_time,                             # 请求开始时间
            end_time=time.time(),                            # 请求完成时间
            predict_tokens=request_data["predict_tokens"],
            max_tokens=request_data["max_tokens"],
            ttft_slo=request_data["ttft_slo"],
        )

    def launch_request(self,request_data,finish_callback,extra_headers=None):
        """同步接口，用于启动异步请求并设置回调函数"""

        # 创建实际的回调函数，它会从future对象中提取结果并传递给用户提供的回调
        real_callback = lambda x: finish_callback(x.result())
        # 在事件循环中提交异步任务
        future = asyncio.run_coroutine_threadsafe(
            self._async_launch_request(request_data, extra_headers), self.loop
        )
        # 添加完成回调
        future.add_done_callback(real_callback)

class UserSession:
    """用户会话类，模拟单个用户与系统的交互过程"""
    def __init__(self, user_config: UserConfig, sharegpt_data=None):
        self.user_config = user_config
        self.last_request_time = None
        self.chat_history = ChatHistory()
        self.question_id = 0
        self.conversation_id = str(uuid.uuid4())
        self.sharegpt_data = sharegpt_data
        self.has_unfinished_request = False

        # 存储性能指标
        self.prompt_lengths = []
        self.generation_lengths = []
        self.ttfts = []
        self.tpots = []
        self.generation_times = []
        self.send_times = []
        self.end_times = []
        self.latencys = []
        self.ttft_slos = []
        self.ttft_slo_comforms = []
        self.tpot_slo_comforms = []
        self.max_generation_lengths = []
        self.predict_generation_lengths = []

        self.finished = False

        # a6000 tp2 设置
        # self.ttft_slo_dict = {
        #         1000:2.5, 2000:3.5, 4000:5.5, 6000:7.5, 8000:9.5, 10000:11.5, 12000:14, 14000:16.5, 16000:19,
        #         18000:21.5, 20000:24, 22000:26.5, 24000:29.5, 26000:32.5, 28000:36}
        # 3090 tp2 设置
        self.ttft_slo_dict = {
                1000:5, 2000:7.5, 4000:12, 6000:17, 8000:22, 10000:27, 12000:32, 14000:38, 16000:44,
                18000:49, 20000:56, 22000:62, 24000:69, 26000:75, 28000:82}
        self.sorted_keys = [1000,2000,4000,6000,8000,10000,12000, 14000,16000, 18000,20000,22000,24000,26000,28000]

    def _update_result(self, response: Response):
        """更新请求结果的统计信息"""
        self.prompt_lengths.append(response.prompt_tokens)
        self.generation_lengths.append(response.generation_tokens)
        self.predict_generation_lengths.append(response.predict_tokens)
        self.max_generation_lengths.append(response.max_tokens)
        self.ttfts.append(response.ttft)
        self.tpots.append((response.generation_time)*1000/response.generation_tokens)
        self.generation_times.append(response.generation_time)
        self.send_times.append(response.send_time)
        self.end_times.append(response.end_time)
        self.latencys.append(response.end_time - response.send_time)
        self.ttft_slos.append(response.ttft_slo)
        self.ttft_slo_comforms.append(True if response.ttft <= response.ttft_slo else False)
        self.tpot_slo_comforms.append(True if self.tpots[-1] <= 50 else False)

    def _select_ttft_slo(self, prompt_tokens: int) -> int:
        """根据提示 tokens 的分段选择对应的 ttft_slo"""
        index = bisect.bisect_left(self.sorted_keys, prompt_tokens)
        if index < len(self.sorted_keys):
            return self.ttft_slo_dict[self.sorted_keys[index]]
        return self.ttft_slo_dict[self.sorted_keys[-1]]

    def _launch_new_request(self, predictor:Predictor, timestamp: float, request_executor: RequestExecutor):
        """发起一个新的请求"""
        prompt_data = self.sharegpt_data["conversations"][2 * self.question_id]
        prompt = prompt_data["value"]
        prompt_tokens = prompt_data.get("num_tokens")
        if self.user_config.dataset == "reasoning.json":
            max_tokens = self.user_config.answer_len
        else :
            max_tokens = self.sharegpt_data["conversations"][2 * self.question_id + 1]["num_tokens"]
        # 做个约束
        max_tokens = min(max_tokens, self.user_config.answer_len)
        self.question_id += 1

        # 将用户查询添加到聊天历史
        self.chat_history.on_user_query(prompt, prompt_tokens)
        messages = self.chat_history.get_messages_for_openai()

        predict_tokens = 2000
        ttft_slo = self._select_ttft_slo(self.chat_history.estimate_prompt_tokens())

        # 是否使用预测器
        if self.user_config.use_predict:
            predict_tokens = predictor.predict(str(messages))

        extra_headers = {
            "X-Flow-Conversation-Id": self.conversation_id,
            "X-Request-Id": str(uuid.uuid4()) 
        }

        request_data = {
            "messages":messages,
            "predict_tokens":predict_tokens,
            "max_tokens":max_tokens,
            "ttft_slo":ttft_slo,}

        # 发送请求
        request_executor.launch_request(
            request_data,
            self._on_request_finished,
            extra_headers=extra_headers,
        )
        self.has_unfinished_request = True
        self.last_request_time = timestamp

    # 请求完成时的回调函数
    def _on_request_finished(self, response: Response):
        # 将系统响应添加到聊天历史
        self.chat_history.on_system_response(response.body, response.generation_tokens)
        self.has_unfinished_request = False
        # 更新请求结果统计
        self._update_result(response)

    def step(self,predictor: Predictor, timestamp: float, request_executor: RequestExecutor):
        """执行一个时间步，处理会话的状态更新和请求发送"""
        # 检查会话是否已完成
        if (self.question_id >= self.user_config.num_rounds and not self.has_unfinished_request):
            self.finished = True
            return

        # 如果是第一次请求，立即发送
        if self.last_request_time is None:
            self._launch_new_request(predictor,timestamp, request_executor)
            return True

        # 如果有未完成的请求，记录日志并等待
        if self.has_unfinished_request:
            return
        else:
            self._launch_new_request(predictor,timestamp, request_executor)
            return True

    def session_summary(self) -> pd.DataFrame:
        df = pd.DataFrame()
        df["prompt_tokens"] = self.prompt_lengths
        df["generation_tokens"] = self.generation_lengths
        df["predict_tokens"] = self.predict_generation_lengths
        df["max_tokens"] = self.max_generation_lengths
        df["ttft"] = self.ttfts
        df["generation_time"] = self.generation_times
        df["user_id"] = self.user_config.user_id
        df["question_id"] = range(1, len(self.prompt_lengths) + 1)
        df["send_time"] = self.send_times
        df["end_time"] = self.end_times
        df["ttft_slo"] = self.ttft_slos
        df["ttft_slo_comform"] = self.ttft_slo_comforms
        df["tpot_slo_comform"] = self.tpot_slo_comforms
        
        return df

class UserSessionManager:
    """用户会话管理器类，负责管理多个用户会话的生命周期和交互过程"""

    def __init__(
        self, workload_config: WorkloadConfig, init_user_id=0
    ):
        self.workload_config = workload_config
        self.sessions = []

        self.init_user_id = init_user_id
        self.user_id = init_user_id
        self.session_summaries = []
        self.start_time = None

        self._load_sharegpt_data()

    def _load_sharegpt_data(self):
        current_file_path = os.path.abspath(__file__)
        with open(os.path.join(os.path.dirname(current_file_path), self.workload_config.dataset), "r", encoding="utf-8") as file:
            self.sharegpt_data = json.load(file)
        logger.info(f"使用 {self.workload_config.dataset} 数据集文件")
        # 过滤出满足对话轮数要求的数据
        self.sharegpt_data = [
            d for d in self.sharegpt_data
            if d["num_round"] >= 2 * self.workload_config.num_rounds and  d["num_round"]%2 == 0
        ]
        logger.info(f"总请求个数:{len(self.sharegpt_data)}")

    def _create_user_session(self):
        """创建一个新的用户会话"""
        self.user_id += 1
        user_config = UserConfig.new_user_config(self.user_id, self.workload_config)
        # 根据是否使用ShareGPT数据集创建不同的用户会话
        # 计算ShareGPT数据集中的数据索引
        sharegpt_data_id = (self.user_id - self.init_user_id)%self.workload_config.round_data + self.init_user_id
        user_session = UserSession(user_config, self.sharegpt_data[sharegpt_data_id])
        self.sessions.append(user_session)

    def _remove_finished_sessions(self):
        """移除已完成的会话，收集其统计信息"""
        # 找出所有已完成的会话
        sessions_to_remove = [s for s in self.sessions if s.finished]
        # 收集已完成会话的统计信息
        for session in sessions_to_remove:
            self.session_summaries.append(session.session_summary())
        # 更新会话列表，只保留未完成的会话
        self.sessions = [s for s in self.sessions if not s.finished]

    def step(self, predictor: Predictor, executor: RequestExecutor):
        """执行一个时间步，处理用户会话的状态更新和请求发送"""
        # 设置开始时间
        timestamp = time.time()
        if self.start_time is None:
            self.start_time = timestamp
        
        self._create_user_session()

        # 更新所有会话的状态
        for session in self.sessions:
            mark = session.step(predictor,timestamp, executor)
            if mark:
                break
        self._remove_finished_sessions()

    # 处理文件并打印性能摘要
    @staticmethod
    def ProcessSummary(
        df: pd.DataFrame,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        pending_queries: int = 0,
        qps: Optional[int] = None,
    ):
        launched_queries = len(df.query(f"{start_time} <= send_time <= {end_time}"))
        df = df.query(f"{start_time} <= end_time <= {end_time}")

        total_time = end_time - start_time

        total_requests = launched_queries + pending_queries
        qps = total_requests / total_time

        total_finished_requests = len(df)
        finished_qps = total_finished_requests / total_time

        total_prompt_tokens = df["prompt_tokens"].sum()
        total_generation_tokens = df["generation_tokens"].sum()
        average_prefill_speed = total_prompt_tokens / total_time
        average_generation_speed = total_generation_tokens / total_time
        average_generation_speed_per_request = (df["generation_tokens"] / df["generation_time"]).mean()
        average_ttft = df["ttft"].mean()
        print("\n")
        print(f"  \033[33mQPS: \033[32m{qps:.4f} reqs/s\033[0m\n")

        print(
            f"  \033[33mProcessing speed: "
            f"\033[32m{finished_qps:.4f} reqs/s\033[0m\n"
        )

        print(
            "  \033[33mInput tokens per second: "
            f"\033[32m{average_prefill_speed:.4f} tokens/s\033[0m\n"
        )

        print(
            "  \033[33mOutput tokens per second: "
            f"\033[32m{average_generation_speed:.4f} tokens/s\033[0m\n"
        )

        print(
            "  \033[33mAverage generation throughput (per request): "
            f"\033[32m{average_generation_speed_per_request:.4f} "
            "tokens/req/s\033[0m\n"
        )

        print(f"  \033[33mAverage TTFT: \033[32m{average_ttft:.4f}s\033[0m\n")

        print(f"Time range: {start_time} - {end_time} ({total_time:.2f}s)")

        print("===============================================================")
        print("\n")
        return df

    def summary(self, start_time: float, end_time: float) -> pd.DataFrame:
        if len(self.session_summaries) == 0 and len(self.sessions) == 0:
            return pd.DataFrame()

        df = pd.concat(
            [s for s in self.session_summaries] + [s.session_summary() for s in self.sessions]
        )
        pending_queries = len([s for s in self.sessions if s.has_unfinished_request])
        qps = self.workload_config.qps

        df = UserSessionManager.ProcessSummary(
            df, start_time, end_time, pending_queries, qps
        )
        return df

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse benchmark configurations.")
    parser.add_argument("--dataset",type=str,default="")
    parser.add_argument("--answer-len",type=int,required=True)
    parser.add_argument("--num-rounds",type=int,required=True)
    parser.add_argument("--qps", type=float, required=True, help="Overall QPS")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--api-key",type=str,required=False,default="EMPTY")
    parser.add_argument("--base-url",type=str,required=True,)
    parser.add_argument("--time",type=int,required=False)
    parser.add_argument("--output",type=str,default="summary.csv",)
    parser.add_argument("--init-user-id", type=int, default=0)
    parser.add_argument("--cv",type=float,default=0.0)
    parser.add_argument("--use-predict",type=bool,default=False)
    parser.add_argument("--log-interval",type=int,default=60)
    parser.add_argument("--round-data",type=int)

    return parser.parse_args()

def main():

    args = parse_arguments()

    step_interval = 1 / args.qps

    # 动态生成每个间隔
    if args.cv != 0.0 :
        k = 1 / (args.cv ** 2)
        theta = step_interval / k

    # 创建请求执行器，用于向模型发送请求
    executor = RequestExecutor(base_url=args.base_url, api_key=args.api_key, model=args.model)

    # 初始化输出长度预测器
    predictor = None
    if args.use_predict:
        predictor = Predictor()

    # 创建工作负载配置对象，包含所有测试参数
    workload_config = WorkloadConfig(
        answer_len=args.answer_len,
        num_rounds=args.num_rounds,
        qps=args.qps,
        model=args.model,
        dataset=args.dataset,
        round_data=args.round_data,
        use_predict=args.use_predict
    )

    # 创建用户会话管理器，负责管理所有用户会话
    manager = UserSessionManager(workload_config, init_user_id=args.init_user_id)

    # 初始化计数器和计时器
    num_steps = 0
    start_time = time.time()
    last_summary_time = start_time

    while True:
        num_steps += 1
        # 执行一步测试，处理所有活跃用户的请求
        manager.step(predictor, executor)
        # 等待指定的步骤间隔
        if args.cv != 0.0 :
            interval = max(0, np.random.gamma(k, theta))
        else:
            interval = step_interval
        time.sleep(interval)

        if time.time() - last_summary_time > args.log_interval:
            summary = manager.summary(last_summary_time, time.time())
            summary.to_csv(args.output, index=False,model='a')
            last_summary_time = time.time()

        if args.time is not None and time.time() - start_time > args.time:
            break

    # 停止异步事件循环
    AsyncLoopWrapper.StopLoop()

if __name__ == "__main__":
    main()