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

from utils import AsyncLoopWrapper, init_logger

logger = init_logger(__name__, logging.INFO)

data_file = ""
round_data = 0
@dataclass
class WorkloadConfig:
    # Max number of users in the system concurrently
    num_users: int

    # Length of shared system prompt
    system_prompt_len: int

    # Length of the user-specific data
    user_info_len: int

    # Length of the answer in one round
    answer_len: int

    # Number of rounds in the conversation
    num_rounds: int

    # Overall QPS
    qps: int

    # Model name
    model: str

    # Whether to include user id in request header
    enable_user_id: bool


@dataclass
class UserConfig:
    # User id
    user_id: int

    # System prompt length
    system_prompt_len: int

    # Length of the user-specific data
    user_info_len: int

    # Answer length
    answer_len: int

    # Gap between two requests
    gap_between_requests: int

    # Num rounds
    num_rounds: int

    # Whether to include user id in request header
    enable_user_id: bool

    @staticmethod
    def new_user_config(user_id: int, workload_config: WorkloadConfig) -> "UserConfig":
        return UserConfig(
            user_id=user_id,
            system_prompt_len=workload_config.system_prompt_len,
            user_info_len=workload_config.user_info_len,
            answer_len=workload_config.answer_len,
            gap_between_requests=workload_config.num_users / workload_config.qps,
            num_rounds=workload_config.num_rounds,
            enable_user_id=workload_config.enable_user_id,
        )

class ChatHistory:

    def __init__(
        self,
    ):
        self.history = []

    def on_user_query(self, query: str):
        """
        添加用户查询到聊天历史中
        AssertionError: 如果当前历史记录最后一条不是系统响应，则抛出异常
        """
        if len(self.history) == 0:
            self.history.append({"role": "user", "content": query})
        else:
            assert self.history[-1]["role"] == "assistant", "Expect system response"
            self.history.append({"role": "user", "content": query})

    def on_system_response(self, response: str):
        """
        添加系统响应到聊天历史中
        
        Args:
            response: 系统生成的响应字符串
        
        Raises:
            AssertionError: 如果历史记录为空或最后一条不是用户查询，则抛出异常
        """
        assert len(self.history) > 0, "Expect user query"
        assert self.history[-1]["role"] == "user", "Expect user query"
        self.history.append({"role": "assistant", "content": response})

    def get_messages_for_openai(self):
        """
        获取符合OpenAI API要求格式的消息列表
        
        Returns:
            list: 包含所有历史消息的列表，每条消息包含role和content字段
        """
        return self.history

    def __len__(self):
        return len(self.history)


@dataclass
class Response:
    body: str
    ttft: float
    generation_time: float
    prompt_tokens: int
    generation_tokens: int
    launch_time: float
    finish_time: float


class RequestExecutor:
    """
    请求执行器类，负责处理与OpenAI API的交互，包括发送请求和处理响应。

    该类封装了与OpenAI API的异步通信逻辑，支持流式响应处理，并收集性能指标如
    首次token响应时间(TTFT)、生成时间、token统计等。
    """
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        # 获取或启动异步事件循环
        self.loop = AsyncLoopWrapper.GetOrStartLoop()
        # 存储请求历史记录
        self.request_history = []

    async def _async_launch_request(self, messages, max_tokens, extra_headers=None):
        """异步发起请求到OpenAI API并处理响应
        
        Args:
            messages: 要发送的消息列表，格式符合OpenAI API要求
            max_tokens: 生成响应的最大token数量
            extra_headers: 可选的额外头信息
        Returns:
            Response对象，包含响应内容和性能指标
        """
        start_time = time.time()
        first_token_time = None
        # 存储生成的完整文本
        words = ""

        extra_body = {}
        # 采样参数全部放到 extra_body
        extra_body.update({
            "min_p": 0.02,
            "top_p": 1,
            "top_k": -1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "repetition_penalty": 1,
            "temperature": 0.8
        })
        # 发送异步请求到OpenAI API，启用流式响应
        response = await self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=0, # 设置为0使输出更加确定性
            stream=True,
            max_tokens=max_tokens,
            stream_options={"include_usage": True}, # 包含token使用统计
            extra_headers=extra_headers,
            extra_body=extra_body
        )

        # 处理流式响应
        async for tok in response:
            if not tok.choices:
                continue
            chunk_message = tok.choices[0].delta.content
            if chunk_message is not None:
                # 记录第一个非空token的到达时间
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
            launch_time=start_time,                             # 请求开始时间
            finish_time=time.time(),                            # 请求完成时间
        )

    def launch_request(
        self,
        chat_history: ChatHistory,
        max_tokens: int,
        finish_callback,
        extra_headers=None,
    ):
        """
        同步接口，用于启动异步请求并设置回调函数
        
        Args:
            chat_history: 聊天历史对象，包含要发送的消息
            max_tokens: 生成响应的最大token数量
            finish_callback: 请求完成时的回调函数，接收Response对象参数
            extra_headers: 可选的额外HTTP头信息
        """
        """
        finish_callback: Callable[[Response], None]
        """
        # 从聊天历史中获取格式化的消息
        messages = chat_history.get_messages_for_openai()
        # 创建实际的回调函数，它会从future对象中提取结果并传递给用户提供的回调
        real_callback = lambda x: finish_callback(x.result())
        # 在事件循环中提交异步任务
        future = asyncio.run_coroutine_threadsafe(
            self._async_launch_request(messages, max_tokens, extra_headers), self.loop
        )
        # 添加完成回调
        future.add_done_callback(real_callback)


class UserSession:
    """
    用户会话类，模拟单个用户与系统的交互过程
    
    该类负责管理用户的聊天历史、生成问题、发送请求并处理响应，
    同时记录会话过程中的各项性能指标。
    """

    def __init__(self, user_config: UserConfig, use_sharegpt=False, sharegpt_data=None):
        self.user_config = user_config
        self.last_request_time = None
        self.chat_history = ChatHistory()
        self.question_id = 0
        self.use_sharegpt = use_sharegpt
        # 添加 conversation_id
        self.conversation_id = str(uuid.uuid4())
        # 如果使用ShareGPT数据，初始化相关设置
        if self.use_sharegpt:
            self.sharegpt_data = sharegpt_data
            # 确定对话开始者（用户或系统）
            if self.sharegpt_data["num_round"] % 2 == 0:
                self.start_with_gpt = False
            else:
                self.start_with_gpt = True
        # 是否有未完成的请求
        self.has_unfinished_request = False
        # 上次记录未完成请求的时间
        self.last_unfinished_log = 0


        # 存储性能指标
        self.prompt_lengths = []
        self.generation_lengths = []
        self.ttfts = []
        self.generation_times = []
        self.launch_times = []
        self.finish_times = []

        self.finished = False

    def _update_result(self, response: Response):
        """
        更新请求结果的统计信息
        
        Args:
            response: API响应对象，包含响应内容和性能指标
        """
        self.prompt_lengths.append(response.prompt_tokens)
        self.generation_lengths.append(response.generation_tokens)
        self.ttfts.append(response.ttft)
        self.generation_times.append(response.generation_time)
        self.launch_times.append(response.launch_time)
        self.finish_times.append(response.finish_time)

    def _build_system_prompt(self):
        """构建系统提示消息"""

        def gen_dummy_text(length):
            return " ".join(["hi"] * length)

        dummy_text_sys = gen_dummy_text(self.user_config.system_prompt_len)
        dummy_text_user = gen_dummy_text(self.user_config.user_info_len)
        system_prompt = (
            f"Hi, here's some system prompt: {dummy_text_sys}."
            + f"For user {self.user_config.user_id}, "
            + f"here are some other context: {dummy_text_user}."
        )
        return system_prompt

    def _build_new_question(self):
        """
        生成一个新的问题
        
        Returns:
            str: 新生成的问题文本
        """
        self.question_id += 1
        return (
            f"Here's question #{self.question_id}: can you tell me "
            + "a new long story with a happy ending?"
        )

    def _launch_new_request(self, timestamp: float, request_executor: RequestExecutor):
        """
        发起一个新的请求
        
        Args:
            timestamp: 当前时间戳
            request_executor: 请求执行器，用于发送API请求
        """
        # 根据是否使用ShareGPT数据获取不同的提示文本
        if self.use_sharegpt:
            if self.start_with_gpt:
                prompt = self.sharegpt_data["conversations"][2 * self.question_id + 1][
                    "value"
                ]
            else:
                prompt = self.sharegpt_data["conversations"][2 * self.question_id][
                    "value"
                ]
            self.question_id += 1
        else:
            prompt = self._build_new_question()

        # 如果是第一次请求，添加系统提示
        # if len(self.chat_history) == 0:
        #     prompt = self._build_system_prompt() + prompt

        # 将用户查询添加到聊天历史
        self.chat_history.on_user_query(prompt)
        # logger.debug(
        #     f"User {self.user_config.user_id} issues request {self.question_id}"
        # )
        # 确定生成响应的最大token数量
        if self.use_sharegpt:
            if self.start_with_gpt:
                try:
                    if data_file == "reasoning.json":
                        max_tokens = self.user_config.answer_len
                    else:
                        max_tokens = self.sharegpt_data["conversations"][2 * self.question_id][
                            "num_tokens"
                        ]
                except:
                    print("使用传入参数 num_tokens")
                    max_tokens = self.user_config.answer_len
            else:
                try:
                    if data_file == "reasoning.json":
                        max_tokens = self.user_config.answer_len
                    else :
                        max_tokens = self.sharegpt_data["conversations"][
                            2 * self.question_id - 1
                        ]["num_tokens"]
                except:
                    print("使用传入参数 num_tokens")
                    max_tokens = self.user_config.answer_len
            max_tokens = min(max_tokens, self.user_config.answer_len)
        else:
            max_tokens = self.user_config.answer_len

        extra_headers = {
            "X-Flow-Conversation-Id": self.conversation_id,
            "X-Request-Id": str(uuid.uuid4()) 
        }
        
        # 发送请求
        request_executor.launch_request(
            self.chat_history,
            max_tokens,
            self._on_request_finished,
            extra_headers=extra_headers
            # extra_headers={"x-user-id": str(self.user_config.user_id)},
        )
        self.has_unfinished_request = True
        self.last_request_time = timestamp

    # 请求完成时的回调函数
    def _on_request_finished(self, response: Response):
        # 将系统响应添加到聊天历史
        self.chat_history.on_system_response(response.body)
        self.has_unfinished_request = False
        # logger.debug(
        #     f"User {self.user_config.user_id} finished one request. "
        #     f"Prompt tokens: {response.prompt_tokens}, "
        #     f"generation tokens: {response.generation_tokens}"
        # )
        # 更新请求结果统计
        self._update_result(response)

    def set_internal_state(self, offset: float, timestamp: float):
        """
        设置会话的内部状态，模拟在指定时间点加入的用户
        
        Args:
            offset: 会话开始相对于系统启动的偏移时间（秒）
            timestamp: 当前时间戳
        """
        """Tell the session is the 'offset' seconds after the start"""
        assert len(self.chat_history) == 0, (
            "Internal state should be set " "before the first request"
        )

        num_passed_questions = int(offset / self.user_config.gap_between_requests) + 1

        passed_time = (num_passed_questions - 1) * self.user_config.gap_between_requests

        # 设置会话的内部状态
        self.last_request_time = timestamp - offset + passed_time
        self.question_id = num_passed_questions
        # logger.debug(
        #     f"Set internal state for user {self.user_config.user_id}, "
        #     f"question_id: {self.question_id}, "
        #     f"last_request_time: {self.last_request_time}"
        # )

    def step(self, timestamp: float, request_executor: RequestExecutor):
        """
        执行一个时间步，处理会话的状态更新和请求发送
        
        Args:
            timestamp: 当前时间戳
            request_executor: 请求执行器，用于发送API请求
        """
        # 检查会话是否已完成
        if (
            self.question_id >= self.user_config.num_rounds
            and not self.has_unfinished_request
        ):
            self.finished = True
            return

        # 如果是第一次请求，立即发送
        if self.last_request_time is None:
            self._launch_new_request(timestamp, request_executor)
            return

        # 检查是否可以发送下一个请求
        if timestamp - self.last_request_time > self.user_config.gap_between_requests:
            # 如果有未完成的请求，记录日志并等待
            if self.has_unfinished_request:
                if timestamp - self.last_unfinished_log > 10:
                    # logger.warning(
                    #     f"User {self.user_config.user_id} has an unfinished "
                    #     "request and unable to fit the QPS requirement."
                    # )
                    self.last_unfinished_log = timestamp
                return
            # 发送下一个请求
            self._launch_new_request(timestamp, request_executor)
            return

    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame()
        df["prompt_tokens"] = self.prompt_lengths
        df["generation_tokens"] = self.generation_lengths
        df["ttft"] = self.ttfts
        df["generation_time"] = self.generation_times
        df["user_id"] = self.user_config.user_id
        df["question_id"] = range(1, len(self.prompt_lengths) + 1)
        df["launch_time"] = self.launch_times
        df["finish_time"] = self.finish_times
        return df


class UserSessionManager:
    """
    用户会话管理器类，负责管理多个用户会话的生命周期和交互过程
    
    该类处理用户会话的创建、初始化、执行和清理，实现多用户场景下的负载模拟，
    并提供性能统计功能以评估系统在多轮问答场景下的表现。
    """

    def __init__(
        self, workload_config: WorkloadConfig, init_user_id=0, use_sharegpt=False
    ):
        self.workload_config = workload_config
        self.sessions = []

         # 计算每个用户的请求之间的间隔时间
        gap_between_requests_per_user = workload_config.num_users / workload_config.qps
        if workload_config.num_rounds == 1:
            session_alive_time = gap_between_requests_per_user
        else:
            session_alive_time = gap_between_requests_per_user * (
                workload_config.num_rounds - 1
            )
        # 计算用户加入的间隔时间
        self.gap_between_users = session_alive_time / (workload_config.num_users + 0)
        # 计算系统预热时间
        self.ramp_up_time = workload_config.num_users * self.gap_between_users

        # logger.info(
        #     f"Gap between users: {self.gap_between_users} secs.\n"
        #     f"Gap between user reqs: {gap_between_requests_per_user} secs.\n"
        #     f"Expected length of user session: {session_alive_time} secs."
        # )
        self.init_user_id = init_user_id
        self.user_id = init_user_id
        self.last_user_join = 0
        self.session_summaries = []
        self.start_time = None

        self.need_ramp_up = True

        self.use_sharegpt = use_sharegpt
        if self.use_sharegpt:
            self._load_sharegpt_data()

    def _load_sharegpt_data(self):
        # with open("sharegpt.json", "r", encoding="utf-8") as file:
        current_file_path = os.path.abspath(__file__)
        with open(os.path.join(os.path.dirname(current_file_path), data_file), "r", encoding="utf-8") as file:
            self.sharegpt_data = json.load(file)
        print(f"使用 {data_file} 数据集文件")
        # 过滤出满足对话轮数要求的数据
        self.sharegpt_data = [
            d
            for d in self.sharegpt_data
            if d["num_round"] >= 2 * self.workload_config.num_rounds
        ]
        # logger.info(f"There are {len(self.sharegpt_data)} users satisfying ")

    def _ramp_up(self, timestamp: float, ramp_up_time: float):
        """
        系统预热阶段，初始化多个用户会话
        
        Args:
            timestamp: 当前时间戳
            ramp_up_time: 预热时间
        """
        for i in range(self.workload_config.num_users):
            new_session = self._create_user_session()
            offset = ramp_up_time - i * self.gap_between_users
            if offset < 0:
                break
            # 设置会话的内部状态，模拟不同时间点加入的用户
            new_session.set_internal_state(offset, timestamp)
        self.need_ramp_up = False

    def _create_user_session(self):
        """
        创建一个新的用户会话
        
        Returns:
            UserSession: 新创建的用户会话对象
        """
        self.user_id += 1
        user_config = UserConfig.new_user_config(self.user_id, self.workload_config)
        # 根据是否使用ShareGPT数据集创建不同的用户会话
        if self.use_sharegpt:
            sharegpt_data_id = (self.user_id - self.init_user_id)%round_data + self.init_user_id
            user_session = UserSession(
                user_config, self.use_sharegpt, self.sharegpt_data[sharegpt_data_id]
            )
        else:
            user_session = UserSession(user_config, self.use_sharegpt)
        self.sessions.append(user_session)
        return user_session

    def _remove_finished_sessions(self):
        """
        移除已完成的会话，收集其统计信息
        """
        # 找出所有已完成的会话
        sessions_to_remove = [s for s in self.sessions if s.finished]
        if len(sessions_to_remove) > 0:
            mark = 0
            # logger.info(
            #     f"Removing {len(sessions_to_remove)} finished sessions, now "
            #     f"active users: {len(self.sessions) - len(sessions_to_remove)}"
            # )
            # 收集已完成会话的统计信息
            for session in sessions_to_remove:
                self.session_summaries.append(session.summary())
        # 更新会话列表，只保留未完成的会话
        self.sessions = [s for s in self.sessions if not s.finished]

    def step(self, timestamp: float, executor: RequestExecutor):
        """
        执行一个时间步，处理用户会话的状态更新和请求发送
        
        Args:
            timestamp: 当前时间戳
            executor: 请求执行器，用于发送API请求
        """
        # 处理系统预热
        if self.need_ramp_up:
            self._ramp_up(timestamp, self.ramp_up_time)

        # 设置开始时间
        if self.start_time is None:
            self.start_time = timestamp

        # 检查是否需要添加新用户
        if timestamp - self.last_user_join > self.gap_between_users:
            self._create_user_session()
            self.last_user_join = timestamp
            # logger.info(
            #     f"Joined a new user {self.user_id}, "
            #     f"now active users: {len(self.sessions)}"
            # )
        # 更新所有会话的状态
        for session in self.sessions:
            session.step(timestamp, executor)
        # 清理已完成的会话
        self._remove_finished_sessions()

    @staticmethod
    def ProcessSummary(
        df: pd.DataFrame,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        pending_queries: int = 0,
        qps: Optional[int] = None,
    ):
        if start_time and end_time:
            launched_queries = len(
                df.query(f"{start_time} <= launch_time <= {end_time}")
            )
            df = df.query(f"{start_time} <= finish_time <= {end_time}")
        else:
            launched_queries = len(df)

        # logger.debug(
        #     f"Launched queries: {launched_queries}, "
        #     f"pending queries: {pending_queries}, "
        #     f"finished queries: {len(df)}"
        # )

        if qps is None:
            qps = 0.0

        if start_time is None:
            start_time = df["launch_time"].min()
        if end_time is None:
            end_time = df["finish_time"].max()
        total_time = end_time - start_time

        total_requests = launched_queries + pending_queries
        _qps = total_requests / total_time

        total_finished_requests = len(df)
        finished_qps = total_finished_requests / total_time

        total_prompt_tokens = df["prompt_tokens"].sum()
        total_generation_tokens = df["generation_tokens"].sum()
        average_prefill_speed = total_prompt_tokens / total_time
        average_generation_speed = total_generation_tokens / total_time
        average_generation_speed_per_request = (
            df["generation_tokens"] / df["generation_time"]
        ).mean()
        average_ttft = df["ttft"].mean()
        # logger.info("Calculating performance summary")
        # print("\n")
        # print("==================== Performance summary ======================")
        # print(f"  \033[33mQPS: \033[32m{qps:.4f} reqs/s\033[0m\n")

        # print(
        #     f"  \033[33mProcessing speed: "
        #     f"\033[32m{finished_qps:.4f} reqs/s\033[0m\n"
        # )

        # print(f"  \033[33mRequests on-the-fly: {pending_queries}\033[0m\n")

        # print(
        #     "  \033[33mInput tokens per second: "
        #     f"\033[32m{average_prefill_speed:.4f} tokens/s\033[0m\n"
        # )

        # print(
        #     "  \033[33mOutput tokens per second: "
        #     f"\033[32m{average_generation_speed:.4f} tokens/s\033[0m\n"
        # )

        # print(
        #     "  \033[33mAverage generation throughput (per request): "
        #     f"\033[32m{average_generation_speed_per_request:.4f} "
        #     "tokens/req/s\033[0m\n"
        # )

        # print(f"  \033[33mAverage TTFT: \033[32m{average_ttft:.4f}s\033[0m\n")

        # print(f"Time range: {start_time} - {end_time} ({total_time:.2f}s)")

        # print("===============================================================")
        # print("\n")
        return df

    def summary(self, start_time: float, end_time: float) -> pd.DataFrame:
        if len(self.session_summaries) == 0 and len(self.sessions) == 0:
            return pd.DataFrame()

        df = pd.concat(
            [s for s in self.session_summaries] + [s.summary() for s in self.sessions]
        )
        pending_queries = len([s for s in self.sessions if s.has_unfinished_request])
        start_time = max(self.start_time, start_time)
        end_time = min(end_time, df["finish_time"].max())
        qps = self.workload_config.qps

        df = UserSessionManager.ProcessSummary(
            df, start_time, end_time, pending_queries, qps
        )
        return df


def warmup_engine(executor):
    # logger.info("Warming up the engine")
    for i in range(10):
        chat_history = ChatHistory()
        chat_history.on_user_query(
            f"WARMUP: Hi, I'm user {i}. Here are some text: {'hi ' * 100}."
        )
        executor.launch_request(chat_history, 100, lambda x: None)

    AsyncLoopWrapper.WaitLoop()


def parse_arguments() -> WorkloadConfig:
    parser = argparse.ArgumentParser(description="Parse benchmark configurations.")

    parser.add_argument(
        "--data-file",
        type=str,
        default=""
    )

    parser.add_argument(
        "--num-users",
        type=int,
        required=True,
        help="Max number of users in the system concurrently",
    )
    parser.add_argument(
        "--shared-system-prompt",
        type=int,
        required=True,
        help="Length of the shared system prompt (tokens)",
    )
    parser.add_argument(
        "--user-history-prompt",
        type=int,
        required=True,
        help="Length of the user-specific history prompt (tokens)",
    )
    parser.add_argument(
        "--answer-len",
        type=int,
        required=True,
        help="Length of the answer in one round",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        required=True,
        help="Number of rounds in the conversation",
    )
    parser.add_argument("--qps", type=float, required=True, help="Overall QPS")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--api-key",
        type=str,
        required=False,
        default="EMPTY",
        help="Bearer token for authentication",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="Base URL of the serving engine endpoint",
    )
    parser.add_argument(
        "--time",
        type=int,
        required=False,
        help="The time to run the simulation in seconds",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="summary.csv",
        help="The output file name (ended with csv or txt) "
        "for the summary csv and txt",
    )
    parser.add_argument(
        "--init-user-id", type=int, default=0, help="The initial user id to start with"
    )

    # 添加是否在请求头中启用用户ID的标志参数（可选，默认不启用）
    parser.add_argument(
        "--request-with-user-id",
        action="store_true",
        help="Whether to enable user id in the request headers",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=30,
        help="The time between two summary loggings in seconds",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Whether to enable verbose logging"
    )
    parser.add_argument(
        "--sharegpt", action="store_true", help="Whether to use sharegpt dataset"
    )
    parser.add_argument(
        "--round-data",type=int
    )

    args = parser.parse_args()
    global data_file,round_data
    round_data = args.round_data
    data_file = args.data_file
    return args


def parse_process_summary():
    parser = argparse.ArgumentParser(
        description="Parse benchmark configurations.", add_help=False
    )

    parser.add_argument("--process-summary", type=str, default=None)

    args, _ = parser.parse_known_args()
    return args


def process_output(filename):
    # logger.warning(
    #     f"Processing the existing summary file {filename}"
    #     ", ignoring all the other arguments"
    # )
    UserSessionManager.ProcessSummary(pd.read_csv(filename), pending_queries=0)


def main():
    args = parse_process_summary()
    if args.process_summary:
        # 如果指定了处理已有摘要文件，则直接处理并返回
        process_output(args.process_summary)
        return

    args = parse_arguments()
    if args.verbose:
        global logger
        logger = init_logger(__name__, log_level=logging.DEBUG)

    # 根据测试轮数设置步骤间隔
    if args.num_rounds == 1:
        step_interval = 0.01
    else:
        step_interval = 0.01

    # 创建请求执行器，用于向模型发送请求
    executor = RequestExecutor(
        base_url=args.base_url, api_key=args.api_key, model=args.model
    )

    # 预热引擎，确保模型服务在正式测试前已准备就绪
    # warmup_engine(executor)

    # 创建工作负载配置对象，包含所有测试参数
    workload_config = WorkloadConfig(
        num_users=args.num_users,
        system_prompt_len=args.shared_system_prompt,
        user_info_len=args.user_history_prompt,
        answer_len=args.answer_len,
        num_rounds=args.num_rounds,
        qps=args.qps,
        model=args.model,
        enable_user_id=args.request_with_user_id, # 是否启用用户ID
    )
    # 创建用户会话管理器，负责管理所有用户会话
    manager = UserSessionManager(
        workload_config, init_user_id=args.init_user_id, use_sharegpt=args.sharegpt
    )
    # 初始化计数器和计时器
    num_steps = 0
    start_time = time.time()
    last_summary_time = start_time
    try:
        while True:
            num_steps += 1
            # 执行一步测试，处理所有活跃用户的请求
            manager.step(time.time(), executor)
            # 等待指定的步骤间隔
            time.sleep(step_interval)

            # 定期生成性能摘要报告
            if time.time() - last_summary_time > args.log_interval:
                manager.summary(last_summary_time, time.time())
                last_summary_time = time.time()

            if args.time is not None and time.time() - start_time > args.time:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted, waiting for the final result")

    # 停止异步事件循环
    AsyncLoopWrapper.StopLoop()
     # 输出最终的性能摘要并保存到文件
    # logger.info(f"Finished benchmarking, dumping summary to {args.output}")
    summary = manager.summary(0, time.time())
    summary.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
