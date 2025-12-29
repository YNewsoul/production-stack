import argparse
import json

import tiktoken

def estimate_num_tokens(text:str) ->int:
    enc = tiktoken.encoding_for_model("gpt-4o")
    return len(enc.encode(text, allowed_special={"<|endoftext|>"}))

parser = argparse.ArgumentParser(description="Process data percentage.")
parser.add_argument("--parse",type=float,default=1,help="The percentage of data to process (0 to 1). Default is 1 (100%).",
)

parser.add_argument("--datas",type=str,required=True, help="The datas to process. Default is sharegpt_process_data.json.",
)
args = parser.parse_args()

with open(args.datas, "r", encoding="utf-8") as file:
    data = json.load(file)

num_of_ids = len(data)
print(f"处理前数据量: {num_of_ids}")
data = data[: int(num_of_ids * args.parse)]

# 确定 sharegpt 从 human 开始
if args.datas == "sharegpt":
    data = [
    d for d in data
    if d.get("conversations") and d["conversations"][0].get("from") != "gpt"
]
print(f"处理后数据长度: {len(data)}")

count = 0

for d in data:

    # if args.dataset == "sharegpt":
    #     # 只保留第一轮（human + gpt）对话
    #     first_human = None
    #     first_gpt = None
    #     for conv in d["conversations"]:
    #         if conv.get("from") == "human" and first_human is None:
    #             first_human = conv
    #         elif conv.get("from") == "gpt" and first_human is not None:
    #             first_gpt = conv
    #             break
    #     if first_human and first_gpt:
    #         d["conversations"] = [first_human, first_gpt]
    #     else:
    #         d["conversations"] = []

    d["num_round"] = len(d["conversations"])  # human is one round, gpt is another round
    human_tokens = []
    gpt_tokens = []
    for conv in d["conversations"]:
        token_number = estimate_num_tokens(conv["value"])
        conv["num_tokens"] = token_number
        if conv["from"] == "human":
            human_tokens.append(token_number)
        if conv["from"] == "gpt":
            gpt_tokens.append(token_number)
            
    count += 1
    if count % 1000 == 0:
        print(f"已处理 {count} 条数据")

output_file = args.datas.replace(f"_preprocess_data.json",".json")
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=2)
print(f"文件保存至: {output_file}")
