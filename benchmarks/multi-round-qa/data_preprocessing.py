import argparse
import json

import tiktoken

def estimate_num_tokens(text:str) ->int:
    enc = tiktoken.encoding_for_model("gpt-4o")
    return len(enc.encode(text, allowed_special={"<|endoftext|>"}))

parser = argparse.ArgumentParser(description="Process data percentage.")
parser.add_argument(
    "--parse",
    type=float,
    default=1,
    help="The percentage of data to process (0 to 1). Default is 1 (100%).",
)

parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="The dataset to process. Default is sharegpt.",
)
args = parser.parse_args()

if args.dataset == "sharegpt":
    with open("ShareGPT_V3_unfiltered_cleaned_split.json", "r", encoding="utf-8") as file:
        data = json.load(file)
elif args.dataset == "coder":
    with open("coder_preprocess_data.json", "r", encoding="utf-8") as file:
        data = json.load(file)
elif args.dataset == "arxiv":
    with open("arxiv_preprocess_data.json", "r", encoding="utf-8") as file:
        data = json.load(file)
elif args.dataset == "reasoning":
    with open("reasoning_preprocess_data.json", "r", encoding="utf-8") as file:
        data = json.load(file)
elif args.dataset == "mixed_arxiv_coder":
    with open("mixed_arxiv_coder_preprocess_data.json", "r", encoding="utf-8") as file:
        data = json.load(file)
elif args.dataset == "mixed_arxiv_sharegpt":
    with open("mixed_arxiv_sharegpt_preprocess_data.json", "r", encoding="utf-8") as file:
        data = json.load(file)

num_of_ids = len(data)
print(f"处理前数据量: {num_of_ids}")
data = data[: int(num_of_ids * args.parse)]

# 确定 sharegpt 从 human 开始
if args.dataset == "sharegpt":
    data = [
    d for d in data
    if d.get("conversations") and d["conversations"][0].get("from") != "gpt"
]
print(f"处理后数据长度: {len(data)}")

count = 0

for d in data:

    if args.dataset == "sharegpt":
        # 只保留第一轮（human + gpt）对话
        first_human = None
        first_gpt = None
        for conv in d["conversations"]:
            if conv.get("from") == "human" and first_human is None:
                first_human = conv
            elif conv.get("from") == "gpt" and first_human is not None:
                first_gpt = conv
                break
        if first_human and first_gpt:
            d["conversations"] = [first_human, first_gpt]
        else:
            d["conversations"] = []

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
            
    # 这几行注释掉，没用
    # if len(human_tokens) == 0:
    #     d["average_human_token"] = 0
    #     d["max_human_token"] = 0
    # else:
    #     d["average_human_token"] = float(np.mean(human_tokens))
    #     d["max_human_token"] = float(np.max(human_tokens))
    # if len(gpt_tokens) == 0:
    #     d["average_gpt_token"] = 0
    #     d["max_gpt_token"] = 0
    # else:
    #     d["average_gpt_token"] = float(np.mean(gpt_tokens))
    #     d["max_gpt_token"] = float(np.max(gpt_tokens))

    count += 1
    if count % 100 == 0:
        print(f"Finished {count}")

# Remove the data that has two consecutive human rounds
del data[260]

if args.dataset == "sharegpt":
    with open("sharegpt.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
elif args.dataset == "coder":
    with open("coder.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
elif args.dataset == "arxiv":
    with open("arxiv.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
elif args.dataset == "reasoning":
    with open("reasoning.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
elif args.dataset == "mixed_arxiv_coder":
    with open("mixed_arxiv_coder.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
elif args.dataset == "mixed_arxiv_sharegpt":
    with open("mixed_arxiv_sharegpt.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
