from datasets import load_dataset
import json
import os
from transformers import AutoTokenizer

from datasets import load_dataset, concatenate_datasets

def estimate_num_tokens(text: str) -> int:
    if not hasattr(estimate_num_tokens, "tokenizer"):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        estimate_num_tokens.tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-V3.1"
        )
    return len(estimate_num_tokens.tokenizer.tokenize(text))

ds_1 = load_dataset("GetSoloTech/Code-Reasoning","")

dataset = concatenate_datasets([
    ds_1['cpp'],
    ds_1['python'],
])

# 验证数据量
print(f"原始训练数据量: {len(dataset)}")


# 转换为目标JSON格式
converted_data = []
id = 0
token_num = 0
out_put = 0
for item in dataset:
    token_num += estimate_num_tokens(item["question"])
    out_put += estimate_num_tokens(item["solution"])
    if estimate_num_tokens(item["question"]) > 500 :
        converted_item = {
            "id": id,
            "conversations": [{"from":"human","value":item["question"]},{"from":"gpt","value":item["solution"]}]
        }
        converted_data.append(converted_item)
        id += 1
    
print(f"处理得到的数据:{len(converted_data)}")
print(f"token_num:{token_num/len(dataset)}")
# 写入输出JSON文件
output_path = "./reasoning_preprocess_data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(converted_data, f, indent=2, ensure_ascii=False)

print(f"humaneval数据集预处理文件已成功转换并保存到 {output_path}")