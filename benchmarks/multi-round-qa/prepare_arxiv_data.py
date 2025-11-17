from datasets import load_dataset
import json
# 加载完整数据集
ds = load_dataset("whu9/arxiv_summarization_postprocess")

# 只取20%的训练数据
dataset = ds["train"].train_test_split(test_size=0.8, seed=42)["train"]


# 验证数据量
print(f"原始训练数据量: {len(ds['train'])}")
print(f"采样后训练数据量: {len(dataset)}")
print(f"采样比例: {len(dataset)/len(ds['train'])*100:.1f}%")


# # 转换为目标JSON格式
# converted_data = []
# id = 0
# for item in dataset:
#     if item["source_num_tokens"]<8300 and item["source_num_tokens"] >3200 and item["source_num_tokens"] + item["summary_num_tokens"]<9700 and item["summary_num_tokens"]<500:
#         converted_item = {
#             "id": id,
#             "conversations": [{"from":"human","value":item["source"]},{"from":"gpt","value":item["summary"]}]
#         }
#         converted_data.append(converted_item)
#         id += 1

# 转换为目标JSON格式
converted_data = []
id = 0
for item in dataset:
    if item["source_num_tokens"]<28000  and item["summary_num_tokens"]<400:
        converted_item = {
            "id": id,
            "conversations": [{"from":"human","value":item["source"]},{"from":"gpt","value":item["summary"]}]
        }
        converted_data.append(converted_item)
        id += 1

print(f"处理得到的数据:{len(converted_data)}")
# 写入输出JSON文件
output_path = "./arxiv_preprocess_data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(converted_data, f, indent=2, ensure_ascii=False)

print(f"arxiv数据集预处理文件已成功转换并保存到 {output_path}")