from datasets import load_dataset
import json
import csv


def statistical_data(dataset):
    # 定义范围
    source_ranges = [(i, i+1000) for i in range(1000, 28000, 1000)]
    summary_ranges = [(i, i+50) for i in range(0, 400, 50)]

    # 创建统计字典
    stats = {}
    for source_start, source_end in source_ranges:
        stats[(source_start, source_end)] = {}
        for summary_start, summary_end in summary_ranges:
            stats[(source_start, source_end)][(summary_start, summary_end)] = 0
        stats[(source_start, source_end)]["total_num"] = 0

    # 统计数据
    for item in dataset:
        source_tokens = item["source_num_tokens"]
        summary_tokens = item["summary_num_tokens"]
        
        # 找到对应的source范围
        source_range = None
        for start, end in source_ranges:
            if start <= source_tokens < end:
                source_range = (start, end)
                break
        
        # 找到对应的summary范围
        summary_range = None
        for start, end in summary_ranges:
            if start <= summary_tokens < end:
                summary_range = (start, end)
                break
        
        # 更新统计计数
        if source_range and summary_range:
            stats[source_range][summary_range] += 1
            stats[source_range]["total_num"] += 1

    # 写入CSV文件
    csv_path = "./arxiv_token_stats.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # 写入表头
        headers = ["source_num_tokens"]
        for start, end in summary_ranges:
            headers.append(f"{start}-{end}")
        headers.append("total_num")
        writer.writerow(headers)
        
        # 写入数据
        for source_start, source_end in source_ranges:
            row = [f"{source_start}-{source_end}"]
            for summary_start, summary_end in summary_ranges:
                row.append(stats[(source_start, source_end)][(summary_start, summary_end)])
            row.append(stats[(source_start, source_end)]["total_num"])
            writer.writerow(row)

    print(f"token范围统计数据已保存到 {csv_path}")


# 加载完整数据集
ds = load_dataset("whu9/arxiv_summarization_postprocess")

# 只取20%的训练数据
dataset = ds["train"].train_test_split(test_size=0.93, seed=42)["train"]


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
    if 1000< item["source_num_tokens"] <28000  and item["summary_num_tokens"]<400:
        converted_item = {
            "id": id,
            "conversations": [{"from":"human","value":item["source"]},{"from":"gpt","value":item["summary"]}]
        }
        converted_data.append(converted_item)
        id += 1

print(f"处理后得到的数据:{len(converted_data)}")
# 写入输出JSON文件
output_path = "./arxiv_preprocess_data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(converted_data, f, indent=2, ensure_ascii=False)
# 统计数据
statistical_data(dataset)

print(f"arxiv数据集预处理文件已成功转换并保存到 {output_path}")

