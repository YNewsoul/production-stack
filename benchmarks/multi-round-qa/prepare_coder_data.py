import json
import csv
import tiktoken

from datasets import load_dataset
'''
    处理Coder数据集,使其能被脚本使用
'''

def estimate_num_tokens(text:str) ->int:
    enc = tiktoken.encoding_for_model("gpt-4o")
    return len(enc.encode(text))

def statistical_data(dataset):
    # 定义范围
    source_ranges = [(i, i+500) for i in range(0, 1000, 500)]
    summary_ranges = [(i, i+50) for i in range(0, 700, 50)]

    # 创建统计字典
    stats = {}
    for source_start, source_end in source_ranges:
        stats[(source_start, source_end)] = {}
        for summary_start, summary_end in summary_ranges:
            stats[(source_start, source_end)][(summary_start, summary_end)] = 0
        stats[(source_start, source_end)]["total_num"] = 0

    # 统计数据
    count = 0
    for item in dataset:
        
        source_tokens = estimate_num_tokens(item["conversations"][0]['value'])
        summary_tokens = estimate_num_tokens(item["conversations"][1]['value'])
        
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
        count += 1
        if count %1000 == 0:
            print(f"成功统计{count}个数据")

    # 写入CSV文件
    csv_path = "./coder_token_stats.csv"
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


dataset = load_dataset("ajibawa-2023/Python-Code-23k-ShareGPT", split="train")

# 转换为目标JSON格式
converted_data = []
for item in dataset:
    converted_item = {
        "id": item["id"],
        "conversations": item["conversations"]
    }
    converted_data.append(converted_item)

# 写入输出JSON文件
output_path = "./coder_preprocess_data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(converted_data, f, indent=2, ensure_ascii=False)

print(f"coder数据集预处理文件已成功转换并保存到 {output_path}")

# 数据分析
statistical_data(dataset)
