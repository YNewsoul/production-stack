
import json
file_path = "/home/paperspace/cys/projects/fork/production-stack/benchmarks/multi-round-qa/sharegpt.json"

num = 0
with open(file_path, 'r', encoding='utf-8') as f:
    # 解析JSON文件内容为Python字典/列表
    data = json.load(f)
    for item in data:
        if item["num_round"] >= 8 and  item["num_round"]%2==0:
            num += 1
print(f"num:{num}")