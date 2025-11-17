import json
import os
from transformers import AutoTokenizer

def estimate_num_tokens(text: str) -> int:
    if not hasattr(estimate_num_tokens, "tokenizer"):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        estimate_num_tokens.tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-V3.1"
        )
    return len(estimate_num_tokens.tokenizer.tokenize(text))

# 文件路径
file_path = '/home/paperspace/cys/projects/fork/production-stack/benchmarks/multi-round-qa/sharegpt.json'
output_path = '/home/paperspace/cys/projects/fork/production-stack/benchmarks/multi-round-qa/sharegpt_filtered.json'

try:
    # 读取原始文件
    print(f"正在读取文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查数据类型
    if not isinstance(data, list):
        print(f"错误: 数据不是预期的列表格式，而是: {type(data).__name__}")
        # 如果是字典，尝试将其转换为列表（假设字典值是列表）
        if isinstance(data, dict):
            print("尝试从字典中提取列表...")
            for key, value in data.items():
                if isinstance(value, list):
                    data = value
                    print(f"成功提取键 '{key}' 对应的列表")
                    break
            else:
                print("错误: 无法从字典中提取有效列表")
                exit(1)
        else:
            exit(1)
    
    # 统计原始数据量
    original_count = len(data)
    print(f"原始数据项数量: {original_count}")
    
    # 过滤数据
    filtered_data = []
    excluded_count = 0
    
    for i, item in enumerate(data):
        # 检查必要的键是否存在
        if 'num_round' not in item or 'average_human_token' not in item:
            print(f"警告: 第 {i+1} 项缺少必要的键，跳过此项")
            excluded_count += 1
            continue
        
        try:
            # 计算条件值
            if item['num_round'] % 2==0 and  item['num_round'] !=0:
                if estimate_num_tokens(item['conversations'][0]['value']) >1000:
                # product = item['num_round'] * item['average_human_token']
                # # 应用过滤条件
                # if product >= 1000:
                    filtered_data.append(item)
                else:
                    excluded_count += 1
        except Exception as e:
            print(f"处理第 {i+1} 项时出错: {str(e)}")
            excluded_count += 1
    
    # 统计结果
    filtered_count = len(filtered_data)
    print(f"过滤后数据项数量: {filtered_count}")
    print(f"剔除的数据项数量: {excluded_count}")
    print(f"保留比例: {(filtered_count/original_count)*100:.2f}%")
    
    # 保存过滤后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print(f"过滤后的数据已保存至: {output_path}")
    
except Exception as e:
    print(f"处理文件时发生错误: {str(e)}")