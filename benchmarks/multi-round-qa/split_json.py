import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--num', type=int)
args = parser.parse_args()

current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)

print(current_script_dir)
# 定义文件路径
input_file = f'{current_script_dir}/{args.dataset}.json'
output_dir = current_script_dir

# # 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 读取输入文件
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查数据类型
    if not isinstance(data, list):
        # 如果不是列表，将其包装成列表（假设是单一对象）
        data = [data]
    
    # 计算每个部分的大小
    total_items = len(data)
    part_size = total_items // args.num
    remainder = total_items % args.num
    
    # 分割数据并保存
    start_idx = 0
    for i in range(args.num):
        # 处理余数，确保前remainder个文件多一个元素
        current_part_size = part_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_part_size
        
        # 获取当前部分的数据
        part_data = data[start_idx:end_idx]
        
        # 保存到新文件
        output_file = os.path.join(output_dir, f'{args.dataset}_{i+1}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(part_data, f, ensure_ascii=False, indent=2)
        
        print(f'已保存 {output_file}，包含 {len(part_data)} 个元素')
        
        start_idx = end_idx
    
    print(f'分割完成！原文件共有 {total_items} 个元素，已平均分为 {args.num} 个部分。')
    
except Exception as e:
    print(f'处理文件时出错：{str(e)}')
    # 尝试获取更多关于文件的信息
    try:
        file_size = os.path.getsize(input_file)
        print(f'文件大小：{file_size} 字节')
        
        # 尝试读取文件的前几行检查格式
        with open(input_file, 'r', encoding='utf-8') as f:
            first_lines = [next(f) for _ in range(5)]
            print('文件前5行内容：')
            for line in first_lines:
                print(line.strip())
    except Exception as e_info:
        print(f'获取文件信息时出错：{str(e_info)}')