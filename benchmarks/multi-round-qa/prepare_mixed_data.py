import argparse
import json
import os
import random


def load_json(path: str):
    # 读取 JSON 文件并返回列表
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Mix arxiv and coder datasets with random ordering.")
    parser.add_argument("--arxiv", default="arxiv_preprocess_data.json", help="Path to arxiv preprocessed json")
    parser.add_argument("--coder", default="coder_preprocess_data.json", help="Path to coder preprocessed json")
    parser.add_argument("--arxiv-count", type=int, default=10000, help="Number of arxiv samples to use")
    parser.add_argument("--coder-count", type=int, default=10000, help="Number of coder samples to use")
    parser.add_argument(
        "--output",
        default="mixed_arxiv_coder_preprocess_data.json",
        help="Output path for the mixed dataset",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    arxiv_data = load_json(args.arxiv)
    coder_data = load_json(args.coder)

    # 按需抽样或直接全部使用
    if len(arxiv_data) < args.arxiv_count:
        print(f"[warn] arxiv dataset has only {len(arxiv_data)} items; will use all of them.")
        arxiv_selected = arxiv_data.copy()
    else:
        arxiv_selected = random.sample(arxiv_data, args.arxiv_count)

    if len(coder_data) < args.coder_count:
        print(f"[warn] coder dataset has only {len(coder_data)} items; will use all of them.")
        coder_selected = coder_data.copy()
    else:
        coder_selected = random.sample(coder_data, args.coder_count)

    # 从仍有剩余的列表中随机选择一条加入混合数据
    mixed = []
    arxiv_idx = 0
    coder_idx = 0
    # 预先打乱子集，避免随机选择时的偏置
    random.shuffle(arxiv_selected)
    random.shuffle(coder_selected)

    while arxiv_idx < len(arxiv_selected) or coder_idx < len(coder_selected):
        available_sources = []
        if arxiv_idx < len(arxiv_selected):
            available_sources.append("arxiv")
        if coder_idx < len(coder_selected):
            available_sources.append("coder")

        pick = random.choice(available_sources)
        if pick == "arxiv":
            mixed.append(arxiv_selected[arxiv_idx])
            arxiv_idx += 1
        else:
            mixed.append(coder_selected[coder_idx])
            coder_idx += 1

    # 重新分配连续 id，避免重复
    for idx, item in enumerate(mixed):
        item["id"] = idx

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(mixed, f, indent=2, ensure_ascii=False)

    print(f"Mixed dataset created with {len(mixed)} items at {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
