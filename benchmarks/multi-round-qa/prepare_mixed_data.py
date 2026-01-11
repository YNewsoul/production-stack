import argparse
import json
import os
import random


def load_json(path: str):
    # 读取 JSON 文件并返回列表
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Mix arxiv and sharegpt datasets with random ordering.")
    parser.add_argument("--arxiv", default="arxiv_preprocess_data.json", help="Path to arxiv preprocessed json")
    parser.add_argument("--sharegpt", default="sharegpt_preprocess_data.json", help="Path to sharegpt preprocessed json")
    parser.add_argument("--arxiv-count", type=int, default=4000, help="Number of arxiv samples to use")
    parser.add_argument("--sharegpt-count", type=int, default=10000, help="Number of sharegpt samples to use")
    parser.add_argument(
        "--arxiv-weight", type=float, default=1.0, help="Sampling weight for arxiv when mixing (e.g., 2 for 2:1)"
    )
    parser.add_argument(
        "--sharegpt-weight",type=float,default=2.5, help="Sampling weight for sharegpt when mixing (e.g., 1 for 2:1)",
    )
    parser.add_argument(
        "--output",
        default="mixed_arxiv_sharegpt_v1_preprocess_data.json",
        help="Output path for the mixed dataset",
    )
    parser.add_argument("--seed", type=int, default=36, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    if args.arxiv_weight <= 0 or args.sharegpt_weight <= 0:
        parser.error("arxiv-weight and sharegpt-weight must be positive numbers.")

    arxiv_data = load_json(args.arxiv)
    sharegpt_data = load_json(args.sharegpt)

    # 按需抽样或直接全部使用
    if len(arxiv_data) < args.arxiv_count:
        print(f"[warn] arxiv dataset has only {len(arxiv_data)} items; will use all of them.")
        arxiv_selected = arxiv_data.copy()
    else:
        arxiv_selected = random.sample(arxiv_data, args.arxiv_count)

    if len(sharegpt_data) < args.sharegpt_count:
        print(f"[warn] sharegpt dataset has only {len(sharegpt_data)} items; will use all of them.")
        sharegpt_selected = sharegpt_data.copy()
    else:
        sharegpt_selected = random.sample(sharegpt_data, args.sharegpt_count)

    # 从仍有剩余的列表中随机选择一条加入混合数据
    mixed = []
    arxiv_idx = 0
    sharegpt_idx = 0
    # 预先打乱子集，避免随机选择时的偏置
    random.shuffle(arxiv_selected)
    random.shuffle(sharegpt_selected)

    while arxiv_idx < len(arxiv_selected) or sharegpt_idx < len(sharegpt_selected):
        available_sources = []
        weights = []
        if arxiv_idx < len(arxiv_selected):
            available_sources.append("arxiv")
            weights.append(args.arxiv_weight)
        if sharegpt_idx < len(sharegpt_selected):
            available_sources.append("sharegpt")
            weights.append(args.sharegpt_weight)

        # Weighted pick; falls back to the only available source when one list is exhausted
        pick = random.choices(available_sources, weights=weights, k=1)[0]
        if pick == "arxiv":
            mixed.append(arxiv_selected[arxiv_idx])
            arxiv_idx += 1
        else:
            mixed.append(sharegpt_selected[sharegpt_idx])
            sharegpt_idx += 1

    # 重新分配连续 id，避免重复
    for idx, item in enumerate(mixed):
        item["id"] = idx

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(mixed, f, indent=2, ensure_ascii=False)

    print(f"Mixed dataset created with {len(mixed)} items at {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
