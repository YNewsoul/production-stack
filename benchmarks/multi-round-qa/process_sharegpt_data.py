import argparse
import csv
import json

import tiktoken


def estimate_num_tokens(text: str) -> int:
    enc = tiktoken.encoding_for_model("gpt-4o")
    return len(enc.encode(text, allowed_special={"<|endoftext|>"}))


def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def keep_first_round(conversations):
    first_human = None
    first_gpt = None
    for conv in conversations:
        role = conv.get("from")
        if role == "human" and first_human is None:
            first_human = conv
        elif role == "gpt" and first_human is not None:
            first_gpt = conv
            break
    if first_human and first_gpt:
        return [first_human, first_gpt]
    return []


def process(raw_data):
    total = len(raw_data)
    print(f"原始数据量: {total}")
    # 过滤掉从 gpt 开始的对话
    filtered = [
        d for d in raw_data
        if d.get("conversations") and d["conversations"][0].get("from") != "gpt"
    ]
    print(f"过滤掉以 gpt 开头后的数据量: {len(filtered)}")

    kept = []
    count = 0
    for d in filtered:
        convs = keep_first_round(d["conversations"])
        if not convs:
            continue
        d["conversations"] = convs
        kept.append(d)
        count += 1
        if count % 1000 == 0:
            print(f"已处理 {count}")
    return kept

def statistical_data(dataset, csv_path: str = "sharegpt_token_stats.csv"):
    # 定义人类提问和 gpt 回复的 token 范围
    human_ranges = [(i, i + 500) for i in range(0, 4000, 500)]
    gpt_ranges = [(i, i + 200) for i in range(0, 2000, 200)]

    stats = {}
    for h_start, h_end in human_ranges:
        stats[(h_start, h_end)] = {}
        for g_start, g_end in gpt_ranges:
            stats[(h_start, h_end)][(g_start, g_end)] = 0
        stats[(h_start, h_end)]["total_num"] = 0

    count = 0
    for item in dataset:
        if len(item.get("conversations", [])) < 2:
            continue
        human_tokens = item["conversations"][0].get("num_tokens") or estimate_num_tokens(
            item["conversations"][0]["value"]
        )
        gpt_tokens = item["conversations"][1].get("num_tokens") or estimate_num_tokens(
            item["conversations"][1]["value"]
        )

        human_range = next(((s, e) for s, e in human_ranges if s <= human_tokens < e), None)
        gpt_range = next(((s, e) for s, e in gpt_ranges if s <= gpt_tokens < e), None)

        if human_range and gpt_range:
            stats[human_range][gpt_range] += 1
            stats[human_range]["total_num"] += 1

        count += 1
        if count % 1000 == 0:
            print(f"已统计 {count} 条数据")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        headers = ["human_num_tokens"] + [f"{s}-{e}" for s, e in gpt_ranges] + ["total_num"]
        writer.writerow(headers)
        for h_start, h_end in human_ranges:
            row = [f"{h_start}-{h_end}"]
            for g_start, g_end in gpt_ranges:
                row.append(stats[(h_start, h_end)][(g_start, g_end)])
            row.append(stats[(h_start, h_end)]["total_num"])
            writer.writerow(row)

    print(f"token 范围统计数据已保存到 {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Process ShareGPT data to first round only.")
    parser.add_argument(
        "--input",
        type=str,
        default="ShareGPT_V3_unfiltered_cleaned_split.json",
        help="Path to the raw ShareGPT dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sharegpt_preprocess_data.json",
        help="Path to write the processed dataset.",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default="sharegpt_token_stats.csv",
        help="Path to write token distribution stats.",
    )
    args = parser.parse_args()

    raw_data = load_data(args.input)
    kept = process(raw_data)

    print(f"保留首轮对话后的数据量: {len(kept)}")

    # 统计数据
    statistical_data(kept, args.stats_output)

    # save_data(kept, args.output)
    # print(f"已写入: {args.output}")


if __name__ == "__main__":
    main()
