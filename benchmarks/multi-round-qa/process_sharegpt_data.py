import argparse
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
        # # 计算 num_tokens 并更新 num_round
        # for conv in convs:
        #     conv["num_tokens"] = estimate_num_tokens(conv["value"])
        d["conversations"] = convs
        # d["num_round"] = len(convs)
        kept.append(d)
        count += 1
        if count % 1000 == 0:
            print(f"Finished {count}")
    return kept


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
    args = parser.parse_args()

    raw_data = load_data(args.input)
    kept = process(raw_data)

    print(f"保留首轮对话后的数据量: {len(kept)}")

    save_data(kept, args.output)
    print(f"已写入: {args.output}")


if __name__ == "__main__":
    main()
