import random
from pathlib import Path

MU = 120          # 期望值
SIGMA = 20        # 标准差，可按需微调
MIN_VAL = 40
MAX_VAL = 200
COUNT = 20000

def generate_numbers():
    data = []
    for _ in range(COUNT):
        # 拒绝采样确保范围内的值
        while True:
            x = random.gauss(MU, SIGMA)
            if MIN_VAL <= x <= MAX_VAL:
                data.append(int(round(x, 4)))  # 保留4位小数，可按需调整
                break
    return data

def save_to_file(path: str):
    path = Path(path)
    nums = generate_numbers()
    # 保存为一行一个数字，便于读取
    path.write_text("\n".join(map(str, nums)), encoding="utf-8")

def read_numbers(path: str):
    path = Path(path)
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

if __name__ == "__main__":
    outfile = "numbers.txt"
    save_to_file(outfile)
    loaded = read_numbers(outfile)
    print(f"生成 {len(loaded)} 条，示例前5个: {loaded[:5]}")
