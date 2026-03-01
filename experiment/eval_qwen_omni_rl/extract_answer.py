import os
import shutil
import pickle
import re

def extract_answer(text: str) -> str | None:
    """
    从形如 <answer>...</answer> 中提取内容，如果没有匹配则返回 None
    """
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

# 原始根目录
source_root = "/root/autodl-tmp/EMER/experiment/eval_qwen_omni_rl/qwen-omni-peft-vadtw-offline-two/checkpoint-175"
# 目标根目录
target_root = "/root/autodl-tmp/EMER/experiment/eval_qwen_omni_rl/output"

# 遍历 source_root 下所有子文件夹
for folder in os.listdir(source_root):
    folder_path = os.path.join(source_root, folder)
    if os.path.isdir(folder_path):
        pkl_path = os.path.join(folder_path, f"{folder}.pkl")
        if os.path.exists(pkl_path):
            # 读取 .pkl 文件
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            # Extract the <answer></answer> from the pkl files
            for key, value in data.items():
                data[key] = extract_answer(value)
            # Save the target into the ./experiment/output
            
            # 构建目标目录和路径
            target_dir = os.path.join(target_root, folder)
            os.makedirs(target_dir, exist_ok=True)
            target_pkl_path = os.path.join(target_dir, f"{folder}.pkl")

            # 保存 pkl 文件
            with open(target_pkl_path, 'wb') as f:
                pickle.dump(data, f)

            print(f"已复制: {folder}/{folder}.pkl")
