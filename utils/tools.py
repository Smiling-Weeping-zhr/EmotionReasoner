import hashlib
import os
import pathlib
import random
import unittest

import numpy as np
import torch
import glob
import json
import os
from typing_extensions import Never

def guard_never(x: Never) -> Never:
    raise RuntimeError(f"unexpected branch: {x}")


def subdir_files_count(path: pathlib.Path) -> int:
    count = 1
    if path.is_dir():
        for ch in path.iterdir():
            count += subdir_files_count(ch)
    return count

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.enabled = False  # type: ignore
    return


def set_iterative_seed(master_seed: int, key: str) -> None:
    """Set seed for reproducibility. Since the entire process needs to be
    interruptable, we need to set seed for each iteration after restoring."""

    patt = f"[seed={master_seed},key={key}]"
    seed = hashlib.sha256(patt.encode("utf-8", "ignore")).digest()
    new_seed = int.from_bytes(seed[:8], byteorder="big") % 2**32
    set_seed(new_seed)
    return

def load_single_json_file(directory: str):
    # 查找以 .json 结尾的文件（只查该目录，不递归）
    json_files = glob.glob(os.path.join(directory, "*.json"))
    
    if len(json_files) == 0:
        raise FileNotFoundError(f"No JSON files found in {directory}")
    elif len(json_files) > 1:
        raise ValueError(f"Multiple JSON files found in {directory}: {json_files}")
    
    # 读取唯一的 JSON 文件
    json_path = json_files[0]
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded JSON file: {json_path}")
    return data

class ToolsUtilsTest(unittest.TestCase):
    def test_set_iterative_seed(self):
        master = 1027

        def get(key: str) -> int:
            set_iterative_seed(master, key)
            return random.randint(0, 1000)

        a = get("stage-a")
        b = get("stage-b")
        c = get("stage-c")
        self.assertEqual(get("stage-c"), c)
        self.assertEqual(get("stage-a"), a)
        self.assertEqual(get("stage-b"), b)

    pass
