# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import wandb
import warnings

# 屏蔽 transformers 的 INFO 和 DEBUG 日志
logging.getLogger().setLevel(logging.ERROR)
# os.environ["WANDB_DISABLED"] = "true"

import ipdb
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import torch
import random
from omegaconf import OmegaConf
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from model.Qwen_omni.train_rl.trainer import Qwen2_5OmniGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from model.Qwen_omni.datasets.mercaptionplus import MERCaptionPlus_Dataset
from datasets import Dataset, DatasetDict

from rouge_score import rouge_scorer
import numpy as np


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    len_control: Optional[bool] = field(
        default=False,
        metadata={"help": "whether using length reward"},
    )
    weighted_reward: Optional[bool] = field(
        default=False,
        metadata={"help": "whether using weighted reward"},
    )
    use_audio_in_video: Optional[bool] = field(
        default=True,
        metadata={"help": "whether to use audio in video for Omni model"},
    )
    model_type: Optional[str] = field(
        default="vl",
        metadata={"help": "Model type to use: 'vl' for Vision-Language models, 'omni' for Omni models"},
    )

# === 预编译正则，提高循环内多次调用效率 ===
ANS_TAG  = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)
THINK_TAG = re.compile(r"<think>\s*(.*?)\s*</think>", re.S)

_FORMAT_PATTERN = re.compile(
    r'^(?!.*<think>.*<think>)'       
    r'(?!.*<answer>.*<answer>)'     
    r'<think>[\s\S]+?</think>\s*<answer>[\s\S]+?</answer>$'
)


def extract_answer(text: str) -> str:
    """和你原来 extract_answer 完全等价"""
    m = ANS_TAG.search(text)
    return m.group(1).strip() if m else ""

def extract_think(text: str) -> str:
    """提取 <think>...</think> 标签中的内容"""
    m = THINK_TAG.search(text)
    return m.group(1).strip() if m else ""

def accuracy_reward(answer_gt, predictions, **kwargs):
    """
    Args: 

    完整覆盖你原来所有分支，返回值 均在 [0.0, 1.0] 之间。
    """
    outputs = [c[0]["content"] for c in predictions]
    rewards = []
    # ipdb.set_trace()
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for pred, ans in zip(outputs, answer_gt):
        pred_ans = extract_answer(pred)
        gt_ans = extract_answer(ans)
        reward = 0
        
        # Score2 -- 计算OV
        labels = [label.strip().lower() for label in gt_ans.strip("[]").split(",")]  # ['concern', 'pessimism']
        pred_ans = [label.strip().lower() for label in pred_ans.split(",")]
        matched = [label for label in labels if label.lower() in pred_ans] # Count the number of words

        if len(pred_ans) > len(labels)*2 + 1:
            scores2 = 0
        else: # Control the format of the answer
            scores2 = len(pred_ans) / len(labels)
        reward += scores2
        
        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH") # "./experiment/train_qwen_omni/logs/debug_log.txt"
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"OV reward: {scores2}\n")
                f.write(f"Prediction: {pred}\n")
                f.write(f"Answer_gt: {ans}\n")
            
    return rewards

# def accuracy_reward1(answer_gt, predictions, **kwargs):
#     """
#     完整覆盖你原来所有分支，返回值 均在 [0.0, 1.0] 之间。
#     """
#     outputs = [c[0]["content"] for c in predictions]
#     rewards = []
#     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

#     # 预初始化 Rouge 计算器
#     rouge = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)

#     for pred, ans in zip(outputs, answer_gt):
#         pred_think = extract_think(pred)
#         gt_think   = extract_think(ans)
#         pred_ans = extract_answer(pred)
#         gt_ans = extract_answer(ans)
#         reward = 0
        
#         # Score1 -- 取 rougeL f-measure 的平均（和你 compute_rouge_score 类似）
#         scores1 = rouge.score(gt_think, pred_think)
#         # # 平均 rouge1/2/L 或者单独用 rougeL
#         scores1 += max(0.0, min(1.0, scores1['rougeL'].fmeasure))
#         reward += scores1
        
#         # Score2 -- 计算OV
#         labels = gt_ans.strip("[]").replace(" ", "").split(",")  # ['concern', 'pessimism']
#         pred_ans = pred_ans.lower()
#         matched = [label for label in labels if label.lower() in pred_ans] # Count the number of words
#         scores2 = len(matched) / len(labels)
#         reward += scores2
        
#         # Penalty term: To control the length of pred_ans, avoid the more you answer, the more scores you get
#         max_words = max(1, round(len(labels)*0.5)+1)

#         # 将 pred_ans 拆成单词（按空格分隔）
#         pred_words = pred_ans.strip().split()
#         if pred_words <= max_words:
#             reward += 0.2
            
#         rewards.append(reward)

#         if os.getenv("DEBUG_MODE") == "true":
#             log_path = os.getenv("LOG_PATH") # "./experiment/train_qwen_omni/logs/debug_log.txt"
#             # local_rank = int(os.getenv("LOCAL_RANK", 0))
#             with open(log_path, "a", encoding="utf-8") as f:
#                 f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
#                 f.write(f"OV reward: {scores2}")
#                 f.write(f"Prediction: {pred}\n")
#                 f.write(f"Answer_gt: {ans}\n")
            
#     return rewards

# def format_reward(predictions, **kwargs):
#     rewards = []
#     for c in predictions:
#         text = c[0]["content"].strip()          # <-- remove surrounding whitespace
#         rewards.append(
#             1.0 if _FORMAT_PATTERN.fullmatch(text) else 0.0
#         )
#     return rewards

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

def main(script_args, training_args, model_args, cfg, processor):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    
    dataset = MERCaptionPlus_Dataset(cfg=cfg, processor=processor)
    # 选择合适的训练器类
    if script_args.model_type.lower() == "omni":
        print("Using Qwen2.5OmniGRPOTrainer for Qwen Omni model")
        trainer_cls = Qwen2_5OmniGRPOTrainer

    print("Using trainer class: ", trainer_cls)

    # 初始化GRPO训练器
    trainer_kwargs = {
        "model": model_args.model_name_or_path,
        "reward_funcs": reward_funcs,
        "args": training_args,
        "script_args": script_args,
        "train_dataset": dataset,
        "eval_dataset": None,
        "peft_config": get_peft_config(model_args),
        "attn_implementation": model_args.attn_implementation,
        "max_pixels": script_args.max_pixels,
        "min_pixels": script_args.min_pixels,
    }
    
    # 对于Omni模型添加额外参数
    if script_args.model_type.lower() == "omni":
        trainer_kwargs["use_audio_in_video"] = script_args.use_audio_in_video
    
    trainer = trainer_cls(**trainer_kwargs)
    
    if training_args.resume_from_checkpoint is not None:
        wandb.init(
            project="huggingface",
            id="0zze17a4",    # 保持与之前相同
            resume="allow"
        )
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    save_dir = cfg.run.train.save_path
    # Save and push to hub
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(training_args.output_dir)
    trainer.save_model(save_dir)

    trainer.state.save_to_json(os.path.join(save_dir, "trainer_state.json"))
    torch.save(trainer.optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
    torch.save(trainer.lr_scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    directory = '/root/autodl-tmp/EMER/experiment/train_qwen_omni'
    cfg = OmegaConf.load(os.path.join(directory, 'hparams.yaml'))
    model_path = cfg.model.ckpt
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
#     if True:
#         wandb.init(
#         id="2xy9klmp",  # 上一个 run 的 ID
#     resume="must",  # 强制恢复
#     project="your-project-name",
#     name="your-run-name"  # 可选：继续写入原 run 名称
# )
    main(script_args, training_args, model_args, cfg, processor)