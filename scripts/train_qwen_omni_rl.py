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
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")
os.environ["WANDB_DISABLED"] = "true"

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

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from model.Qwen_omni.train_rl.trainer import Qwen2_5OmniGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from model.Qwen_omni.datasets.mercaptionplus import MERCaptionPlus_Dataset
from utils.evaluation.wheel import func_map_label_to_synonym, read_format2raws, read_candidate_synonym_merge, func_get_wheel_cluster
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
        default_factory=lambda: ["OV", "format", "think"],
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
    use_sft_base_model: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use the SFT model as the base model"},
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

format_mapping = read_format2raws()          # level3 -> level2
raw_mapping = read_candidate_synonym_merge() # level2 -> level1

def extract_answer(text: str) -> str:
    """和你原来 extract_answer 完全等价"""
    m = ANS_TAG.search(text)
    return m.group(1).strip() if m else ""

def extract_think(text: str) -> str:
    """提取 <think>...</think> 标签中的内容"""
    m = THINK_TAG.search(text)
    return m.group(1).strip() if m else ""

def parse_label_string(label_str):
    # 去掉 []
    items = label_str.strip("[]").split(",")
    # 去掉首尾空格和引号
    return [x.strip().strip("'").strip('"').lower() for x in items if x.strip()]

def ov_mer_reward(answer_gt, predictions, **kwargs):
    """
    Args: 

    完整覆盖你原来所有分支，返回值 均在 [0.0, 1.0] 之间。
    """
    candidate_metrics = ['case3_wheel1_level1', 'case3_wheel2_level1', 'case3_wheel3_level1', 'case3_wheel4_level1', 'case3_wheel5_level1']
    outputs = [c[0]["content"] for c in predictions]
    rewards, pred_ans_list, gt_ans_list = [], [], []
    
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for pred, ans in zip(outputs, answer_gt):
        # kk = '<answer>\'apathetic\', \'calm\', \'bored\'</answer>'
        pred_ans = extract_answer(pred)
        gt_ans = extract_answer(ans)
        
        # Score2 -- 计算OV
        # gt_ans = [label.strip().lower() for label in gt_ans.strip("[]").split(",")]  # samplenew3_00100477 error format data
        gt_ans = parse_label_string(gt_ans)
        pred_ans = [label.strip().lower() for label in pred_ans.split(",")]
        
        pred_ans_list.append(pred_ans)
        gt_ans_list.append(gt_ans)
    
    for pred, gt in zip(pred_ans_list, gt_ans_list):
        
        if len(pred) > len(gt)*2 + 1: # To control the length of the answer
            rewards.append(0)
            continue
        
        accuracy, recall = [], []
        for metric in candidate_metrics:
            if metric.startswith('case3'): # level1 -> cluster center
                _, wheelname, levelname = metric.split('_')
                wheel_map = func_get_wheel_cluster(wheelname, levelname)
            else:
                wheel_map = None
            
            gt_set = set(func_map_label_to_synonym(gt, format_mapping, raw_mapping, wheel_map, metric))
            pred_set = set(func_map_label_to_synonym(pred, format_mapping, raw_mapping, wheel_map, metric))

            if len(gt_set) == 0: continue
            if len(pred_set) == 0:
                accuracy.append(0)
                recall.append(0)
            else:
                accuracy.append(len(gt_set & pred_set)/len(pred_set))
                recall.append(len(gt_set & pred_set)/len(gt_set))

        if accuracy == [] or recall == []: # If np.mean([]), then nan
            rewards.append(0)
            continue
        avg_precision, avg_recall = np.mean(accuracy), np.mean(recall)
        if avg_precision + avg_recall == 0:
            fscore = 0.0
        else:
            fscore = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        rewards.append(fscore)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH") # "./experiment/train_qwen_omni/logs/debug_log.txt"
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {fscore} -------------\n")
                f.write(f"OV reward: {fscore}\n")
                f.write(f"Prediction: {pred}\n")
                f.write(f"Answer_gt: {ans}\n")
            
    return rewards


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
        labels = parse_label_string(gt_ans)  # ['concern', 'pessimism'] # samplenew3_00100477 error format data
        pred_ans = [label.strip().lower() for label in pred_ans.split(",")]
        matched = [label for label in labels if label.lower() in pred_ans] # Count the number of words

        if len(pred_ans) > len(labels)*2 + 1:
            scores2 = 0
        else: # Control the format of the answer
            scores2 = len(matched) / len(labels)
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

def think_reward(answer_gt, predictions, **kwargs):
    """
    完整覆盖你原来所有分支，返回值 均在 [0.0, 1.0] 之间。
    """
    outputs = [c[0]["content"] for c in predictions]
    rewards = []

    # 预初始化 Rouge 计算器
    rouge = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)

    for pred, ans in zip(outputs, answer_gt):
        pred_think = extract_think(pred)
        gt_think   = extract_think(ans)
        
        # Score1 -- 取 rougeL f-measure 的平均（和你 compute_rouge_score 类似）
        scores1 = rouge.score(gt_think, pred_think)
        # # 平均 rouge1/2/L 或者单独用 rougeL
        f1 = scores1['rouge1'].fmeasure
        f2 = scores1['rouge2'].fmeasure
        f3 = scores1['rougeL'].fmeasure
        score_mean = (f1 + f2 + f3) / 3
        
        reward = max(0.0, min(1.0, score_mean))
        rewards.append(reward)
            
    return rewards

def format_reward(predictions, **kwargs):
    rewards = []
    for c in predictions:
        text = c[0]["content"].strip()          # <-- remove surrounding whitespace
        rewards.append(
            1.0 if _FORMAT_PATTERN.fullmatch(text) else 0.0
        )
    return rewards

reward_funcs_registry = {
    "OV": accuracy_reward,
    "EmotionWheel": ov_mer_reward,
    "think": think_reward,
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
        "use_sft_base_model": script_args.use_sft_base_model,
        "sft_base_model_path": cfg.model.peft_ckpt if script_args.use_sft_base_model else None,
    }
    
    # 对于Omni模型添加额外参数
    if script_args.model_type.lower() == "omni":
        trainer_kwargs["use_audio_in_video"] = script_args.use_audio_in_video
    
    trainer = trainer_cls(**trainer_kwargs)
    
    try:
        if training_args.resume_from_checkpoint is not None:
            wandb.init(
                project="huggingface",
                id="n3xwxyvv",    
                resume="allow"
            )
            checkpoint = training_args.resume_from_checkpoint
            trainer.train(resume_from_checkpoint=checkpoint)
        else:
            trainer.train()
    except Exception as e:
        print("⚠️ Training crashed with error:", str(e))
        # 立即保存当前模型和优化器/调度器状态
        crash_dir = os.path.join(save_dir, "crash_save")
        os.makedirs(crash_dir, exist_ok=True)
        trainer.save_model(crash_dir)
        trainer.state.save_to_json(os.path.join(crash_dir, "trainer_state.json"))
        torch.save(trainer.optimizer.state_dict(), os.path.join(crash_dir, "optimizer.pt"))
        torch.save(trainer.lr_scheduler.state_dict(), os.path.join(crash_dir, "scheduler.pt"))
        raise  # 再次抛出错误，方便调试
    

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
    directory = './experiment/train_qwen_omni'
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