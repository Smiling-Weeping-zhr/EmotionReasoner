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
import pandas as pd
import warnings

# 屏蔽 transformers 的 INFO 和 DEBUG 日志
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")
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
from typing import Optional, Dict
from model.Qwen_omni.train_rl.utils import extract_answer, parse_label_string, extract_think, extract_score, offline_assessment, coarse_assessment

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from model.Qwen_omni.train_rl.trainer import Qwen2_5OmniVADTWGRPOTrainer
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
    model_alpha: Optional[float] = field(
        default=1.4,
        metadata={"help": "Weighting factor for combining rewards"},
    )
    vad_dict: Optional[Dict[str, list]] = field(
        default=None,
        metadata={"help": "Dictionary containing NRC-VAD weights for words"}
    )

format_mapping = read_format2raws()          # level3 -> level2
raw_mapping = read_candidate_synonym_merge() # level2 -> level1

_FORMAT_PATTERN = re.compile(
    r'^(?!.*<think>.*<think>)'       
    r'(?!.*<answer>.*<answer>)'     
    r'<think>[\s\S]+?</think>\s*<answer>[\s\S]+?</answer>$'
)

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
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH") # "./experiment/train_qwen_omni/logs/debug_log.txt"
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} -------------\n")
                f.write(f"Prediction Text: {pred}\n")
                f.write(f"Answer_gt: {ans}\n")
    
    for pred, gt in zip(pred_ans_list, gt_ans_list):
        
        pred_txt = " ".join(pred)
        gt_txt = " ".join(gt)
        
        if len(pred) > len(gt)*2 + 1 or len(pred_txt) > 3*len(gt_txt) + 5: # To control the length of the answer
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
                # f.write(f"Prediction: {predictions}\n")
                f.write(f"Answer_gt: {gt}\n")
                
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

def offline_think_reward(predictions, ref_model, processor, generation_config, answer_gt=None, **kwargs):
    outputs = [c[0]["content"] for c in predictions]
    rewards = []
    for pred in outputs:
        pred_think = extract_think(pred)
        if pred_think == "":
            rewards.append(0.1)
            continue
        
        score = None
        cnt = 0
        while score is None:
            assessment = offline_assessment(pred=pred_think, ref_model=ref_model, processor=processor, generation_config=generation_config)
            score = extract_score(assessment)
            cnt += 1
            if cnt > 5:
                print("Error in online assessment:", pred_think)
                score = 0.0
                break
        
        if score > 10 or score < 0:
            print("Error score:", score)
            score = 0.1
        score = score / 10.0
        rewards.append(score)
    
    return rewards

def offline_EW_Fscore(answer_gt, predictions, ref_model, processor, generation_config, **kwargs):
    """
    计算思考部分的离线评分 + 情感轮的 Fscore
    """
    offline_think_rewards = offline_think_reward(predictions, ref_model, processor, generation_config, **kwargs)
    ov_mer_rewards = ov_mer_reward(answer_gt=answer_gt, predictions=predictions, **kwargs)

    combined_score = [ (2*a*b/(a+b) if (a+b) != 0 else 0.0)
        for a, b in zip(ov_mer_rewards, offline_think_rewards)]
    
    return combined_score

def sentiment_coarse_reward(answer_gt, predictions, ref_model, processor, generation_config, **kwargs):
    outputs = [c[0]["content"] for c in predictions]
    rewards, pred_ans_list, gt_ans_list = [], [], []
    sentiment_coarse_score = 0.4
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
        
        pred_txt = " ".join(pred)
        gt_txt = " ".join(gt)
        
        if len(pred) < min(3, len(gt)/2+1) or len(pred) > len(gt)*2 + 1 or len(pred_txt) > 3*len(gt_txt) + 5: # To control the length of the answer
            rewards.append(0)
            continue
        
        pred_sentiment = coarse_assessment(reason=pred, ref_model=ref_model, processor=processor, generation_config=generation_config)
        ans_sentiment = coarse_assessment(reason=gt, ref_model=ref_model, processor=processor, generation_config=generation_config)
        # ipdb.set_trace()
        if pred_sentiment == ans_sentiment:
            rewards.append(sentiment_coarse_score)
        else:
            rewards.append(0)
    return rewards

reward_funcs_registry = {
    "EmotionWheel": ov_mer_reward,
    "think": think_reward,
    "offline": offline_think_reward,
    "format": format_reward,
    "offline_EW_Fscore": offline_EW_Fscore,
    "coarse": sentiment_coarse_reward,
}

def main(script_args, training_args, model_args, cfg, processor):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    nrc_df = pd.read_csv(cfg.dataset.NRC_VAD)
    vad_dict = {row.term: [row.valence, row.arousal, row.dominance] for _, row in nrc_df.iterrows()}
    
    dataset = MERCaptionPlus_Dataset(cfg=cfg, processor=processor)
    # 选择合适的训练器类
    if script_args.model_type.lower() == "omni":
        print("Using Qwen2.5OmniGRPOTrainer for Qwen Omni model")
        trainer_cls = Qwen2_5OmniVADTWGRPOTrainer

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
        "alpha": script_args.model_alpha,
        "vad_dict": vad_dict,
    }

    # 对于Omni模型添加额外参数
    if script_args.model_type.lower() == "omni":
        trainer_kwargs["use_audio_in_video"] = script_args.use_audio_in_video
    
    trainer = trainer_cls(**trainer_kwargs)
    try:
        if training_args.resume_from_checkpoint is not None:
            wandb.init(
                project="huggingface",
                id="0o6qxx33",    
                resume="allow"
            )
            checkpoint = training_args.resume_from_checkpoint
            trainer.train(resume_from_checkpoint=checkpoint)
        else:
            trainer.train()
    except Exception as e:
        print("⚠️ Training crashed with error:", str(e))
        # 立即保存当前模型和优化器/调度器状态
        save_dir = training_args.output_dir
        crash_dir = os.path.join(save_dir, "crash_save")
        os.makedirs(crash_dir, exist_ok=True)
        trainer.save_model(crash_dir)
        trainer.state.save_to_json(os.path.join(crash_dir, "trainer_state.json"))
        torch.save(trainer.optimizer.state_dict(), os.path.join(crash_dir, "optimizer.pt"))
        torch.save(trainer.lr_scheduler.state_dict(), os.path.join(crash_dir, "scheduler.pt"))
        # raise  # 再次抛出错误，方便调试
    
    # Save and push to hub
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(training_args.output_dir)
    trainer.save_model(save_dir)

    trainer.state.save_to_json(os.path.join(save_dir, "trainer_state.json"))
    torch.save(trainer.optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
    torch.save(trainer.lr_scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))


if __name__ == "__main__":
    directory = './experiment/train_qwen_omni'
    cfg = OmegaConf.load(os.path.join(directory, 'hparams.yaml'))
    model_path = cfg.model.ckpt
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args, cfg, processor)