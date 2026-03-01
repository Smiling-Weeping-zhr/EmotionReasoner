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
import math
import re
import pandas as pd
import gc
from collections import defaultdict
from typing import Any, Callable, Optional, Union
import random

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AutoConfig,  # 添加这一行
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
import ipdb
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from model.Qwen_omni.train_rl.utils import extract_answer, parse_label_string
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

from qwen_omni_utils import process_mm_info, process_audio_info, process_vision_info

import copy

if is_peft_available():
    from peft import PeftConfig, get_peft_model, PeftModel
    

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class Qwen2_5OmniVADTWGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method with Qwen2.5Omni model support.
    This extends the GRPO algorithm to handle multimodal inputs including audio and video.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        script_args = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        use_audio_in_video: bool = True,
        use_sft_base_model: bool = False,
        sft_base_model_path: str = None,
        alpha: float = 1.4,
        vad_dict: dict = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # 在 Qwen2_5OmniGRPOTrainer.__init__ 方法中的模型初始化部分
            if "Qwen2.5-Omni" in model_id:
                # 为 Qwen2.5-Omni 模型创建一个新的 model_init_kwargs 字典
                omni_model_init_kwargs = model_init_kwargs.copy()
                if "use_cache" in omni_model_init_kwargs: # Enable gradient checkpoint
                    del omni_model_init_kwargs["use_cache"]
                # 加载配置并修改
                config = AutoConfig.from_pretrained(model_id)
                config.enable_audio_output = False  # 在配置中禁用音频输出
                print(f'Model ID: {model_id}')
                
                # 使用修改后的配置加载模型
                model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    config=config,
                    **omni_model_init_kwargs
                )
                # 确保 talker 被禁用
                model.disable_talker()
                print("Talker component has been disabled")
            else:
                raise ValueError(f"Unsupported model ID for Qwen2.5OmniGRPOTrainer: {model_id}")
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            if use_sft_base_model:
                model = PeftModel.from_pretrained(model, sft_base_model_path, is_local=True)
                for name, param in model.named_parameters():
                    if "lora" in name.lower():
                        param.requires_grad = True
                        
                print(f"You are using the sft base model {sft_base_model_path}.")
            else:
                model = get_peft_model(model, peft_config)
                print("Initialize the peft model.")
                
            model.gradient_checkpointing_enable()
            # freeze audio_tower
            for param in model.thinker.audio_tower.parameters():
                param.requires_grad = False
            # freeze visual
            for param in model.thinker.visual.parameters():
                param.requires_grad = False
            model.print_trainable_parameters()
            
        is_deepspeed = False
        # 在创建参考模型部分
        # if is_deepspeed_zero3_enabled():
        if is_deepspeed:
            if "Qwen2.5-Omni" in model_id:
                # 同样处理参考模型
                omni_model_init_kwargs = model_init_kwargs.copy()
                if "use_cache" in omni_model_init_kwargs: # Enable gradient checkpointing
                    del omni_model_init_kwargs["use_cache"]
                # model.base_model.thinker BUG
                config = AutoConfig.from_pretrained(model_id)
                config.enable_audio_output = False  # 在配置中禁用音频输出
                
                self.ref_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    model_id, 
                    config=config,
                    **omni_model_init_kwargs
                )
                self.ref_model.disable_talker()
                print("Reference model talker component has been disabled")
            else:
                raise ValueError(f"Unsupported model ID for reference model: {model_id}")
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        if processing_class is None:
            if "Qwen2.5-Omni" in model_id:
                processing_class = Qwen2_5OmniProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                # processing_class.tokenizer.padding_side = 'left' # right is wrong
            else:
                raise ValueError(f"Unsupported model ID for processor: {model_id}")

        # Save audio processing flag
        self.use_audio_in_video = use_audio_in_video

        # The rest of the initialization is the same as in Qwen2VLGRPOTrainer
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.weighted_reward = script_args.weighted_reward
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,   
            temperature=1, # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.offline_generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=False,
            temperature=1,
            num_return_sequences=1,
            pad_token_id=pad_token_id,
        )
        self.shuffled_num_generations = self.num_generations // 2
        self.shuffled_generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,   
            temperature=1, # HACK
            num_return_sequences=self.shuffled_num_generations,
            pad_token_id=pad_token_id,
        )
        self.len_control = script_args.len_control
        self.beta = args.beta
        self.alpha = alpha
        self.vad_dict = vad_dict
        self.epsilon_high = 0.28
        self.epsilon_low = 0.20

        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_per_token_logps(self, model, input_ids, return_token_weight, **kwargs):
        #model.thinker(input_ids, **kwargs).logits # Full fine-tune
        # logits = model.base_model.thinker(input_ids, **kwargs).logits # Lora PEFT, it is frozen, don't have grad
        logits = model(input_ids, **kwargs).logits # (B, L, V)
        # Shift logits and labels
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        per_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
            if return_token_weight:
                per_logps.append(log_probs)
        
        return torch.stack(per_token_logps), (
            torch.stack(per_logps) if return_token_weight else None )

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        answer_gt = [x["answer_gt"] for x in inputs] # The answer_gt
        
        input_copy = copy.deepcopy(inputs[0]['prompt'])
        # input_copy = self.remove_none_from_data(input_copy)

        audios, images, videos = process_mm_info(input_copy, use_audio_in_video=self.use_audio_in_video)

        text = self.processing_class.apply_chat_template(
            input_copy,
            add_generation_prompt=True, 
            tokenize=False
        )

        prompt_inputs = self.processing_class(
            text=text,  
            audio=audios,
            images=images, 
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video
        )
        
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            # 为Qwen2.5Omni模型生成文本和音频
            prompt_completion_ids = unwrapped_model.generate(
                **prompt_inputs, 
                generation_config=self.generation_config,
                use_audio_in_video=self.use_audio_in_video,
                return_audio=False
            )
            
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
            
                
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        for key in ["input_ids", "attention_mask"]:
            if key in prompt_inputs:
                prompt_inputs.pop(key)
                
        repeat_factor = len(prompt_completion_ids)

        if inputs[0]['data_type'] == 'image' or inputs[0]['data_type'] == 'image_clver':
            # 处理图像
            if "pixel_values" in prompt_inputs:
                prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(repeat_factor, 1)
                if "image_grid_thw" in prompt_inputs:
                    prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].repeat(repeat_factor, 1)


        elif inputs[0]['data_type'] == 'video_audio':
            # 处理视频
            prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].repeat(len(prompt_completion_ids), 1)
            # prompt_inputs["video_second_per_grid"] = prompt_inputs["video_second_per_grid"].repeat_interleave(repeat_factor, dim=0)
            if 'second_per_grid_ts' in prompt_inputs:
                del prompt_inputs["second_per_grid_ts"]

            # 处理音频
            if "input_features" in prompt_inputs:
                prompt_inputs["input_features"] = prompt_inputs["input_features"].repeat_interleave(repeat_factor, dim=0)
                if "feature_attention_mask" in prompt_inputs:
                    prompt_inputs["feature_attention_mask"] = prompt_inputs["feature_attention_mask"].repeat_interleave(repeat_factor, dim=0)

        elif inputs[0]['data_type'] == 'image_audio':
            # 处理图像
            if "pixel_values" in prompt_inputs:
                prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(repeat_factor, 1)
                if "image_grid_thw" in prompt_inputs:
                    prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].repeat(repeat_factor, 1)

            # 处理音频
            if "input_features" in prompt_inputs:
                prompt_inputs["input_features"] = prompt_inputs["input_features"].repeat_interleave(repeat_factor, dim=0)
                if "feature_attention_mask" in prompt_inputs:
                    prompt_inputs["feature_attention_mask"] = prompt_inputs["feature_attention_mask"].repeat_interleave(repeat_factor, dim=0)
        else:
            raise ValueError(f'We do not support the data type {inputs}')

        per_token_logps, per_logps = self._get_per_token_logps(model, prompt_completion_ids, return_token_weight=True, **prompt_inputs)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]
        per_logps = per_logps[:, prompt_length - 1 :]
        
        # Calculate ref token-wise logps
        with torch.inference_mode(): 
            if self.ref_model is not None:
                ref_per_token_logps, ref_per_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, return_token_weight=False, **prompt_inputs)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps, ref_per_logps = self._get_per_token_logps(model, prompt_completion_ids, return_token_weight=False, **prompt_inputs)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        # 计算KL散度
        x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1
        
        # 解码生成的完成, Warning: skip_special_tokens: <|endoftext|>([151643]) 
        # will contribute the mismatch of token_weights * per_token_loss
        completion_ids_length = len(completion_ids[0])
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        completions_special_tokens = self.processing_class.batch_decode(completion_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
        # 计算奖励
        answer_gt = [answer for answer in answer_gt for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(answer_gt), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["answer_gt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            if reward_func.__name__ in ['offline_think_reward', 'offline_EW_Fscore', 'sentiment_coarse_reward']:
                with torch.inference_mode():
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        with unwrap_model_for_generation(model, self.accelerator) as ref_model:
                            output_reward_func = reward_func(answer_gt=answer_gt, predictions=completions, ref_model=ref_model, processor=self.processing_class,
                                                             generation_config=self.offline_generation_config, **reward_kwargs)
            else:
                output_reward_func = reward_func(answer_gt=answer_gt, predictions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # === 总奖励 START ===
        if not self.weighted_reward:
            rewards = rewards_per_func.sum(dim=1)
        else:
            # [accuracy reward, format reward]
            reward_weights = torch.tensor([2.0, 1.0], dtype=torch.float32, device=device) 
            rewards = (rewards_per_func * reward_weights).sum(dim=1)

        # # 长度控制奖励
        if self.len_control:
            mask = rewards_per_func[:, 0] > 0.1
            lenth_list = completion_mask.sum(1)
            selected_indices = torch.nonzero(mask, as_tuple=True)[0].tolist()
                    
            if len(selected_indices) > 1:     
                for idx in selected_indices: # TODO: The <think>, and the </answer>, and calculate the 
                    if 160 <= lenth_list[idx] <= 258: # [57, 258]
                        rewards[idx] += 0.2

        # 分组奖励计算
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # 奖励归一化
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        # # 损失计算
        # per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        # per_token_loss = -(per_token_loss - self.beta * per_token_kl) # With KL loss
        
        coef_1 = torch.exp(per_token_logps - per_token_logps.detach()) # old_per_token_logps
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        
        # GRPO
        # loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # VAD-TW-GRPO
        token_weights = []
        for completion in completions_special_tokens:
            token_weights.append(self.cal_VAD_token_weights(completion, completion_ids_length))
        
        # Pad, and supplement wth 0 behind
        token_vad_weights = pad_sequence(token_weights, batch_first=True, padding_value=0).to(per_token_loss.device)
        token_weights = compute_token_importance_kl_logs_uniform(per_logps, token_vad_weights, completion_mask, completion_ids, self.num_generations, max_weight=self.alpha)
        
        weighted_loss = per_token_loss * token_weights
        loss = (weighted_loss * completion_mask).sum() / completion_mask.sum()
        
        # 记录指标
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
        
        gathered_rewards = self.accelerator.gather_for_metrics(rewards)
        
        # num_devices = gathered_rewards.size(0) // self.num_generations 
        # rewards_per_device = gathered_rewards.view(num_devices, self.num_generations)
        rewards_per_device, num_devices = self.safe_group_rewards(gathered_rewards, self.num_generations, mode="pad")
        wrong_devices = (rewards_per_device <= 1).all(dim=1)
        wrong_ratio = wrong_devices.sum().item() / num_devices
        
        correct_devices = (rewards_per_device >= 2).all(dim=1)
        correct_ratio = correct_devices.sum().item() / num_devices
        
        self._metrics["loss"].append(loss.item())
        self._metrics["all_wrong"].append(wrong_ratio)
        self._metrics["all_correct"].append(correct_ratio)
        
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        del per_token_logps, per_logps, token_weights, rewards_per_func, rewards, completion_mask, per_token_kl
        torch.cuda.empty_cache()
        gc.collect()
        
        return loss
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()


    def safe_group_rewards(self, gathered_rewards: torch.Tensor, num_generations: int, mode: str = "pad"):
        total_samples = gathered_rewards.shape[0]
        remainder = total_samples % num_generations

        if remainder != 0:
            if mode == "pad":
                pad_size = num_generations - remainder
                pad_values = torch.zeros(pad_size, device=gathered_rewards.device, dtype=gathered_rewards.dtype)
                gathered_rewards = torch.cat([gathered_rewards, pad_values], dim=0)
                print(f"[safe_group_rewards] Warning: padded {pad_size} samples to match num_generations")
            elif mode == "truncate":
                gathered_rewards = gathered_rewards[: total_samples - remainder]
                print(f"[safe_group_rewards] Warning: truncated {remainder} extra samples to match num_generations")
            else:
                raise ValueError(f"Unsupported mode: {mode}. Use 'pad' or 'truncate'.")

        num_devices = gathered_rewards.size(0) // num_generations
        reshaped_rewards = gathered_rewards.view(num_devices, num_generations)
        return reshaped_rewards, num_devices
    
    def cal_VAD_token_weights(self, completion: str, completion_ids_length: int) -> torch.Tensor:
        """
        """
        # Extract words and their positions from text
        matches = list(re.finditer(r"<[^>]+>|\b\w+\b", completion))
        words_with_positions = [(m.group().lower(), m.start(), m.end()) for m in matches]
        
        # Get token offsets from tokenizer
        tokens = self.processing_class.tokenizer(completion, return_offsets_mapping=True)
        
        # Map each token to its corresponding VAD
        token_vad_value = []
        word_idx = 0
        for offset in tokens['offset_mapping']:
            word, start_pos, end_pos = words_with_positions[word_idx]

            # Move to next word if token starts after current word
            while offset[0] >= end_pos and word_idx < len(words_with_positions) - 1:
                word_idx += 1
                word, start_pos, end_pos = words_with_positions[word_idx]

            # Assign VAD if token is within current word, else default to [0,0,0]
            if offset[0] < end_pos:
                token_vad_value.append(self.vad_dict.get(word, [0.0, 0.0, 0.0]))
            else:
                token_vad_value.append([0.0, 0.0, 0.0])

        while len(token_vad_value) < completion_ids_length:
            # Solve the mismatch of skip_special_token <|endoftext|> ([151643])
            token_vad_value.append([0.0, 0.0, 0.0]) 
        
        while len(token_vad_value) > completion_ids_length:
            token_vad_value.pop() 
        
        # Trans List to tensor
        token_vad_value_tensor = torch.tensor(token_vad_value)
        # 0.7*max_vad + 0.3*rms -> Emotion polarity-driven
        max_vals, _ = torch.abs(token_vad_value_tensor).max(dim=-1)
        rms_vals = torch.sqrt(torch.mean(token_vad_value_tensor ** 2, dim=-1))
        token_vad_weights = 0.7 * max_vals + 0.3 * rms_vals
        
        token_vad_weights = torch.ones_like(token_vad_weights) + token_vad_weights
        return token_vad_weights
    

def compute_token_importance_kl_logs_uniform(per_logps, emotion_polarity, completion_mask, completion_ids, num_generations, max_weight=1.5):
    # Basic dimension checks
    if not isinstance(per_logps, torch.Tensor) or per_logps.ndim != 3:
        return torch.ones_like(per_logps[..., 0])
        
    total_size, token_length, vocab_size = per_logps.size()
    if total_size % num_generations != 0:
        return torch.ones_like(per_logps[..., 0])
        
    batch_size = total_size // num_generations
    # Reshape tensors for group-wise computation
    grouped_logps = per_logps.view(batch_size, num_generations, token_length, vocab_size)
    grouped_masks = completion_mask.view(batch_size, num_generations, token_length)
    grouped_masks = grouped_masks.unsqueeze(-1)  # (batch_size, num_generations, token_length, 1)
    
    # Create uniform distribution for masked positions (log space)
    uniform_logps = torch.full_like(grouped_logps, -math.log(vocab_size))
    
    # Set logps to uniform distribution for positions beyond sequence length
    masked_logps = torch.where(grouped_masks == 1, grouped_logps, uniform_logps)
    
    # Calculate mean distribution for each token position
    mean_logps = masked_logps.mean(dim=1, keepdim=True)  # (batch_size, 1, token_length, vocab_size)
    
    # Calculate KL divergence between each sequence and the mean
    diff = mean_logps - masked_logps
    diff = torch.clamp(diff, min=-11.0, max=11.0)
    token_kl = (torch.exp(diff) - diff - 1).sum(dim=-1)
    token_kl = token_kl.mean(dim=1)
    # emotion-polarity driven
    token_kl = emotion_polarity * token_kl
    # Emotion polarity-driven and entropy importance weights
    # Apply min-max normalization to KL divergence
    kl_min = token_kl.min(dim=-1, keepdim=True)[0]
    kl_max = token_kl.max(dim=-1, keepdim=True)[0]
    normalized_kl = (token_kl - kl_min) / (kl_max - kl_min + 1e-8)
    # Map normalized KL to weights range [1.0, max_weight]
    token_weights = 1.0 + (max_weight - 1.0) * normalized_kl
    
    # Apply the completion mask to ensure consistency
    token_weights = token_weights * completion_mask

    # log token_weights
    return token_weights