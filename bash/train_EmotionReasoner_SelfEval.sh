#!/bin/bash

export DEBUG_MODE="true" 
export LOG_PATH="./experiment/train_qwen_omni/logs/debug_log.txt"
export WANDB_NAME=Reward_VADTW_Online_func4_lr5e-5
export CUDA_LAUNCH_BLOCKING=1

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12366" \
    ./scripts/train_qwen_omni_vadtw.py \
    --dataset_name MERCaptionPlus \
    --output_dir /root/autodl-tmp/EMER-MLLM/qwen-omni-peft-vadtw-online \
    --deepspeed ./experiment/train_qwen_omni/zero2.json \
    --model_name_or_path /root/autodl-tmp/EMER-MLLM/Qwen2.5-Omni-7B \
    --reward_funcs "EmotionWheel" "format" "SelfEval" \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing true \
    --learning_rate 5e-5 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 true \
    --logging_steps 1 \
    --len_control true \
    --weighted_reward false\
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name $WANDB_NAME \
    --save_steps 1 \
    --save_total_limit 10 \
    --beta 0 \
    --model_alpha 1.70 \
    --max_grad_norm 5 \
    --save_only_model false \
    --num_generations 16 \
    --model_type omni \
    --use_audio_in_video false \
    --use_sft_base_model false \
    --use_peft true \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target_modules "q_proj" "k_proj" "v_proj" "o_proj" "gate_proj" "up_proj" "down_proj" "embed_tokens" \