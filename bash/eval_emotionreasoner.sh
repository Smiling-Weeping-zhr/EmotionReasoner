#!/bin/bash

cd /root/autodl-tmp/EMER

# datasets=('MER2024' 'MER2023' 'CMU-MOSI' 'CH-SIMS-v2' 'CH-SIMS' 'MELD' 'IEMOCAP' 'OV-MERD' 'CMU-MOSEI')
datasets=('MER2023' 'MER2024' 'CMU-MOSI' 'CH-SIMS' 'CH-SIMS-v2' 'OV-MERD')
# paths=("/root/autodl-tmp/Qwen-RLVR/qwen-omni-peft-func4-coarse/checkpoint-195" "/root/autodl-tmp/Qwen-RLVR/qwen-omni-peft-vad-tw-polarity/checkpoint-185")
# for path in "${paths[@]}"
# do
#     for dataset in "${datasets[@]}"
#     do
#         echo "Evaluating $dataset with checkpoint $path ..."
#         python /root/autodl-tmp/EMER/main.py eval_qwen_omni_rl --dataset="$dataset" --peft-ckpt="$path"
#         echo "Done with $dataset and checkpoint $path"
#     done
#     datasets=('CMU-MOSEI')
# done
for dataset in "${datasets[@]}"
do
    echo "Evaluating $dataset ..."
    python /root/autodl-tmp/EMER/main.py eval_qwen_omni_rl --dataset="$dataset" --peft-ckpt="/root/autodl-tmp/Qwen-RLVR1/qwen-omni-peft-vadtw-offline-two/checkpoint-180"
    echo "Done with $dataset"
done

echo "Calculating metrics for qwen_omni ..."
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python /root/autodl-tmp/EMER/main.py calculate_metrics --model='qwen_omni_rl'
echo "Metrics calculation done."