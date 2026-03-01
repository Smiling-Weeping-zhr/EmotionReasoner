import os

from enum import Enum
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
import torch
from omegaconf import DictConfig, OmegaConf
import ipdb
import torch
from utils.utils import *
from torch.utils.data import DataLoader
import pickle
from typing import List, Tuple, Any
from model.Qwen_omni.datasets.mer2024 import MER2024_Dataset
from model.Qwen_omni.datasets.mer2023 import MER2023_Dataset
from model.Qwen_omni.datasets.cmu_mosei import CMU_MOSEI_Dataset
from model.Qwen_omni.datasets.cmu_mosi import CMU_MOSI_Dataset
from model.Qwen_omni.datasets.ch_sims import CH_SIMS_Dataset
from model.Qwen_omni.datasets.ch_sims_v2 import CH_SIMS_v2_Dataset
from model.Qwen_omni.datasets.meld import MELD_Dataset
from model.Qwen_omni.datasets.ov_merd import OVMERD_Dataset
from model.Qwen_omni.datasets.iemocap import IEMOCAP_Dataset
from tqdm import tqdm

import logging
logging.getLogger().setLevel(logging.ERROR)

def get_name2cls(dataset: str, cfg: DictConfig, processor: Any) -> torch.utils.data.Dataset:
    if dataset == 'MER2023':          return MER2023_Dataset(cfg=cfg, processor=processor)
    if dataset == 'MER2024':          return MER2024_Dataset(cfg=cfg, processor=processor)
    if dataset == 'MELD':             return MELD_Dataset(cfg=cfg, processor=processor)
    if dataset == 'IEMOCAP':      return IEMOCAP_Dataset(cfg=cfg, processor=processor)
    if dataset == 'CMU-MOSI':          return CMU_MOSI_Dataset(cfg=cfg, processor=processor)
    if dataset == 'CMU-MOSEI':         return CMU_MOSEI_Dataset(cfg=cfg, processor=processor)
    if dataset == 'CH-SIMS':             return CH_SIMS_Dataset(cfg=cfg, processor=processor)
    if dataset == 'CH-SIMS-v2':           return CH_SIMS_v2_Dataset(cfg=cfg, processor=processor)
    if dataset == 'OV-MERD':            return OVMERD_Dataset(cfg=cfg, processor=processor)
    print ('dataset cls not provided!')
    return None  

def eval_Qwen_omni(cfg: DictConfig, model: torch.nn.Module, processor: Any) -> Tuple[List, List]:
    """
    Evaluate emotion recognition on the MER2024 dataset, modality (audio, video).

    Args:
        cfg (DictConfig): Configuration object containing dataset and model settings.
        model (Torch.nn.Module): The trained model to evaluate.

    Returns:
        Tuple[List, List]: Lists of true labels, predicted labels.
    """
    peft_ckpt = cfg.model.get("peft_ckpt", None)
    if peft_ckpt is not None:
        ckpt_step_path = cfg.model.peft_ckpt
        suffix = str(os.path.join(*ckpt_step_path.split(os.sep)[-2:]))
    else:
        suffix = 'base_model'
    if cfg.run.type == 'eval':
        pkl_file = f'./experiment/eval_{cfg.model.name}/{suffix}/{cfg.dataset.name}/{cfg.dataset.name}.pkl'
    elif cfg.run.type == 'reason':
        pkl_file = f'./experiment/eval_{cfg.model.name}/reason/{cfg.dataset.name}/{cfg.dataset.name}.pkl'
    else:
        raise ValueError(f"The run type {cfg.run.type} error!")
    
    # First Step to extract emotion labels
    emotion_extraction = {}
    dataset = get_name2cls(cfg.dataset.name, cfg, processor)
    
    dir_path = os.path.dirname(pkl_file)
    os.makedirs(dir_path, exist_ok=True)
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            emotion_extraction = pickle.load(f)
            
    dataloader = DataLoader(dataset, batch_size=cfg.run.eval.batch_size, shuffle=False, collate_fn=dataset.my_collate_fn, num_workers=0)
    cnt=0
    for batch in tqdm(dataloader, desc="Evaluating"):
        assert len(batch['file']) == 1, 'Error, batch size must be one'
        inputs = batch['inputs'][0]
        video_file = batch['file'][0]
        if video_file in emotion_extraction:
            continue
        inputs.to(model.device).to(model.dtype) # sample_00004469 # test_dia38_utt3 - 5mins
        # Forward pass through the model
        with torch.inference_mode():
            output = model.generate(**inputs, use_audio_in_video=False, return_audio=False)
            response = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            response = response[0].split('assistant', 1)[-1].strip()
            emotion_extraction[video_file] = response
            del inputs, output
            torch.cuda.empty_cache()
            # ipdb.set_trace()
        cnt+=1
        if cnt%100 == 0:
            with open(pkl_file, 'wb') as f:
                pickle.dump(emotion_extraction, f)

    dir_path = os.path.dirname(pkl_file)
    os.makedirs(dir_path, exist_ok=True)
    with open(pkl_file, 'wb') as f:
        pickle.dump(emotion_extraction, f)
    print(f'Emotion labels have been extracted and placed into {pkl_file}')
    pass