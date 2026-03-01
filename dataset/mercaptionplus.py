import pandas as pd
import torch

import webdataset as wds
from moviepy import *
import os
import pandas as pd
import torchaudio
import torchaudio.transforms as T
from typing import List, Tuple
from dataset.base_dataset import BaseDataset
from model.AffectGPT.ImageBind.data import load_audio, transform_audio
from processor.video_processor import AlproVideoEvalProcessor, AlproVideoTrainProcessor,ToTHWC, ToUint8, load_video, load_face
from utils.utils import string_to_list
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

class MERCaptionPlus_Dataset(BaseDataset):
    def __init__(self, cfg, vis_transform=None, img_transform=None, txt_transform=None):

        self.cfg = cfg
        self.n_frms = None
        
        mercaptionplus_path = self.cfg.dataset.path_dict['MERCaptionPlus']
        csv1_path = os.path.join(mercaptionplus_path, 'track2_train_mercaptionplus.csv')
        csv2_path = os.path.join(mercaptionplus_path, 'track3_train_mercaptionplus.csv')
        csv3_path = os.path.join(mercaptionplus_path, 'subtitle_chieng.csv')
        self.vis_transform = vis_transform
        self.img_transform = img_transform
        self.txt_transform = txt_transform
        
        df1 = pd.read_csv(csv1_path)
        df2 = pd.read_csv(csv2_path)
        df3 = pd.read_csv(csv3_path)
        merged_df = pd.merge(df1, df2, on='name', how='inner')
        self.full_df = pd.merge(merged_df, df3, on='name', how='inner')
        
        self.label_type_candidates = ['description', 'ovlabel']
        self.label_type = cfg.run.label_type
            
        self.processor = AlproVideoTrainProcessor()
        self.processor = self.processor.from_config(cfg.model.vis_processor.train)
        
        # 生成完整视频文件名，带.mp4后缀
        self.df['video_file'] = self.df['name'].apply(lambda x: x + '.mp4')
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = os.path.join(os.path.join(self.cfg.dataset.path_dict['MERCaptionPlus'], 'video'), row['video_file'])
    
        raw_frame, video_tensor = self.processor(video_path)
        raw_audio = load_audio([video_path], "cpu", clips_per_video=8)[0]
        audio = transform_audio(raw_audio, "cpu") # [8, 1, 128, 204]
        
        sample = {
            'frame': video_tensor,
            'raw_frame': raw_frame,
            'audio': audio,
            'raw_audio': raw_audio,
            'english': row['english'],
            'chinese': row['chinese'],
            'ov_label': row['openset'],
            'description': row['reason']
        }

        cur_label_type = super().get_cur_label_type(self.label_type_candidates, self.label_type)
        qa_pair = self.get_qa_pairs(self.cfg.dataset.name, cur_label_type, sample)
        
        sample['qa_pair'] = qa_pair    
        
        return sample
    
    def my_collate_fn(self, batch):
        # batch 是 list of sample dict
        frames = [item['frame'] for item in batch]
        raw_frames = [item['raw_frame'] for item in batch]
        audio = [item['audio'] for item in batch]
        raw_audio = [item['raw_audio'] for item in batch]
        chinese = [item['chinese'] for item in batch]
        english = [item['english'] for item in batch]
        qa_pair = [item['qa_pair'] for item in batch]

        return {
            'frame': torch.stack(frames), 
            'raw_frame': torch.stack(raw_frames),
            'audio': torch.stack(audio),
            'raw_audio': torch.stack(raw_audio),
            'english': english,
            'chinese': chinese,
            'qa_pair': qa_pair
        }