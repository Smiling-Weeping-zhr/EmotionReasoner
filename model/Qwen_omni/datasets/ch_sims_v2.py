import os
import pandas as pd

import torch
from moviepy import *
import os
import pandas as pd
import torchaudio.transforms as T
from typing import List, Tuple
from model.Qwen_omni.datasets.base_dataset import BaseDataset

class CH_SIMS_v2_Dataset(BaseDataset):
    def __init__(self, cfg, processor):
        super().__init__(processor=processor)
        self.cfg = cfg
        self.processor = processor
        
        sims_path = self.cfg.dataset.path_dict['CH-SIMS-v2']
        csv1_path = os.path.join(sims_path, 'meta_translated.csv')
        
        df = pd.read_csv(csv1_path)
        
        if cfg.run.type == 'eval':
            self.df = df.loc[df['mode'] == 'test']
            self.dir = 'test'
        elif cfg.run.type == 'train':
            self.df = df.loc[df['mode'] == 'train']
            self.dir = 'train'
        else:
            raise ValueError("Invalid run type. Must be 'eval' or 'train'.")
        
        self.df['file'] = self.df.apply(lambda row: f"{row['video_id']}_{row['clip_id']}", axis=1)

        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = os.path.join(os.path.join(self.cfg.dataset.path_dict['CH-SIMS-v2'], f'{self.dir}/video_new'), row['file']+'.mp4')
        audio_path = os.path.join(os.path.join(self.cfg.dataset.path_dict['CH-SIMS-v2'], f'{self.dir}/audio'), row['file']+'.wav')
        openface_path = os.path.join(os.path.join(self.cfg.dataset.path_dict['CH-SIMS-v2'], f'{self.dir}/openface_face'), row['file']+'.npy')
    
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        elif not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        elif not os.path.exists(openface_path):
            raise FileNotFoundError(f"Openface file not found: {openface_path}")
        
        if not self.cfg.fewshot:
            conversations = super().get_infer_conv(cfg=self.cfg, video_path=video_path, audio_path=audio_path, image_path=openface_path, subtitle=row['text_en'])
            sample = super().process_video_audio_text(conversations=conversations)
        else:
            raise ValueError(f"{self.cfg.fewshot} error.")
        
        sample['annotation'] = row['annotation']
        sample['label'] = row['label']
        sample['text'] = row['text']
        sample['text_en'] = row['text_en']
        sample['video_file'] = row['file']
        sample['video_path'] = video_path
        sample['audio_path'] = audio_path
    
        return sample
    
    def my_collate_fn(self, batch):
        inputs = [item['inputs'] for item in batch]
        annotation = [item['annotation'] for item in batch]
        label = [item['label'] for item in batch]
        text = [item['text'] for item in batch]
        text_en = [item['text_en'] for item in batch]
        video_file = [item['video_file'] for item in batch]
        video_path = [item['video_path'] for item in batch]
        audio_path = [item['audio_path'] for item in batch]

        return {
            'inputs': inputs,
            'annotation': annotation,
            'label': torch.tensor(label),
            'text': text,
            'text_en': text_en,
            'file': video_file,
            'video_path': video_path,
            'audio_path': audio_path
        }
        
    def get_name2label(self): # (name, label_word)
        name2label = {}
        for index, row in self.df.iterrows():
            name = row['file']
            val = row['annotation']
            name2label[name] = val
        return name2label
    
    def get_test_name2gt(self): # (name, label_float)
        name2gt = {}
        for index, row in self.df.iterrows():
            name = row['file']
            val = row['label']
            name2gt[name] = val
        return name2gt