import os
import pandas as pd

import torch
import ipdb
from moviepy import *
import os
import pandas as pd
from typing import List, Tuple
from model.Qwen_omni.datasets.base_dataset import BaseDataset

class MER2023_Dataset(BaseDataset):
    def __init__(self, cfg, processor):
        super().__init__(processor=processor)
        self.cfg = cfg
        self.processor = processor
        
        mer2023_path = self.cfg.dataset.path_dict['MER2023']
        

        if cfg.run.type in 'eval':
            csv1_path = os.path.join(mer2023_path, 'test1-label.csv')
            csv2_path = os.path.join(mer2023_path, 'transcription-engchi-polish.csv')
            df1 = pd.read_csv(csv1_path)
            df2 = pd.read_csv(csv2_path)
            self.df = pd.merge(df1, df2, on='name', how='inner')
            self.dir = 'test'
        elif cfg.run.type == 'train':
            csv1_path = os.path.join(mer2023_path, 'train-label.csv')
            csv2_path = os.path.join(mer2023_path, 'transcription-engchi-polish.csv')
            df1 = pd.read_csv(csv1_path)
            df2 = pd.read_csv(csv2_path)
            self.df = pd.merge(df1, df2, on='name', how='inner')
            self.dir = 'train'
        else:
            raise ValueError("Invalid run type. Must be 'eval' or 'train'.")
        
        dir = self.dir  # 'train' or 'test'
        
        self.df['audio_path'] = self.df['name'].apply(lambda f: os.path.join(mer2023_path, f'{dir}/audio', f + '.wav'))
        self.df['video_path'] = self.df['name'].apply(lambda f: os.path.join(mer2023_path, f'{dir}/video', f + '.avi'))
        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row['audio_path']
        video_path = row['video_path']
        openface_path = None
    
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        elif not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        # elif not os.path.exists(openface_path):
        #     raise FileNotFoundError(f"Openface file not found: {openface_path}")

        if self.cfg.model.name == 'qwen_omni_rl' and not self.cfg.fewshot:
            conversations = super().get_train_conv(cfg=self.cfg, video_path=video_path, audio_path=audio_path, image_path=openface_path, subtitle=row['english'])
            sample = super().process_video_audio_text(conversations=conversations)
        elif not self.cfg.fewshot:
            conversations = super().get_infer_conv(cfg=self.cfg, video_path=video_path, audio_path=audio_path, image_path=openface_path, subtitle=row['english'])
            sample = super().process_video_audio_text(conversations=conversations)
        else:
            conversations = super().get_fewshot_conv(cfg=self.cfg, fewshot_dict=self.fewshot_dict[row['name']], video_path=video_path, audio_path=audio_path, subtitle=row['english'])
            sample = super().process_video_audio_text(conversations=conversations)
            
        sample['discrete'] = row['discrete']
        sample['valence'] = row['valence']
        sample['text'] = row['chinese']
        sample['text_en'] = row['english']
        sample['file'] = row['name']
        sample['video_path'] = video_path
        sample['audio_path'] = audio_path
    
        return sample
    
    def my_collate_fn(self, batch):
        inputs = [item['inputs'] for item in batch]
        chinese = [item['text'] for item in batch]
        english = [item['text_en'] for item in batch]
        discretes = [item['discrete'] for item in batch]
        valences = [item['valence'] for item in batch]
        video_file = [item['file'] for item in batch]
        video_path = [item['video_path'] for item in batch]
        audio_path = [item['audio_path'] for item in batch]

        return {
            'inputs': inputs,
            'text': chinese,
            'text_en': english,
            'discrete': discretes,
            'valence': torch.tensor(valences),
            'file': video_file,
            'video_path': video_path,
            'audio_path': audio_path
        }

    def get_test_name2gt(self):
        name2gt = {}
        for index, row in self.df.iterrows():
            name = row['name']
            val = row['discrete']
            name2gt[name] = val
        return name2gt
    
    get_name2label = get_test_name2gt