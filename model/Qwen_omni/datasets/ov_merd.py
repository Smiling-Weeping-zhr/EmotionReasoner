import os
import pandas as pd

import torch
from moviepy import *
import os
import pandas as pd
from typing import List, Tuple
from utils.utils import string_to_list
from model.Qwen_omni.datasets.base_dataset import BaseDataset

class OVMERD_Dataset(BaseDataset):
    def __init__(self, cfg, processor):
        super().__init__(processor=processor)
        self.cfg = cfg
        self.processor = processor
        
        ovmerd_path = self.cfg.dataset.path_dict['OV-MERD']
        csv1_path = os.path.join(ovmerd_path, 'final-EMER-reason.csv')
        csv2_path = os.path.join(ovmerd_path, 'track2_train_ovmerd.csv')
        csv3_path = os.path.join(ovmerd_path, 'subtitle_chieng.csv')
        
        df1 = pd.read_csv(csv1_path).rename(columns={'english': 'reason_en', 'chinese': 'reason_ch'})
        df2 = pd.read_csv(csv2_path)
        df3 = pd.read_csv(csv3_path)
        
        df = pd.merge(df1, df2, on="name", how="inner")
        self.df = pd.merge(df, df3, on="name", how="inner")
        self.dir = ''
    
        self.df['file'] = self.df['name']
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = os.path.join(os.path.join(self.cfg.dataset.path_dict['OV-MERD'], 'video'), row['name']+'.mp4')
        audio_path = os.path.join(os.path.join(self.cfg.dataset.path_dict['OV-MERD'], 'audio'), row['name']+'.wav')
        openface_path = os.path.join(os.path.join(self.cfg.dataset.path_dict['OV-MERD'], 'openface_face'), row['name']+'.npy')

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        elif not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        elif not os.path.exists(openface_path):
            raise FileNotFoundError(f"Openface file not found: {openface_path}")
        
        if self.cfg.model.name == 'qwen_omni_rl' and not self.cfg.fewshot:
            conversations = super().get_train_conv(cfg=self.cfg, video_path=video_path, audio_path=audio_path, image_path=openface_path, subtitle=row['english'])
            sample = super().process_video_audio_text(conversations=conversations)
        elif not self.cfg.fewshot:
            conversations = super().get_infer_conv(cfg=self.cfg, video_path=video_path, audio_path=audio_path, image_path=openface_path, subtitle=row['english'])
            sample = super().process_video_audio_text(conversations=conversations)
        else:
            conversations = super().get_fewshot_conv(cfg=self.cfg, fewshot_dict=self.fewshot_dict[row['name']], video_path=video_path, audio_path=audio_path, subtitle=row['english'])
            sample = super().process_video_audio_text(conversations=conversations)
        
        sample['reason_en'] = row['reason_en']
        sample['reason_ch'] = row['reason_ch']
        sample['english'] = row['english']
        # sample['valence'] = row['valence']
        sample['openset'] = row['openset']
        sample['chinese'] = row['chinese']
        # sample['english'] = row['english']
        sample['file'] = row['name']
        sample['video_path'] = video_path
        sample['audio_path'] = audio_path

        return sample
    
    def my_collate_fn(self, batch):
        inputs = [item['inputs'] for item in batch]
        english = [item['english'] for item in batch]
        chinese = [item['chinese'] for item in batch]
        reason_en = [item['reason_en'] for item in batch]
        openset = [item['openset'] for item in batch]
        reason_ch = [item['reason_ch'] for item in batch]
        # english = [item['english'] for item in batch]
        file = [item['file'] for item in batch]
        video_path = [item['video_path'] for item in batch]
        audio_path = [item['audio_path'] for item in batch]

        return {
            'inputs': inputs,
            'discrete': None,
            'reason_ch': reason_ch,
            'reason_en': reason_en,
            'openset': openset,
            'text': chinese,
            'text_en': english,
            'file': file,
            'video_path': video_path,
            'audio_path': audio_path
        }
        
    def get_test_name2gt(self):
        name2gt = {}
        for index, row in self.df.iterrows():
            name = row['name']
            openset_list = string_to_list(row['openset'])
            if len(openset_list) == 0: openset_list = ['neutral']
            name2gt[name] = ', '.join(openset_list)
        return name2gt
    
    get_name2label = get_test_name2gt