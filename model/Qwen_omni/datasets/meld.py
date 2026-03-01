import os
import pandas as pd

import torch
from moviepy import *
import os
import pandas as pd
from typing import List, Tuple
from model.Qwen_omni.datasets.base_dataset import BaseDataset


class MELD_Dataset(BaseDataset):
    def __init__(self, cfg, processor):
        super().__init__(processor=processor) # Initial BaseDataset
        self.cfg = cfg
        self.processor = processor
        
        meld_path = self.cfg.dataset.path_dict['MELD']
        

        if cfg.run.type == 'eval':
            csv1_path = os.path.join(meld_path, 'test_sent_emo.csv')
            self.df = pd.read_csv(csv1_path)
            self.df['file'] = self.df.apply(lambda row: f"test_dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}", axis=1)
            self.dir = 'test'
        elif cfg.run.type == 'train':
            csv1_path = os.path.join(meld_path, 'train_sent_emo.csv')
            self.df = pd.read_csv(csv1_path)
            self.df['file'] = self.df.apply(lambda row: f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}", axis=1)
            self.df = self.df[self.df['file'] != 'dia125_utt3']
            self.dir = 'train'
        else:
            raise ValueError("Invalid run type. Must be 'eval' or 'train'.")
        
        base_path = self.cfg.dataset.path_dict['MELD']
        dir = self.dir  # 'train' or 'test'
        
        # self.df['openface_path'] = self.df['file'].apply(lambda f: os.path.join(base_path, f'{dir}/openface_face', f + '.npy'))
        self.df['audio_path'] = self.df['file'].apply(lambda f: os.path.join(base_path, f'{dir}/subaudio', f + '.wav'))
        self.df['video_path'] = self.df['file'].apply(lambda f: os.path.join(base_path, f'{dir}/subvideo', f + '.mp4'))
        
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
                
        openface_path = None
        audio_path = row['audio_path']
        video_path = row['video_path']
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        elif not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        # elif not os.path.exists(openface_path):
        #     raise FileNotFoundError(f"Openface file not found: {openface_path}")
        if self.cfg.model.name == 'qwen_omni_rl' and not self.cfg.fewshot:
            conversations = super().get_train_conv(cfg=self.cfg, video_path=video_path, audio_path=audio_path, image_path=openface_path, subtitle=row['Utterance'])
            sample = super().process_video_audio_text(conversations=conversations)
        elif not self.cfg.fewshot:
            conversations = super().get_infer_conv(cfg=self.cfg, video_path=video_path, audio_path=audio_path, image_path=openface_path, subtitle=row['Utterance'])
            sample = super().process_video_audio_text(conversations=conversations)
        else:
            conversations = super().get_fewshot_conv(cfg=self.cfg, fewshot_dict=self.fewshot_dict[row['file']], video_path=video_path, audio_path=audio_path, subtitle=row['Utterance'])
            sample = super().process_video_audio_text(conversations=conversations)
            
        sample['annotation'] = row['Emotion']
        sample['discrete'] = row['Sentiment']
        sample['text'] = row['Utterance']
        sample['file'] = row['file']
        sample['video_path'] = video_path
        sample['audio_path'] = audio_path
        

        return sample
    
    def my_collate_fn(self, batch):
        inputs = [item['inputs'] for item in batch]
        discrete = [item['discrete'] for item in batch]
        label = [item['annotation'] for item in batch]
        text = [item['text'] for item in batch]
        video_file = [item['file'] for item in batch]
        video_path = [item['video_path'] for item in batch]
        audio_path = [item['audio_path'] for item in batch]

        return {
            'inputs': inputs,
            'discrete': discrete, # positive, negative, neutral
            'label': label, # anger, joy
            'text_en': text,
            'file': video_file,
            'video_path': video_path,
            'audio_path': audio_path
        }
        

    def get_test_name2gt(self):
        name2gt = {}
        for index, row in self.df.iterrows():
            name = row['file']
            val = row['Emotion']
            name2gt[name] = val
        return name2gt
    
    get_name2label = get_test_name2gt