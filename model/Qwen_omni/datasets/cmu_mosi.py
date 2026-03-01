import os
import pandas as pd

import pickle
import torch
from moviepy import *
import os
import pandas as pd
import ipdb
from typing import List, Tuple
from model.Qwen_omni.datasets.base_dataset import BaseDataset

class CMU_MOSI_Dataset(BaseDataset):
    def __init__(self, cfg, processor):
        super().__init__(processor=processor) # Initial BaseDataset
        self.cfg = cfg
        self.processor = processor
        
        cmu_path = self.cfg.dataset.path_dict['CMU-MOSI']
        csv1_path = os.path.join(cmu_path, 'label.csv')
        
        df = pd.read_csv(csv1_path)
        
        if self.cfg.fewshot:
            fewshot_pkl = f'/root/autodl-tmp/EMER/experiment/retrieval_qwen_omni/fewshot/CMU-MOSI/CMU-MOSI-fewshot{self.cfg.icl.fewshot}'
            if os.path.exists(fewshot_pkl):
                with open(fewshot_pkl, 'rb') as f:
                    self.fewshot_dict = pickle.load(f)
            else:
                self.fewshot_dict = None

        if cfg.run.type == 'eval':
            self.df = df.loc[df['mode'] == 'test']
            self.dir = 'test'
        elif cfg.run.type == 'train':
            self.df = df.loc[df['mode'] == 'train']
            self.dir = 'train'
            self.label_type_candidates = ['valence', 'sentiment']
        else:
            raise ValueError("Invalid run type. Must be 'eval' or 'train'.")
        
        self.df['file'] = self.df.apply(lambda row: f"{row['video_id']}_{row['clip_id']}", axis=1)
        base_path = self.cfg.dataset.path_dict['CMU-MOSI']
        dir = self.dir  # 'train' or 'test'

        self.df['openface_path'] = self.df['file'].apply(lambda f: os.path.join(base_path, f'{dir}/openface_face', f + '.npy'))
        self.df['audio_path'] = self.df['file'].apply(lambda f: os.path.join(base_path, f'{dir}/subaudio', f + '.wav'))
        self.df['video_path'] = self.df['file'].apply(lambda f: os.path.join(base_path, f'{dir}/subvideo', f + '.mp4'))
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        openface_path = row['openface_path']
        audio_path = row['audio_path']
        video_path = row['video_path']
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        elif not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        # elif not os.path.exists(openface_path):
        #     raise FileNotFoundError(f"Openface file not found: {openface_path}")

        if self.cfg.model.name == 'qwen_omni_rl' and not self.cfg.fewshot:
            conversations = super().get_train_conv(cfg=self.cfg, video_path=video_path, audio_path=audio_path, image_path=openface_path, subtitle=row['text'])
            sample = super().process_video_audio_text(conversations=conversations)
        elif not self.cfg.fewshot:
            conversations = super().get_infer_conv(cfg=self.cfg, video_path=video_path, audio_path=audio_path, image_path=openface_path, subtitle=row['text'])
            sample = super().process_video_audio_text(conversations=conversations)
        else:
            conversations = super().get_fewshot_conv(cfg=self.cfg, fewshot_dict=self.fewshot_dict[row['file']], video_path=video_path, audio_path=audio_path, subtitle=row['text'])
            sample = super().process_video_audio_text(conversations=conversations)
            
        sample['annotation'] = row['annotation']
        sample['label'] = row['label']
        sample['text'] = row['text']
        sample['file'] = row['file']
        sample['video_path'] = video_path
        sample['audio_path'] = audio_path
        
        return sample
    
    def my_collate_fn(self, batch):
        inputs = [item['inputs'] for item in batch]
        annotation = [item['annotation'] for item in batch]
        label = [item['label'] for item in batch]
        text = [item['text'] for item in batch]
        # qa_pair = [item['qa_pair'] for item in batch] if self.cfg.run.type == 'train' else None
        qa_pair = None
        video_file = [item['file'] for item in batch]
        video_path = [item['video_path'] for item in batch]
        audio_path = [item['audio_path'] for item in batch]

        return {
            'inputs': inputs,
            'annotation': annotation,
            'label': torch.tensor(label),
            'text_en': text,
            'qa_pair': qa_pair,
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
    
    # def get_test_name2gt(self):
    #     name2gt = {}
    #     for index, row in self.df.iterrows():
    #         name = row['file']
    #         val = row['label']
    #         name2gt[name] = val
    #     return name2gt