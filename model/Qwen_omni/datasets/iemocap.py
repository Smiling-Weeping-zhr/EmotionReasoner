import os
import pandas as pd

import torch
from moviepy import *
import os
import pandas as pd
from typing import List, Tuple
from model.Qwen_omni.datasets.base_dataset import BaseDataset

emos = ['happy', 'sad', 'neutral', 'anger']
emo2idx, idx2emo = {}, {}
for ii, emo in enumerate(emos):
    emo2idx[emo] = ii
    idx2emo[ii]  = emo
    
class IEMOCAP_Dataset(BaseDataset):
    def __init__(self, cfg, processor):
        super().__init__(processor=processor) # Initial BaseDataset
        self.cfg = cfg
        self.processor = processor

        iemocap_path = self.cfg.dataset.path_dict['IEMOCAP']
        csv1_path = os.path.join(iemocap_path, 'label_ses05.csv')
            
        self.df = pd.read_csv(csv1_path)

        if cfg.run.type == 'eval':
            self.dir = 'test'
        elif cfg.run.type == 'train':
            self.label_type_candidates = ['valence', 'sentiment']
            self.df['file'] = self.df['name']
            self.df['file_name'] = self.df['name']
            self.dir = 'train'
        else:
            raise ValueError("Invalid run type. Must be 'eval' or 'train'.")
    
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = os.path.join(os.path.join(self.cfg.dataset.path_dict['IEMOCAP'], self.dir, 'subvideo-tgt'), row['name']+'.avi')
        audio_path = os.path.join(os.path.join(self.cfg.dataset.path_dict['IEMOCAP'], self.dir, 'subaudio'), row['name']+'.wav')
        openface_path = os.path.join(os.path.join(self.cfg.dataset.path_dict['IEMOCAP'], self.dir, 'openface_face'), row['name']+'.npy')
        
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
            conversations = super().get_fewshot_conv(cfg=self.cfg, fewshot_dict=self.fewshot_dict[row['file']], video_path=video_path, audio_path=audio_path, subtitle=row['english'])
            sample = super().process_video_audio_text(conversations=conversations)
        
        sample['emo'] = idx2emo[row['emo']]
        sample['val'] = row['val']
        sample['text'] = row['chinese']
        sample['text_en'] = row['english']
        sample['file'] = row['name']
        sample['video_path'] = video_path
        sample['audio_path'] = audio_path
        
        # if self.cfg.run.type == 'train':
        #     cur_label_type = super().get_cur_label_type(self.label_type_candidates, self.label_type)
        #     qa_pair = self.get_qa_pairs(self.cfg.dataset.name, cur_label_type, sample)
        #     sample['qa_pair'] = qa_pair
            
        return sample
    
    def my_collate_fn(self, batch):
        inputs = [item['inputs'] for item in batch]
        emo = [item['emo'] for item in batch]
        val = [item['val'] for item in batch]
        text = [item['text'] for item in batch]
        text_en = [item['text_en'] for item in batch]
        qa_pair = None
        # qa_pair = [item['qa_pair'] for item in batch] if self.cfg.run.type == 'train' else None
        video_file = [item['file'] for item in batch]
        video_path = [item['video_path'] for item in batch]
        audio_path = [item['audio_path'] for item in batch]

        return {
            'inputs': inputs,
            'emo': emo,
            'val': torch.tensor(val),
            'text': text,
            'text_en': text_en,
            'qa_pair': qa_pair,
            'file': video_file,
            'video_path': video_path,
            'audio_path': audio_path
        }
        
    def get_test_name2gt(self):
        name2gt = {}
        for index, row in self.df.iterrows():
            name = row['name']
            val = idx2emo[row['emo']]
            name2gt[name] = val
        return name2gt