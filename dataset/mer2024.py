import glob
import os
import pandas as pd
import json
from copy import deepcopy

import torch
import webdataset as wds
import cv2
from moviepy import *
import os
import pandas as pd
from dataset.base_dataset import BaseDataset
from processor.video_processor import AlproVideoEvalProcessor, AlproVideoTrainProcessor
from torch.utils.data import Dataset

class MER2024_Dataset(BaseDataset):
    def __init__(self, cfg, vis_transform=None, img_transform=None, txt_transform=None):

        self.cfg = cfg
        self.n_frms = None
        
        self.needed_data = super().get_needed_data(self.cfg.model.multi_type)
        mer2024_path = self.cfg.dataset.path_dict['MER2024']
        csv1_path = os.path.join(mer2024_path, 'test1-corpus.csv')
        self.vis_transform = vis_transform
        self.img_transform = img_transform
        self.txt_transform = txt_transform
        
        self.df = pd.read_csv(csv1_path)
        # train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=419*1027, stratify=full_df["discrete"])

        if cfg.run.type == 'eval':
            self.processor = AlproVideoEvalProcessor()
            self.processor = self.processor.from_config(cfg.inference.vis_processor.train)
            self.dir = ''
            # self.df = test_df
        elif cfg.run.type == 'train':
            self.processor = AlproVideoTrainProcessor()
            self.processor = self.processor.from_config(cfg.model.vis_processor.train)
            self.dir = ''
            raise ValueError("MER2024 dataset does not support training mode yet.")
            # self.df = train_df
        else:
            raise ValueError("Invalid run type. Must be 'eval' or 'train'.")
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = os.path.join(os.path.join(self.cfg.dataset.path_dict['MER2024'], 'video'), row['name']+'.mp4')
        audio_path = os.path.join(os.path.join(self.cfg.dataset.path_dict['MER2024'], 'audio'), row['name']+'.wav')
        openface_path = os.path.join(os.path.join(self.cfg.dataset.path_dict['MER2024'], 'openface_face'), row['name']+'.npy')

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        elif not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        elif not os.path.exists(openface_path):
            raise FileNotFoundError(f"Openface file not found: {openface_path}")
        
        sample = super().read_frame_face_audio_text(video_path=video_path, audio_path=audio_path, face_path=openface_path, needed_data=self.needed_data)
        
        sample['discrete'] = row['emo']
        # sample['valence'] = row['valence']
        sample['chinese'] = row['chinese']
        # sample['english'] = row['english']
        sample['file'] = row['name']

        return sample
    
    def my_collate_fn(self, batch):
        # batch 是 list of sample dict
        frames = [item['frame'] for item in batch] if 'frame' in self.needed_data else None
        raw_frames = [item['raw_frame'] for item in batch] if 'frame' in self.needed_data else None
        audio = [item['audio'] for item in batch] if 'audio' in self.needed_data else None
        raw_audio = [item['raw_audio'] for item in batch] if 'audio' in self.needed_data else None
        face = [item['face'] for item in batch] if 'face' in self.needed_data else None
        raw_face = [item['raw_face'] for item in batch] if 'face' in self.needed_data else None
        discretes = [item['discrete'] for item in batch]
        # valences = [item['valence'] for item in batch]
        chinese = [item['chinese'] for item in batch]
        # english = [item['english'] for item in batch]
        file = [item['file'] for item in batch]

        return {
            'frame': torch.stack(frames) if frames is not None else None,
            'raw_frame': torch.stack(raw_frames) if raw_frames is not None else None,
            'audio': torch.stack(audio) if audio is not None else None,
            'raw_audio': torch.stack(raw_audio) if raw_audio is not None else None,
            'face': torch.stack(face) if face is not None else None,
            'raw_face': torch.stack(raw_face) if raw_face is not None else None,
            'discrete': discretes,
            'valence': None,
            'text': chinese,
            'text_en': None,
            'file': file
        }
        
    def get_test_name2gt(self):
        name2gt = {}
        for index, row in self.df.iterrows():
            name = row['name']
            val = row['emo']
            name2gt[name] = val
        return name2gt