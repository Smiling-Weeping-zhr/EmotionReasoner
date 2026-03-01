import glob
import os
import json
import pickle
import random
import time
import itertools
import pandas as pd
import json

import torch
from torch.utils.data import Dataset
import webdataset as wds
import cv2
from moviepy import *
import os
import pandas as pd
import torchaudio
import torchaudio.transforms as T
from typing import List, Tuple
from dataset.base_dataset import BaseDataset
from processor.video_processor import AlproVideoEvalProcessor, AlproVideoTrainProcessor

class MER2023_Dataset(BaseDataset):
    def __init__(self, cfg, vis_transform=None, img_transform=None, txt_transform=None):

        self.cfg = cfg
        self.n_frms = None
        
        self.needed_data = super().get_needed_data(self.cfg.model.multi_type)
        mer2023_path = self.cfg.dataset.path_dict['MER2023']
        csv1_path = os.path.join(mer2023_path, 'test1-label.csv')
        csv2_path = os.path.join(mer2023_path, 'transcription-engchi-polish.csv')
        self.vis_transform = vis_transform
        self.img_transform = img_transform
        self.txt_transform = txt_transform
        
        df1 = pd.read_csv(csv1_path)
        df2 = pd.read_csv(csv2_path)

        self.df = pd.merge(df1, df2, on='name', how='inner')
        self.labels = self.df['discrete'].unique()
        if cfg.run.type == 'eval':
            self.processor = AlproVideoEvalProcessor()
            self.processor = self.processor.from_config(cfg.inference.vis_processor.train)
            self.dir = ''
        elif cfg.run.type == 'train':
            self.processor = AlproVideoTrainProcessor()
            self.processor = self.processor.from_config(cfg.model.vis_processor.train)
            self.dir = ''
        else:
            raise ValueError("Invalid run type. Must be 'eval' or 'train'.")
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = os.path.join(os.path.join(self.cfg.dataset.path_dict['MER2023'], 'video'), row['name']+'.avi')
        audio_path = os.path.join(os.path.join(self.cfg.dataset.path_dict['MER2023'], 'audio'), row['name']+'.wav')
        openface_path = os.path.join(os.path.join(self.cfg.dataset.path_dict['MER2023'], 'openface_face'), row['name']+'.npy')
    
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        elif not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        elif not os.path.exists(openface_path):
            raise FileNotFoundError(f"Openface file not found: {openface_path}")

        sample = super().read_frame_face_audio_text(video_path=video_path, audio_path=audio_path, face_path=openface_path, needed_data=self.needed_data)
        sample['discrete'] = row['discrete']
        sample['valence'] = row['valence']
        sample['text'] = row['chinese']
        sample['text_en'] = row['english']
        sample['file'] = row['name']
    
        return sample
    
    
    def get_labels(self) -> List:
        return self.labels
    
    def my_collate_fn(self, batch):
        # batch 是 list of sample dict
        frames = [item['frame'] for item in batch] if 'frame' in self.needed_data else None
        raw_frames = [item['raw_frame'] for item in batch] if 'frame' in self.needed_data else None
        audio = [item['audio'] for item in batch] if 'audio' in self.needed_data else None
        raw_audio = [item['raw_audio'] for item in batch] if 'audio' in self.needed_data else None
        face = [item['face'] for item in batch] if 'face' in self.needed_data else None
        raw_face = [item['raw_face'] for item in batch] if 'face' in self.needed_data else None
        chinese = [item['text'] for item in batch]
        english = [item['text_en'] for item in batch]
        discretes = [item['discrete'] for item in batch]
        valences = [item['valence'] for item in batch]
        video_file = [item['file'] for item in batch]

        return {
            'frame': torch.stack(frames) if frames is not None else None,
            'raw_frame': torch.stack(raw_frames) if raw_frames is not None else None,
            'audio': torch.stack(audio) if audio is not None else None,
            'raw_audio': torch.stack(raw_audio) if raw_audio is not None else None,
            'face': torch.stack(face) if face is not None else None,
            'raw_face': torch.stack(raw_face) if raw_face is not None else None,
            'text': chinese,
            'text_en': english,
            'discrete': discretes,
            'valence': torch.tensor(valences),
            'file': video_file
        }

    def get_test_name2gt(self):
        name2gt = {}
        for index, row in self.df.iterrows():
            name = row['name']
            val = row['discrete']
            name2gt[name] = val
        return name2gt