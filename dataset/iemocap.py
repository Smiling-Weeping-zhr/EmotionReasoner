import os
import pandas as pd
import json
from copy import deepcopy

import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
import webdataset as wds
import cv2
from moviepy import *
import os
import pandas as pd
import torchaudio
import torchaudio.transforms as T
from dataset.base_dataset import BaseDataset
from model.Qwen_omni.ImageBind.data import load_audio, transform_audio
from processor.video_processor import AlproVideoEvalProcessor, AlproVideoTrainProcessor
from torch.utils.data import Dataset

emos = ['happy', 'sad', 'neutral', 'anger']
emo2idx, idx2emo = {}, {}
for ii, emo in enumerate(emos):
    emo2idx[emo] = ii
    idx2emo[ii]  = emo
    
class IEMOCAP_Dataset(BaseDataset):
    def __init__(self, cfg, vis_transform=None, img_transform=None, txt_transform=None):
        super().__init__() # Initial BaseDataset
        self.cfg = cfg
        self.n_frms = None
        
        self.needed_data = super().get_needed_data(self.cfg.model.multi_type)
        iemocap_path = self.cfg.dataset.path_dict['IEMOCAP']
        csv1_path = os.path.join(iemocap_path, 'label_ses05.csv')
        self.vis_transform = vis_transform
        self.img_transform = img_transform
        self.txt_transform = txt_transform
            
        
        self.df = pd.read_csv(csv1_path)

        if cfg.run.type == 'eval':
            self.processor = AlproVideoEvalProcessor()
            self.processor = self.processor.from_config(cfg.inference.vis_processor.train)
            self.dir = 'test'
        elif cfg.run.type == 'train':
            self.processor = AlproVideoTrainProcessor()
            self.processor = self.processor.from_config(cfg.model.vis_processor.train)
            self.label_type_candidates = ['valence', 'sentiment']
            self.df['file'] = self.df['name']
            self.df['file_name'] = self.df['name']
            self.dir = 'train'
            # self.label_type = cfg.run.label_type
            # raise ValueError("IEMOCAP dataset does not support training mode yet.")
        else:
            raise ValueError("Invalid run type. Must be 'eval' or 'train'.")
    
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = os.path.join(os.path.join(self.cfg.dataset.path_dict['IEMOCAP'], 'subvideo-tgt'), row['name']+'.avi')
        audio_path = os.path.join(os.path.join(self.cfg.dataset.path_dict['IEMOCAP'], 'subaudio'), row['name']+'.wav')
        openface_path = os.path.join(os.path.join(self.cfg.dataset.path_dict['IEMOCAP'], 'openface_face'), row['name']+'.npy')
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        elif not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        elif not os.path.exists(openface_path):
            raise FileNotFoundError(f"Openface file not found: {openface_path}")
        
        sample = super().read_frame_face_audio_text(video_path=video_path, audio_path=audio_path, face_path=openface_path, needed_data=self.needed_data)
        sample['emo'] = idx2emo[row['emo']]
        sample['val'] = row['val']
        sample['text'] = row['chinese']
        sample['text_en'] = row['english']
        sample['file'] = row['name']
        
        # if self.cfg.run.type == 'train':
        #     cur_label_type = super().get_cur_label_type(self.label_type_candidates, self.label_type)
        #     qa_pair = self.get_qa_pairs(self.cfg.dataset.name, cur_label_type, sample)
        #     sample['qa_pair'] = qa_pair
            
        return sample
    
    def my_collate_fn(self, batch):
        # batch 是 list of sample dict
        frames = [item['frame'] for item in batch] if 'frame' in self.needed_data else None
        raw_frames = [item['raw_frame'] for item in batch] if 'frame' in self.needed_data else None
        audio = [item['audio'] for item in batch] if 'audio' in self.needed_data else None
        raw_audio = [item['raw_audio'] for item in batch] if 'audio' in self.needed_data else None
        face = [item['face'] for item in batch] if 'face' in self.needed_data else None
        raw_face = [item['raw_face'] for item in batch] if 'face' in self.needed_data else None
        emo = [item['emo'] for item in batch]
        val = [item['val'] for item in batch]
        text = [item['text'] for item in batch]
        text_en = [item['text_en'] for item in batch]
        qa_pair = None
        # qa_pair = [item['qa_pair'] for item in batch] if self.cfg.run.type == 'train' else None
        video_file = [item['file'] for item in batch]

        return {
            'frame': torch.stack(frames) if frames is not None else None,
            'raw_frame': torch.stack(raw_frames) if raw_frames is not None else None,
            'audio': torch.stack(audio) if audio is not None else None,
            'raw_audio': torch.stack(raw_audio) if raw_audio is not None else None,
            'face': torch.stack(face) if face is not None else None,
            'raw_face': torch.stack(raw_face) if raw_face is not None else None,
            'emo': emo,
            'val': torch.tensor(val),
            'text': text,
            'text_en': text_en,
            'qa_pair': qa_pair,
            'file': video_file
        }
        
    def get_test_name2gt(self):
        name2gt = {}
        for index, row in self.df.iterrows():
            name = row['name']
            val = idx2emo[row['emo']]
            name2gt[name] = val
        return name2gt