import os
import pandas as pd
import json

import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
import webdataset as wds
import cv2
from moviepy import *
import os
import pandas as pd
from dataset.base_dataset import BaseDataset
from processor.video_processor import AlproVideoEvalProcessor, AlproVideoTrainProcessor, load_face
from torch.utils.data import Dataset

class CMU_MOSI_Dataset(BaseDataset):
    def __init__(self, cfg, vis_transform=None, img_transform=None, txt_transform=None):
        super().__init__() # Initial BaseDataset
        self.cfg = cfg
        self.n_frms = None
        
        self.needed_data = super().get_needed_data(self.cfg.model.multi_type)
        cmu_path = self.cfg.dataset.path_dict['CMU-MOSI']
        csv1_path = os.path.join(cmu_path, 'label.csv')
        self.vis_transform = vis_transform
        self.img_transform = img_transform
        self.txt_transform = txt_transform
            
        
        df = pd.read_csv(csv1_path)

        if cfg.run.type == 'eval':
            self.processor = AlproVideoEvalProcessor()
            self.processor = self.processor.from_config(cfg.inference.vis_processor.train)
            self.df = df.loc[df['mode'] == 'test']
            self.dir = 'test'
        elif cfg.run.type == 'train':
            self.processor = AlproVideoTrainProcessor()
            self.processor = self.processor.from_config(cfg.model.vis_processor.train)
            self.df = df.loc[df['mode'] == 'train']
            self.dir = 'train'
            self.label_type_candidates = ['valence', 'sentiment']
            # self.label_type = cfg.run.label_type
        else:
            raise ValueError("Invalid run type. Must be 'eval' or 'train'.")
        
        self.df['file'] = self.df.apply(lambda row: f"{row['video_id']}_{row['clip_id']}", axis=1)
        files_to_remove = ['8qrpnFRGt2A_13']
        self.df = self.df[~self.df['file'].isin(files_to_remove)]
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        openface_path = os.path.join(os.path.join(self.cfg.dataset.path_dict['CMU-MOSI'], f'{self.dir}/openface_face'), row['file']+'.npy')
        audio_path = os.path.join(os.path.join(self.cfg.dataset.path_dict['CMU-MOSI'], f'{self.dir}/subaudio'), row['file']+'.wav')
        video_path = os.path.join(os.path.join(self.cfg.dataset.path_dict['CMU-MOSI'], f'{self.dir}/subvideo'), row['file']+'.mp4')
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        elif not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        elif not os.path.exists(openface_path):
            raise FileNotFoundError(f"Openface file not found: {openface_path}")

        sample = super().read_frame_face_audio_text(video_path=video_path, audio_path=audio_path, face_path=openface_path, needed_data=self.needed_data)
        sample['annotation'] = row['annotation']
        sample['label'] = row['label']
        sample['text'] = row['text']
        sample['file'] = row['file']
        
        # if self.cfg.run.type == 'train':
        #     cur_label_type = super().get_cur_label_type(self.label_type_candidates, self.label_type)
        #     qa_pair = self.get_qa_pairs(self.cfg.dataset.name, cur_label_type, sample)
        #     sample['qa_pair'] = qa_pair
            
        return sample
    
    def my_collate_fn(self, batch):

        frames = [item['frame'] for item in batch] if 'frame' in self.needed_data else None
        raw_frames = [item['raw_frame'] for item in batch] if 'frame' in self.needed_data else None
        audio = [item['audio'] for item in batch] if 'audio' in self.needed_data else None
        raw_audio = [item['raw_audio'] for item in batch] if 'audio' in self.needed_data else None
        face = [item['face'] for item in batch] if 'face' in self.needed_data else None
        raw_face = [item['raw_face'] for item in batch] if 'face' in self.needed_data else None
        annotation = [item['annotation'] for item in batch]
        label = [item['label'] for item in batch]
        text = [item['text'] for item in batch]
        # qa_pair = [item['qa_pair'] for item in batch] if self.cfg.run.type == 'train' else None
        qa_pair = None
        video_file = [item['file'] for item in batch]

        return {
            'frame': torch.stack(frames) if frames is not None else None,
            'raw_frame': torch.stack(raw_frames) if raw_frames is not None else None,
            'audio': torch.stack(audio) if audio is not None else None,
            'raw_audio': torch.stack(raw_audio) if raw_audio is not None else None,
            'face': torch.stack(face) if face is not None else None,
            'raw_face': torch.stack(raw_face) if raw_face is not None else None,
            'annotation': annotation,
            'label': torch.tensor(label),
            'text_en': text,
            'qa_pair': qa_pair,
            'file': video_file
        }
        
    def get_name2label(self): # (name, label_word)
        name2label = {}
        for index, row in self.df.iterrows():
            name = row['file']
            val = row['annotation']
            name2label[name] = val
        return name2label
    
    def get_test_name2gt(self):
        name2gt = {}
        for index, row in self.df.iterrows():
            name = row['file']
            val = row['label']
            name2gt[name] = val
        return name2gt