import os
import pandas as pd
import json
from copy import deepcopy

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import webdataset as wds
from moviepy import *
import os
import pandas as pd
from dataset.base_dataset import BaseDataset
from utils.utils import string_to_list
from model.Qwen_omni.ImageBind.data import load_audio, transform_audio
from processor.video_processor import AlproVideoEvalProcessor, AlproVideoTrainProcessor

class OVMERD_Dataset(BaseDataset):
    def __init__(self, cfg, vis_transform=None, img_transform=None, txt_transform=None):

        self.cfg = cfg
        self.n_frms = None
        
        self.needed_data = super().get_needed_data(self.cfg.model.multi_type)
        ovmerd_path = self.cfg.dataset.path_dict['OV-MERD']
        csv1_path = os.path.join(ovmerd_path, 'final-EMER-reason.csv')
        csv2_path = os.path.join(ovmerd_path, 'track2_train_ovmerd.csv')
        csv3_path = os.path.join(ovmerd_path, 'subtitle_chieng.csv')
        
        df1 = pd.read_csv(csv1_path).rename(columns={'english': 'reason_en', 'chinese': 'reason_ch'})
        df2 = pd.read_csv(csv2_path)
        df3 = pd.read_csv(csv3_path)

        self.vis_transform = vis_transform
        self.img_transform = img_transform
        self.txt_transform = txt_transform
        
        df = pd.merge(df1, df2, on="name", how="inner")
        self.df = pd.merge(df, df3, on="name", how="inner")
        # train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=419*1027, stratify=full_df["discrete"])

        if cfg.run.type == 'eval' or cfg.run.type == 'train': # OV-MERD don't have train datasets
            self.processor = AlproVideoEvalProcessor()
            self.processor = self.processor.from_config(cfg.inference.vis_processor.train)
            self.dir = ''
            # self.df = test_df
        # elif cfg.run.type == 'train':
        #     self.processor = AlproVideoTrainProcessor()
        #     self.processor = self.processor.from_config(cfg.model.vis_processor.train)
        #     raise ValueError("MER2024 dataset does not support training mode yet.")
        #     # self.df = train_df
        else:
            raise ValueError("Invalid run type. Must be 'eval' or 'train'.")
    
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
        
        sample = super().read_frame_face_audio_text(video_path=video_path, audio_path=audio_path, face_path=openface_path, needed_data=self.needed_data)
        
        sample['reason_en'] = row['reason_en']
        sample['reason_ch'] = row['reason_ch']
        sample['english'] = row['english']
        # sample['valence'] = row['valence']
        sample['openset'] = row['openset']
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
        english = [item['english'] for item in batch]
        chinese = [item['chinese'] for item in batch]
        reason_en = [item['reason_en'] for item in batch]
        openset = [item['openset'] for item in batch]
        reason_ch = [item['reason_ch'] for item in batch]
        # english = [item['english'] for item in batch]
        file = [item['file'] for item in batch]

        return {
            'frame': torch.stack(frames) if frames is not None else None,
            'raw_frame': torch.stack(raw_frames) if raw_frames is not None else None,
            'audio': torch.stack(audio) if audio is not None else None,
            'raw_audio': torch.stack(raw_audio) if raw_audio is not None else None,
            'face': torch.stack(face) if face is not None else None,
            'raw_face': torch.stack(raw_face) if raw_face is not None else None,
            'discrete': None,
            'reason_ch': reason_ch,
            'reason_en': reason_en,
            'openset': openset,
            'text': chinese,
            'text_en': english,
            'file': file
        }
        
    def get_test_name2gt(self):
        name2gt = {}
        for index, row in self.df.iterrows():
            name = row['name']
            openset_list = string_to_list(row['openset'])
            if len(openset_list) == 0: openset_list = ['neutral']
            name2gt[name] = ', '.join(openset_list)
        return name2gt