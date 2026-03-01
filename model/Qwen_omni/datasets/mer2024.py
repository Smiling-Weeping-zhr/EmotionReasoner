import os
import pandas as pd

from moviepy import *
import os
import pandas as pd
from typing import List, Tuple
from model.Qwen_omni.datasets.base_dataset import BaseDataset

class MER2024_Dataset(BaseDataset):
    def __init__(self, cfg, processor):
        super().__init__(processor=processor)
        self.cfg = cfg
        self.processor = processor
        
        mer2024_path = self.cfg.dataset.path_dict['MER2024']
        

        if cfg.run.type == 'eval':
            self.dir = 'test'
            csv1_path = os.path.join(mer2024_path, 'test1-corpus.csv')
            self.df = pd.read_csv(csv1_path)
        elif cfg.run.type == 'train':
            self.dir = 'train'
            csv1_path = os.path.join(mer2024_path, 'label-disdim.csv')
            csv2_path = os.path.join(mer2024_path, 'label-transcription.csv')
            df1 = pd.read_csv(csv1_path)
            df2 = pd.read_csv(csv2_path)
            self.df = pd.merge(df1, df2, on='name', how='inner')
            self.df = self.df.rename(columns={'discrete': 'emo'})
        else:
            raise ValueError("Invalid run type. Must be 'eval' or 'train'.")
        
        base_path = self.cfg.dataset.path_dict['MER2024']
        dir = self.dir  # 'train' or 'test'
        
        self.df['audio_path'] = self.df['name'].apply(lambda f: os.path.join(base_path, f'{dir}/audio', f + '.wav'))
        self.df['video_path'] = self.df['name'].apply(lambda f: os.path.join(base_path, f'{dir}/video', f + '.mp4'))
        
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
        
        if not self.cfg.fewshot:
            conversations = super().get_infer_conv(cfg=self.cfg, video_path=video_path, audio_path=audio_path, image_path=openface_path, subtitle=row['chinese'])
            sample = super().process_video_audio_text(conversations=conversations)
        else:
            conversations = super().get_fewshot_conv(cfg=self.cfg, fewshot_dict=self.fewshot_dict[row['name']], video_path=video_path, audio_path=audio_path, subtitle=row['chinese'])
            sample = super().process_video_audio_text(conversations=conversations)
        
        sample['discrete'] = row['emo']
        # sample['valence'] = row['valence']
        sample['chinese'] = row['chinese']
        sample['english'] = row['english'] if self.dir == 'train' else None
        sample['file'] = row['name']
        sample['video_path'] = video_path
        sample['audio_path'] = audio_path

        return sample
    
    def my_collate_fn(self, batch):
        inputs = [item['inputs'] for item in batch]
        discretes = [item['discrete'] for item in batch]
        # valences = [item['valence'] for item in batch]
        chinese = [item['chinese'] for item in batch]
        english = [item['english'] for item in batch] if self.dir == 'train' else None
        file = [item['file'] for item in batch]
        video_path = [item['video_path'] for item in batch]
        audio_path = [item['audio_path'] for item in batch]

        return {
            'inputs': inputs,
            'discrete': discretes,
            'valence': None,
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
            val = row['emo']
            name2gt[name] = val
        return name2gt
    
    get_name2label = get_test_name2gt