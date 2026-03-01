import os
import pandas as pd

from moviepy import *
import os
import pandas as pd
from typing import List, Tuple
from model.Qwen_omni.datasets.base_dataset import BaseDataset

class MERCaptionPlus_Dataset(BaseDataset):
    def __init__(self, cfg, processor):
        super().__init__(processor=processor)
        self.cfg = cfg
        self.processor = processor
        
        mercaptionplus_path = self.cfg.dataset.path_dict['MERCaptionPlus']
        csv1_path = os.path.join(mercaptionplus_path, 'track2_train_mercaptionplus.csv')
        csv2_path = os.path.join(mercaptionplus_path, 'track3_train_mercaptionplus.csv')
        csv3_path = os.path.join(mercaptionplus_path, 'subtitle_chieng.csv')
        
        df1 = pd.read_csv(csv1_path)
        df2 = pd.read_csv(csv2_path)
        df3 = pd.read_csv(csv3_path)
        merged_df = pd.merge(df1, df2, on='name', how='inner')
        self.df = pd.merge(merged_df, df3, on='name', how='inner')
        
        base_path = self.cfg.dataset.path_dict['MERCaptionPlus']

        self.df['audio_path'] = self.df['name'].apply(lambda f: os.path.join(base_path, f'audio', f + '.wav'))
        self.df['video_path'] = self.df['name'].apply(lambda f: os.path.join(base_path, f'video', f + '.mp4'))
        self.df['answer_gt'] = self.df.apply(
            lambda row: f"<think>{row['reason']}</think>\n<answer>{row['openset'].strip('[]')}</answer>",
            axis=1
        )
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        openface_path = None
        audio_path = row['audio_path']
        video_path = row['video_path']
        sample = {}
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        elif not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if not self.cfg.run.train.fewshot and self.cfg.run.type == 'train':
            conversations = super().get_train_conv(cfg=self.cfg, video_path=video_path, audio_path=audio_path, image_path=openface_path, subtitle=row['chinese'] if pd.isna(row['english']) else row['english'])
        elif self.cfg.run.type == 'eval' and self.cfg.fewshot == True:
            conversations = super().get_fewshot_conv(cfg=self.cfg, fewshot_dict=self.fewshot_dict[row['name']], video_path=video_path, audio_path=audio_path, subtitle=row['chinese'] if pd.isna(row['english']) else row['english'])
            sample = super().process_video_audio_text(conversations=conversations)
        else:
            raise ValueError('we did not support this')
        
        sample['prompt'] = conversations
        sample['file'] = row['name']
        sample['english'] = row['english']
        sample['answer_gt'] = row['answer_gt']
        sample['data_type'] = self.cfg.run.train.modality
        sample['chinese'] = row['chinese']
        sample['openset'] = row['openset']
        sample['reason'] = row['reason']  
        sample['video_path'] = video_path
        sample['audio_path'] = audio_path
        
        return sample
    
    def my_collate_fn(self, batch):
        # input_ids_policy = [item['inputs'] for item in batch]
        prompt = [item['prompt'] for item in batch]
        file = [item['file'] for item in batch]
        answer_gt = [item['answer_gt'] for item in batch]
        modality = [item['data_type'] for item in batch]
        chinese = [item['chinese'] for item in batch]
        english = [item['english'] for item in batch]
        openset = [item['openset'] for item in batch]
        reason = [item['reason'] for item in batch]
        video_path = [item['video_path'] for item in batch]
        audio_path = [item['audio_path'] for item in batch]

        return {
            # 'input_ids_policy': input_ids_policy,
            'prompt': prompt,
            'file': file,
            'answer_gt': answer_gt,
            'modality': modality,
            'english': english,
            'chinese': chinese,
            'openset': openset,
            'reason': reason,
            'video_path': video_path,
            'audio_path': audio_path
        }
    
    def get_name2label(self):
        name2gt = {}
        for index, row in self.df.iterrows():
            name = row['name']
            val = row['answer_gt']
            name2gt[name] = val
        return name2gt