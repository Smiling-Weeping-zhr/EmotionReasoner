"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
from typing import Iterable
import random
import torch
import numpy as np
from PIL import Image

from model.Qwen_omni.ImageBind.data import load_audio, transform_audio
from processor.video_processor import load_video, load_face
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate

class BaseDataset(Dataset):
    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.annotation = []
        # print("ann paths", ann_paths)
        for ann_path in ann_paths:
            # print("ann_path", ann_path)
            ann = json.load(open(ann_path, "r"))
            if isinstance(ann, dict):
                self.annotation.extend(json.load(open(ann_path, "r"))['annotations'])
                # self.annotation.extend(json.load(open(ann_path, "r")))
            else:
                self.annotation.extend(json.load(open(ann_path, "r")))
    
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)
    
    def get_needed_data(self, face_or_frame):
        if face_or_frame == 'affectgpt_face_frame_audio_text': # (face, frame, audio, text)
            needed_data = ['audio', 'frame', 'face', 'text']
        elif face_or_frame == 'face': # (face, audio, text)
            needed_data = ['audio', 'face', 'text']
        elif face_or_frame == 'frame': # (frame, audio, text)
            needed_data = ['audio', 'frame', 'text']
        elif face_or_frame == 'audioonly': # (audio)
            needed_data = ['audio']
        elif face_or_frame == 'textonly':  # (text)
            needed_data = []
        elif face_or_frame == 'faceonly':  # (face)
            needed_data = ['face']
        elif face_or_frame == 'frameonly': # (frame)
            needed_data = ['frame']
        elif face_or_frame == 'multiface_text': # (multi, text)
            needed_data = ['face', 'audio', 'text']
        elif face_or_frame == 'multiface_audio_face_text': # (multi, face, audio, text)
            needed_data = ['face', 'audio', 'text']
        elif face_or_frame == 'image': # (image)
            needed_data = ['image', 'text']
        elif face_or_frame == 'multiframe_audio_frame_text': # (multi, face, audio, text)
            needed_data = ['frame', 'audio', 'text']
        elif face_or_frame == 'multiface_audio_face_frame_text': # (multi, face, audio, text)
            needed_data = ['frame', 'face', 'audio', 'text']
        elif face_or_frame == 'audio_text': # (audio, text)
            needed_data = ['audio', 'text']
        elif face_or_frame == 'face_text': # (face, text)
            needed_data = ['face', 'text']
        elif face_or_frame == 'frame_text': # (frame, text)
            needed_data = ['frame', 'text']
        return needed_data
    
    def read_frame_face_audio_text(self, video_path=None, face_path=None, audio_path=None, needed_data=None):

        sample_data = {}

        # step1: read (raw_frame, frame)
        frame, raw_frame = None, None
        if video_path is not None and 'frame' in needed_data:
            raw_frame, msg = load_video(
                video_path=video_path,
                n_frms = self.n_frms,
                height = 224,
                width  = 224,
                sampling ="uniform",
                return_msg = True
            )
            frame = self.vis_processor.transform(raw_frame) # [3, 8, 224, 224] # 建议可视化，看看这部分数据扩增是否合适
        sample_data['frame'] = frame
        sample_data['raw_frame'] = raw_frame
        # print (sample_data)

        # step2: read (raw_face, face)
        face, raw_face = None, None
        if face_path is not None and 'face' in needed_data:
            raw_face, msg = load_face(
                face_npy=face_path,
                n_frms = self.n_frms,
                height = 224,
                width  = 224,
                sampling ="uniform",
                return_msg=True
            )
            face = self.vis_processor.transform(raw_face) # [3, 8, 224, 224] # 建议可视化，看看这部分数据扩增是否合适
        sample_data['face'] = face
        sample_data['raw_face'] = raw_face
        # print (sample_data)

        # step3: read audio [需要针对没有 audio track 的 video 进行额外处理]
        audio, raw_audio = None, None
        if audio_path is not None and 'audio' in needed_data:
            raw_audio = load_audio([audio_path], "cpu", clips_per_video=8)[0] # [8, 1, 16000*2s]
            audio = transform_audio(raw_audio, "cpu") # [8, 1, 128, 204]
        sample_data['audio'] = audio
        sample_data['raw_audio'] = raw_audio

        return sample_data
    
    def get_cur_label_type(self, label_type_candidates, label_type):
        """
            label_type: 
                hybird: random choose different formats of prompts for training.
                label_type: a fixed data type for training.
            label_type_candidates: candidates from prompt.
        """
        if label_type == 'hybird':
            index = random.randint(0, len(label_type_candidates) -1)
            return label_type_candidates[index]
        else:
            assert label_type in label_type_candidates, f'error label type: {label_type} not in {label_type_candidates}'
            return label_type
    
    def func_get_qa_description(self, sample, question_only=False):
        question = "Please infer the person's emotional state and provide your reasoning process."

        if question_only:
            return question
        else:
            return {
                'question': question, 
                'answer':sample['description'],
                }
    
    def func_get_qa_ovlabel(self, sample, question_only=False):
        question = "Please recognize all possible emotional states of the character."

        if question_only:
            return question
        else:
            return {
                'question': question,
                'answer':  f"The character's emotional state is {sample['ovlabel']}."
                }
    
    def func_get_qa_onehot_w_candidates(self, sample, question_only=False):
        question = f"Please select the label that can best describe the person's emotional state from the provided candidate labels: {self.candidate_labels}."

        if question_only:
            return question
        else:
            return {
                'question': question,
                'answer':   f"The most likely label is {sample['onehot']}."
                }

    def func_get_qa_onehot_wo_candidates(self, sample, question_only=False):
        question = "Please recognize the character's most likely emotional state."

        if question_only:
            return question
        else:
            return {
                'question': question,
                'answer':  f"The character's emotional state is {sample['onehot']}."
                }

    def func_get_qa_valence(self, sample, question_only=False):
        question = f"Please identify the overall positive or negative emotional polarity of the main characters. " \
                 + f"The output should be a ﬂoating-point number ranging from {self.minval} to {self.maxval}. " \
                 + f"Here, {self.minval} indicates extremely negative emotions, 0 indicates neutral emotions, and {self.maxval} indicates extremely positive emotions. " \
                 + f"Please provide your judgment as a ﬂoating-point number."
        
        if question_only:
            return question
        else:
            return {
                'question': question,
                'answer': 'The valence score is %.2f.' %(sample['valence']),
                }

    def func_get_qa_sentiment(self, sample, question_only=False):
        question = "Please select the most likely sentiment label that can best describe the person's emotional state: positive, negative, neutral."
        
        if question_only:
            return question
        else:
            return {
                'question': question,
                'answer':  f"The character's sentiment state is {sample['sentiment']}.",
                }

    def func_get_qa_direct(self, sample):
        return {
            'question': sample['question'],
            'answer':   sample['answer'],
            }
    
    def func_get_qa_caption(self, sample, modality):
        if modality == 'image':
            return {
            'question': self.func_random_prompts(self.image_caption_prompt_candidates),
            'answer':   sample['caption'],
            }
        elif modality == 'audio':
            return {
            'question': self.func_random_prompts(self.audio_caption_prompt_candidates),
            'answer':   sample['caption'],
            }
    
    def func_get_qa_preference(self, sample):

        a1 = sample['preference']['a1']
        a2 = sample['preference']['a2']
        p  = sample['preference']['p']

        question = f"We provide two descriptions. a1: {a1} \t\t\t a2: {a2} Please select the one that best matches the video content."
        
        assert p in ['a1', 'a2', 'same']
        if p in ['a1', 'a2']:
            answer = f"The best one is {p}."
        else:
            answer = f'These two sentences describe the content of the video with the same accuracy.'

        return {
            'question': question,
            'answer':   answer,
            }

    # this (q, a) is used to determinate the reward value
    def func_get_description_reward(self, sample):
        reason = sample['description']
        reward = sample['reward']

        question = f"We have provided a description: {reason} \t\t\t Please evaluate and decide whether to accept or reject this description based on its alignment with the video content."

        assert reward in ['accept', 'reject']
        answer = f'{reward} this sentence.'

        return {
            'question': question,
            'answer':   answer,
        }

    ## 获取 <question, answer> 用于后续训练
    def get_qa_pairs(self, dataset, label_type, sample):
        
        '''
        Params:
            datasets: Basic, Sentiment
            label_type: description, valence, 
            sample
        '''
        # EMERFine 指的是 (training set) 那 332 samples，同时包含 ovlabel/description
        if dataset in ['EMERCoarse', 'EMERFine']:
            candidates = {
                'description': self.func_get_qa_description(sample),
                'ovlabel':     self.func_get_qa_ovlabel(sample),
            }
        
        elif dataset in ['EMERCoarseFilter']:
            candidates = {
                'description': self.func_get_qa_description(sample),
                'ovlabel':     self.func_get_qa_ovlabel(sample),
                'sentiment':   self.func_get_qa_sentiment(sample),
                'valence':     self.func_get_qa_valence(sample),
            }
        
        elif dataset in ['MERCaptionPlus', 'OVMERD']:
            candidates = {
                'description': self.func_get_qa_description(sample),
                'ovlabel':     self.func_get_qa_ovlabel(sample),
            }
        
        elif dataset in ['Preference']: # 带 preference 优化
            candidates = {
                'description': self.func_get_qa_description(sample),
                'ovlabel':     self.func_get_qa_ovlabel(sample),
                'sentiment':   self.func_get_qa_sentiment(sample),
                'valence':     self.func_get_qa_valence(sample),
                'preference':  self.func_get_qa_preference(sample),
            }

        elif dataset in ['Preference2', 'Preference4']: # 不带 preference 优化
            candidates = {
                'description': self.func_get_qa_description(sample),
                'ovlabel':     self.func_get_qa_ovlabel(sample),
                'sentiment':   self.func_get_qa_sentiment(sample),
                'valence':     self.func_get_qa_valence(sample),
            }
        
        elif dataset in ['Preference3']: # 不带 preference 优化
            candidates = {
                'reward': self.func_get_description_reward(sample),
            }
        
        ## case1: Zebang's labels
        elif dataset in ['MERRCoarse', 'MERRFine', 'MAFW']:
            candidates = {
                'description': self.func_get_qa_description(sample),
            }

        ## case2: onehot labels
        elif dataset in ['MER2023', 'MER2024', 'MELD', 'IEMOCAPFour']:
            candidates = {
                'onehot_w_candidates':  self.func_get_qa_onehot_w_candidates(sample),
                'onehot_wo_candidates': self.func_get_qa_onehot_wo_candidates(sample),
            }

        ## case3: valence scores
        elif dataset in ['CMUMOSI', 'CMUMOSEI', 'SIMS', 'SIMSv2']:
            candidates = {
                'valence':   self.func_get_qa_valence(sample),
                'sentiment': self.func_get_qa_sentiment(sample),
            }

        ## case4: instruction dataset
        elif dataset in ['VideoChat', 'LLaVA', 'EmoVIT']:
            candidates = {
                'qa':  self.func_get_qa_direct(sample),
            }

        elif dataset in ['MiniGPT4']:
            candidates = {
                'caption': self.func_get_qa_caption(sample, 'image'),
            }

        elif dataset in ['WavCaps', 'TextrolSpeech', 'PromptSpeech']:
            candidates = {
                'caption': self.func_get_qa_caption(sample, 'audio'),
            }

        return candidates[label_type] # 包含 question, answer 两部分内容



class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        # TODO For now only supports datasets with same underlying collater implementations

        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)