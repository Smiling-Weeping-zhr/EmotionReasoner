"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re
import random
from typing import Iterable, List, Dict, Tuple

from omegaconf import OmegaConf
from qwen_omni_utils import process_mm_info
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate
# from utils.utils import get_emotion_synonyms

import re

def extract_label_content(text: str) -> str:
    """只提取 <label>...</label> 中的内容"""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return match.group(1).strip() if match else ""

# Train Args
QUESTION_TEMPLATE = [
    "As an expert in emotion analysis, infer the individual's emotion based on facial expressions, body language, environment, and subtitles, with detailed reasoning in <think></think> tags and the final answer in <answer></answer> tags.",
    "As an emotion analysis expert, analyze the person's emotion by examining facial cues, gestures, surrounding environment, and subtitle content. Provide detailed reasoning within <think> tags and give your final answer inside <answer></answer> tags.",
    "As a specialist in emotional understanding, deduce the individual's emotion using facial expressions, physical movements, contextual setting, and text subtitles. Explain your thought process in <think></think> and the final emotion in <answer></answer>.",
    "As an expert in recognizing human emotions, interpret the person's emotional state based on their appearance, body posture, surroundings, and the subtitle dialogue. Place your reasoning in <think></think> and your conclusion in <answer></answer>.",
    "As a professional in emotion recognition, determine the person's emotion by analyzing facial expressions, body actions, environment, and subtitles. Describe your reasoning inside <think></think> and your answer in <answer></answer>.",
    "As a trained analyst of emotions, use visual, behavioral, environmental, and textual clues to infer the subject's emotion. Justify your reasoning in <think> tags and state your final answer in <answer> tags.",
    "Analyze the subject's emotion from facial expressions, movements, context, and subtitles. Detail your reasoning in <think> and state your conclusion in <answer>.",
    "Using facial, bodily, environmental, and textual signals, infer the emotional state of the person. Explain your thinking in <think> and give the result in <answer>.",
    "From the person's expressions, posture, environment, and subtitles, infer their emotion. Reason in <think> and conclude in <answer>. <think> reasoning process here </think><answer> answer here </answer>",
]

class BaseDataset(Dataset):
    def __init__(self, processor):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.annotation = []
        self.processor = processor

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

    
    def get_infer_conv(self, cfg: OmegaConf, video_path: str, audio_path: str, image_path: str, subtitle: str) -> List:
        if cfg.run.eval.modality == 'video_audio_face':
            conversations = [
                {"role": "system", "content": [{"type": "text", "text": "You're an emotional expert, and answer me question."}]},
                {"role": "user", "content":  [
                    {"type": "video", "video": video_path},
                    {"type": "audio", "audio": audio_path},
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": f"Subtitle content of this video: {subtitle}; As an expert in the field of emotions, please focus on the facial expressions, body movements, environment, subtitle content, etc., in the video to discern clues related to the emotions of the individual. Please recognize all possible emotional states of the character."}
                ]}
            ]
        elif cfg.run.eval.modality == 'video_audio' and cfg.run.type == 'reason':
            conversations = [
                {"role": "system", "content": [{"type": "text", "text": "You're an emotional expert, and answer me question."}]},
                {"role": "user", "content":  [
                    {"type": "video", "video": video_path},
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": f"Subtitle content of this video: {subtitle}; As an expert in the field of emotions, please focus on the facial expressions, body movements, environment, subtitle content, etc., in the video to discern clues related to the emotions of the individual. Please infer the person's emotional state and provide your reasoning process."}
                ]}
            ]
        elif cfg.run.eval.modality == 'video_audio':
            conversations = [
                {"role": "system", "content": [{"type": "text", "text": "You're an emotional expert, and answer me question."}]},
                {"role": "user", "content":  [
                    {"type": "video", "video": video_path},
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": f"Subtitle content of this video: {subtitle}; As an expert in the field of emotions, please focus on the facial expressions, body movements, environment, subtitle content, etc., in the video to discern clues related to the emotions of the individual. Please recognize all possible emotional states of the character."}
                ]}
            ]
        else:
            raise ValueError(f"The modality {cfg.run.eval.modality} is not supported now.")
        
        return conversations
    
    def get_train_conv(self, cfg: OmegaConf, video_path: str, audio_path: str, image_path: str, subtitle: str) -> List:
        sys_promt = random.choices(QUESTION_TEMPLATE)
        if cfg.run.train.modality == 'video_audio':
            conversations = [
                {"role": "system", "content": [{"type": "text", "text": sys_promt}]},
                {"role": "user", "content":  [
                    {"type": "video", "video": video_path},
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": f"Subtitle content of this video: {subtitle}; \nPlease recognize all possible emotional states of the character."}
                ]},
            ]
        else:
            raise ValueError(f"The type {cfg.run.train.modality} is still not supported.")
        
        return conversations
    
    def process_video_audio_text(self, conversations: List) -> Dict:
        sample = {}
        
        text = self.processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)

        sample['inputs'] = inputs
        
        return sample
    
