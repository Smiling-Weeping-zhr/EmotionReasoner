"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os 

def set_ola_env():
    os.environ['LOWRES_RESIZE'] = '384x32'
    os.environ['HIGHRES_BASE'] = '0x32'
    os.environ['VIDEO_RESIZE'] = "0x64"
    os.environ['VIDEO_MAXRES'] = "480"
    os.environ['VIDEO_MINRES'] = "288"
    os.environ['MAXRES'] = '1536'
    os.environ['MINRES'] = '0'
    os.environ['FORCE_NO_DOWNSAMPLE'] = '1'
    os.environ['LOAD_VISION_EARLY'] = '1' # The initial position is very important
    os.environ['PAD2STRIDE'] = '1'
    
    
import json
import logging
import torch
import pickle
import random
import re
import shutil
import os
import subprocess
import cv2
import numpy as np
from tqdm import tqdm
import ast
import ipdb
import glob
from typing import Optional, List, Tuple, Dict
from moviepy import *
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from iopath.common.download import download
from iopath.common.file_io import file_lock, g_pathmgr
from utils.registry import registry

emotion = {
    'positive': [
        "happy", "joyful", "cheerful", "optimistic", "enthusiastic", "confident", 
        "excited", "hopeful", "inspired", "grateful", "pleased", "content", 
        "delighted", "energetic", "loving", "kind", "friendly", "amused", 
        "peaceful", "proud", "motivated", "satisfied", "thrilled", "serene", 
        "upbeat", "radiant", "fulfilled", "generous", "playful", "affectionate", 
        "relaxed"
    ],
    'negative': [
        "sad", "angry", "frustrated", "anxious", "depressed", "annoyed", 
        "upset", "jealous", "stressed", "fearful", "pessimistic", "bitter", 
        "guilty", "miserable", "insecure", "lonely", "irritated", "resentful", 
        "bored", "confused", "hurt", "discouraged", "panicked", "hostile", 
        "embarrassed", "hateful", "regretful", "vulnerable", "moody", "enraged", "passive"
    ],
    'happy': ["joyful", "cheerful", "delighted", "glad", "content", "pleased",
              "ecstatic", "elated", "jubilant", "blissful", "overjoyed", "satisfied",
              "radiant", "buoyant", "upbeat", "thrilled", "exuberant", "merry", "lighthearted",
              "sunny", "gleeful", "euphoric", "chirpy", "jovial", "perky", "blithe", "gay", "chipper",
              "zestful", "sparkling"
    ],
    'worry': ["anxious", "nervous", "concerned", "uneasy", "apprehensive", "troubled",
                "distressed", "tense", "fearful", "fretful", "panicky", "agitated", "restless",
                "edgy", "worried", "distraught", "perturbed", "alarmed", "upset", "shaky", "jumpy",
                "timid", "disquieted", "disturbed", "neurotic", "tormented", "wary", "hesitant",
                "doubtful", "insecure"
    ],
    'neutral': ["unbiased", "impartial", "nonpartisan", "objective", "disinterested", "detached",
                "evenhanded", "equitable", "indifferent", "uninvolved", "reserved", "open-minded",
                "moderate", "fair", "aloof", "dispassionate", "balanced", "unprejudiced", "unemotional",
                "middle-of-the-road", "nonaligned", "apathetic", "uncommitted", "indecisive",
                "ambivalent", "cool", "calm", "steady", "level-headed", "temperate"
    ],
    'angry' : ["mad", "furious", "irate", "enraged", "annoyed", "resentful", "outraged", "indignant",
               "cross", "exasperated", "irritated", "infuriated", "vexed", "fuming", "incensed",
               "livid", "agitated", "wrathful", "peeved", "testy", "choleric","riled", "grumpy",
               "snappy", "stormy", "heated", "hostile"
    ],
    'surprised': ["astonished", "amazed", "startled", "shocked", "stunned", "astounded",
                  "speechless", "flabbergasted", "dumbfounded", "bewildered", "baffled",
                  "taken_aback", "aghast", "confounded", "disconcerted", "rattled", "upended",
                  "jarred", "shaken", "perplexed", "dazed", "nonplussed", "wide-eyed", "overwhelmed",
                  "incredulous", "unsettled", "jolted", "floored", "staggered", "blindsided"
    ],
    'sad': ["unhappy", "sorrowful", "depressed", "melancholy", "mournful", "gloomy", "downcast",
            "downhearted", "despondent", "dismal", "blue", "dispirited", "forlorn", "woeful",
            "heartbroken", "grief-stricken", "glum", "lugubrious", "morose", "wistful", "doleful",
            "crestfallen", "tearful", "dejected", "bummed", "low", "somber", "sullen", "brokenhearted", "troubled"
    ],
    'disgust': ["revolting", "repulsive", "repugnant", "nauseating", "sickening", "abhorrent",
                "loathsome", "vile", "gross", "offensive", "detestable", "disagreeable", "abominable",
                "foul", "displeasing", "distasteful", "obnoxious", "hideous", "unclean", "unpleasant",
                "horrid", "nasty", "beastly", "grotesque", "ghastly", "icky", "rank", "yucky",
                "scornful", "contaminated"
    ]
}

def get_emotion_synonyms(emo: str) -> str:
    emo_list = None
    if emo in ['happy', 'joy', 'happiness']:
        emo_list = emotion['happy']
    elif emo in ['sad', 'sadness']:
        emo_list = emotion['sad']
    elif emo in ['neutral']:
        emo_list = emotion['neutral']
    elif emo in ['angry', 'anger']:
        emo_list = emotion['angry']
    elif emo in ['disgust']:
        emo_list = emotion['disgust']
    elif emo in ['surprised', 'surprise']:
        emo_list = emotion['surprised']
    elif emo in ['fear', 'worry', 'worried']:
        emo_list = emotion['worry']
    elif emo in ['positive']:
        emo_list = emotion['positive']
    elif emo in ['negative']:
        emo_list = emotion['negative']
    else:
        raise ValueError(f"{emo} not exists.")
    
    num_words = random.randint(4, 5)
    selected = random.sample(emo_list, num_words)
    selected.append(emo)

    result = ', '.join(selected)
    
    return result


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def get_cache_path(rel_path):
    return os.path.expanduser(os.path.join(registry.get_path("cache_root"), rel_path))


def get_abs_path(rel_path):
    return os.path.join(registry.get_path("library_root"), rel_path)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def string_to_list(str):

    if type(str) == np.ndarray:
        str = str.tolist()

    # 如果本身就是list了，那么就不需要其他操作了
    if isinstance(str, list):
        return str
    
    if str == '':
        str = []
    elif pd.isna(str):
        str = []
    else:
        if str[0] == '[':  str = str[1:]
        if str[-1] == ']': str = str[:-1]
        str = [item.strip() for item in re.split('[\'\",]', str) if item.strip() not in ['', ',']]
    return str


def makedir(dir_path):
    """
    Create the directory if it does not exist.
    """
    is_success = False
    try:
        if not g_pathmgr.exists(dir_path):
            g_pathmgr.mkdirs(dir_path)
        is_success = True
    except BaseException:
        print(f"Error creating directory: {dir_path}")
    return is_success


def abspath(resource_path: str):
    """
    Make a path absolute, but take into account prefixes like
    "http://" or "manifold://"
    """
    regex = re.compile(r"^\w+://")
    if regex.match(resource_path) is None:
        return os.path.abspath(resource_path)
    else:
        return resource_path


def makedir(dir_path):
    """
    Create the directory if it does not exist.
    """
    is_success = False
    try:
        if not g_pathmgr.exists(dir_path):
            g_pathmgr.mkdirs(dir_path)
        is_success = True
    except BaseException:
        logging.info(f"Error creating directory: {dir_path}")
    return is_success


def is_url(input_url):
    """
    Check if an input string is a url. look for http(s):// and ignoring the case
    """
    is_url = re.match(r"^(?:http)s?://", input_url, re.IGNORECASE) is not None
    return is_url

def extract_face_npy_from_video_openface(
    video_path: str,
    openface_bin: str = "/path/to/OpenFace/build/bin/FeatureExtraction",
):
    """
    使用 OpenFace 处理视频，提取人脸并保存为单个 .npy 文件（形状 [n_frames, 112, 112, 3]）。
    当未检测到人脸时自动跳过。

    参数:
        video_path (str): 输入视频路径（如 `/data/03HDEC.mp4`）。
        openface_bin (str): OpenFace 的 FeatureExtraction 可执行文件路径。
    """
    # 获取视频所在目录的父目录（/path/to）
    parent_dir = os.path.dirname(video_path)
    grandparent_dir = os.path.dirname(parent_dir)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 设置中间文件和输出目录
    trash_dir = os.path.join(grandparent_dir, 'trash')
    output_dir = os.path.join(grandparent_dir, 'openface_face')
    os.makedirs(trash_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{video_name}.npy")
    if os.path.exists(output_path):
        return output_path
    # Step 1: 调用 OpenFace 处理视频
    cmd = [
        openface_bin,
        "-f", video_path,
        "-out_dir", trash_dir,
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        print(f"警告：OpenFace 处理视频失败（可能未检测到人脸），跳过: {video_path}")
        return None

    # Step 2: 检查是否生成对齐图像目录
    aligned_dir = os.path.join(trash_dir, f"{video_name}_aligned")
    if not os.path.exists(aligned_dir):
        print(f"警告：未检测到人脸，跳过视频: {video_path}")
        return None

    # Step 3: 检查目录中是否有.bmp文件
    bmp_files = [f for f in os.listdir(aligned_dir) if f.endswith(".bmp")]
    if not bmp_files:
        print(f"警告：无有效人脸图像，跳过视频: {video_path}")
        return None

    # Step 4: 处理有效的人脸图像
    try:
        # 按帧编号排序
        bmp_files = sorted(
            bmp_files,
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )

        # 合并所有帧
        frames = []
        for bmp_file in bmp_files:
            bmp_path = os.path.join(aligned_dir, bmp_file)
            img = cv2.imread(bmp_path)
            if img is None:
                continue
            if img.shape != (112, 112, 3):
                img = cv2.resize(img, (112, 112))
            frames.append(img)

        if not frames:
            print(f"警告：所有帧均无效，跳过视频: {video_path}")
            return None

        video_array = np.stack(frames, axis=0)

        # 保存结果
        
        np.save(output_path, video_array)
        return output_path

    except Exception as e:
        print(f"处理视频时出错（{video_path}）: {str(e)}")
        return None
    
def extract_face_npy_from_video_mtcnn(video_path, file_name, face_size=112, frame_interval=5):
    
    video_dir = os.path.dirname(video_path)
    parent_dir = os.path.abspath(os.path.join(video_dir, os.pardir))
    output_dir = os.path.join(parent_dir, "openface_face")
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, f"{file_name}.npy")
    if os.path.exists(save_path):
        # print(f"[INFO] Audio already exists: {audio_path}")
        return save_path
    
    mtcnn = MTCNN(image_size=face_size, margin=20, keep_all=False, post_process=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # print(f"[ERROR] Cannot open video {video_path}")
        return None

    faces = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval != 0:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_imgs = mtcnn(rgb_frame)

        if isinstance(face_imgs, torch.Tensor):
            if face_imgs.ndim == 4:
                for i in range(face_imgs.shape[0]):
                    # face_np = face_imgs[i].permute(1, 2, 0).int().numpy().astype(np.uint8)
                    face_np = (face_imgs[i].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
                    # face_np = (face_imgs[i].permute(1, 2, 0).clamp(0, 1) * 255).cpu().numpy()
                    # face_np = np.clip(face_np * 1.2, 0, 255).astype(np.uint8)
                    # face_np = face_imgs[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy().astype(np.float32)
                    faces.append(face_np)
            elif face_imgs.ndim == 3:
                # face_np = face_imgs.permute(1, 2, 0).int().numpy().astype(np.uint8)
                face_np = (face_imgs.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
                # face_np = (face_imgs.permute(1, 2, 0).clamp(0, 1) * 255).cpu().numpy()
                # face_np = np.clip(face_np * 1.2, 0, 255).astype(np.uint8)
                # face_np = face_imgs.permute(1, 2, 0).clamp(0, 1).cpu().numpy().astype(np.float32)
                faces.append(face_np)

    cap.release()

    if len(faces) == 0:
        print("[WARNING] No faces detected.")
        return None

    faces_array = np.stack(faces, axis=0)
    
    np.save(save_path, faces_array)
    # print(f"[SUCCESS] Saved {len(faces)} faces to {save_path}")

    return save_path

def extract_audio_16k_from_video(video_path):
    video_dir = os.path.dirname(video_path)

    parent_dir = os.path.abspath(os.path.join(video_dir, os.pardir))
    audio_dir = os.path.join(parent_dir, 'subaudio')
    file_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(audio_dir, exist_ok=True)

    audio_path = os.path.join(audio_dir, f"{file_name}.wav")
    if os.path.exists(audio_path):
        # print(f"[INFO] Audio already exists: {audio_path}")
        return audio_path

    # 使用ffmpeg提取音频
    cmd = f"ffmpeg -loglevel error -y -i \"{video_path}\" -ar 16000 -ac 1 \"{audio_path}\""
    # print(f"Extracting audio from {video_path} ...")
    ret = os.system(cmd)
    if ret != 0:
        # print(f"[ERROR] ffmpeg failed to extract audio from {video_path}")
        return None

    # print(f"[SUCCESS] Saved audio to {audio_path}")
    return audio_path

def cleanup_dir(dir):
    """
    Utility for deleting a directory. Useful for cleaning the storage space
    that contains various training artifacts like checkpoints, data etc.
    """
    if os.path.exists(dir):
        logging.info(f"Deleting directory: {dir}")
        shutil.rmtree(dir)
    logging.info(f"Deleted contents of directory: {dir}")

def get_file_size(filename):
    """
    Given a file, get the size of file in MB
    """
    size_in_mb = os.path.getsize(filename) / float(1024**2)
    return size_in_mb

def get_emotion_label(emo: str) -> str:
    """
    Get the emotion label from the given emotion string.
    """
    count = random.randint(4, 5)
    selected_words = random.sample(emotion[emo], count)
    result_str = ", ".join(selected_words)+f', {emo}'
    return result_str

def build_affectgpt_icl_prompt(demo_sample, test_subtitle, user_prompt, task_instruction=None, few_shot=16):
    """
    构建16-shot的四模态情感分析 ICL prompt

    :param examples: list of dicts, 每个 dict 含 audio, frames, faces, text, sentiment
    :param query_sample: dict, 不含 sentiment(待预测)
    :param task_instruction: str, 可选任务描述
    :return: str, 拼接后的 prompt
    """
    
    assert few_shot <= 16, f"Current few shot is {few_shot}, please make sure it is less than or equal to 32"

    if task_instruction is None:
        task_instruction = 'A video contains audio, frames, facial expressions of the main character, and text subtitles. ' \
                            + 'Please recognize all possible emotional states of the character ' \
                            + 'Here are some examples.'

    prompt = f"{task_instruction}\n"
    # ipdb.set_trace()
    for i in range(few_shot):
        # ipdb.set_trace()
        # emo_related_labels = get_emotion_label(demo_sample[i]['label'].lower())
        prompt += f"<Demonstration {i+1}>\n"
        prompt += f"###Human: The audio and video merged info is: <Multi><MultiHere{i+1}></Multi>." \
                    + f"The audio content is as follows: <Audio><AudioHere{i+1}></Audio>. " \
                    + f"Additionally, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere{i+1}></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{demo_sample[i]['subtitle']}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_prompt} " \
                    + f"###Assistant: The character's emotional state is {demo_sample[i]['label']} \n"
        prompt += f"<\Demonstration {i+1}>\n"
    
    prompt += f"<Test Sample>\n"
    prompt += f"###Human:Human: The audio and video merged info is: <Multi><MultiHere></Multi>." \
                + f"The audio content is as follows: <Audio><AudioHere></Audio>. " \
                + f"Additionally, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. "  \
                + f"The subtitle of this video is: <Subtitle>{test_subtitle}</Subtitle>. " \
                + f"Now, please answer my question based on all the provided information. {user_prompt} " \
                + f"###Assistant: \n"
    prompt += f"<\Test Sample>\n"
    
    return prompt


def get_prompt_for_multimodal(face_or_frame, subtitle, user_message):

        # step5: get prompts for differet cases [可能存在三种数据加载情况，从而能够扩展至4种模态输入]
        if face_or_frame == 'affectgpt_face_frame_audio_text': # (face, frame, audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + f"Additionally, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'affectgpt_face_audio_text': # (face, audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'affectgpt_frame_audio_text': # (frame, audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == "affectgpt_frame_text": # (frame, text)
            assert subtitle is not None
            prompt = f"###Human: We uniformly sample raw frames from the video: <Video><FrameHere></Video>. " \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'affectgpt_audio': # (audio)
            prompt = f"###Human: The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'affectgpt_text':  # (text)
            assert subtitle is not None
            prompt = f"###Human: The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'affectgpt_face':  # (face)
            prompt = f"###Human: We uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'affectgpt_frame': # (frame)
            prompt = f"###Human: We uniformly sample raw frames from the video: <Video><FrameHere></Video>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'affectgpt_image': # (image)
            prompt = f"###Human: The image content is as follows: <Image><ImageHere></Image>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'multiface_text': # (multi, text)
            assert subtitle is not None
            prompt = f"###Human: The audio and video merged info is: <Multi><MultiHere></Multi>. " \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'multiface_audio_face_text': # (multi, face, audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio and video merged info is: <Multi><MultiHere></Multi>. " \
                    + f"The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'multiframe_audio_frame_text': # (multi, face, audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio and video merged info is: <Multi><MultiHere></Multi>. " \
                    + f"The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'multiface_audio_face_frame_text': # (multi, frame, face, audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio and video merged info is: <Multi><MultiHere></Multi>. " \
                    + f"The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. "  \
                    + f"Meanwhile, we uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        return prompt
    
def extract_mer2024_prediction(response: str=None, emotion_words: List=[]) -> str:
    """
    Return the emotion word from the response (But only return the first emotion states).
    """
    for word in emotion_words:
        if word in response.lower():
            return word
    return 'unknown'  # 或者你可以选择跳过这些样本

def get_sentiment(response: str=None, emo2num: dict={}) -> int:
    for key in emo2num:
        if key in response:
            return emo2num[key]
    return -1 # not predict successfully, return a false answer, waf 

def get_hit_score(response: str=None, synonyms: List=[]) -> int:
    # response_list = ast.literal_eval(response)
    for synonym in synonyms:
        if synonym in response:
            return 1
    return 0

def get_emotion_wheels(labels: List) -> Dict:
    """
    Get the emotion wheels for datasets.
    """
    wheel1_csv = pd.read_excel('./dataset/emotion_wheel/wheel1.xlsx')
    wheel2_csv = pd.read_excel('./dataset/emotion_wheel/wheel2.xlsx')
    wheel3_csv = pd.read_excel('./dataset/emotion_wheel/wheel3.xlsx')
    wheel4_csv = pd.read_excel('./dataset/emotion_wheel/wheel4.xlsx')
    wheel5_csv = pd.read_excel('./dataset/emotion_wheel/wheel5.xlsx')
    
    format_csv = pd.read_csv('./dataset/emotion_wheel/format.csv')
    # synonym_csv = pd.read_excel('./dataset/emotion_wheel/synonym.xlsx')
    
    emo_synonyms = {}
    # ipdb.set_trace() 
    format_csv['format'] = format_csv['format'].fillna('')
    emo_synonyms = {
        row['name']: row['format'].split(',') 
        for _, row in format_csv.iterrows()
    }
    for label in labels:
        if label not in emo_synonyms:
            emo_synonyms[label] = [label] 
            
    # ipdb.set_trace()
    last_emo = None
    for wheel in [wheel1_csv, wheel2_csv, wheel3_csv, wheel4_csv, wheel5_csv]:
        for index, row in wheel1_csv.iterrows():
            # print(f"第{index}行：", row['level1'], row['level2'], row['level3']) 
            if pd.isna(row['level1']) == True:
                if pd.isna(row['level2']) == False:
                    emo_synonyms[last_emo].append(row['level2'].lower())
                if pd.isna(row['level3']) == False:
                    emo_synonyms[last_emo].append(row['level3'].lower())
            else:
                last_emo = row['level1'].lower()
                if last_emo not in emo_synonyms:
                    emo_synonyms[last_emo] = [last_emo.lower()]
                if pd.isna(row['level2']) == False:
                    emo_synonyms[last_emo].append(row['level2'].lower())
                if pd.isna(row['level3']) == False:
                    emo_synonyms[last_emo].append(row['level3'].lower())
    # ipdb.set_trace()
    return emo_synonyms