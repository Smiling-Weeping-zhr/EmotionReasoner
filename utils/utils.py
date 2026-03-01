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
from typing import Optional, List, Tuple, Dict, Union
from moviepy import *
from urllib.parse import urlparse

import pandas as pd
from iopath.common.download import download
from iopath.common.file_io import file_lock, g_pathmgr
from utils.registry import registry

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

def video_to_frames(video_path, out_dir, fps=2.0, min_frames=2, max_frames=5):
    """
    将视频拆成图片帧，并限制帧率和总帧数范围
    
    参数:
        video_path (str): 输入视频路径 (.mp4)
        out_dir (str): 输出文件夹路径
        fps (float): 目标帧率（每秒几帧）
        min_frames (int): 最少帧数
        max_frames (int): 最多帧数
    """
    os.makedirs(out_dir, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    # 获取视频参数
    video_fps = cap.get(cv2.CAP_PROP_FPS)              # 原始帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
    duration = total_frames / video_fps                # 视频时长（秒）

    # 计算目标帧数
    target_frames = int(fps * duration)
    target_frames = max(min_frames, min(max_frames, target_frames))

    # 均匀采样帧索引
    frame_indices = [int(i * total_frames / target_frames) for i in range(target_frames)]

    print(f"视频总帧数: {total_frames}, 时长: {duration:.2f}s, 采样帧数: {target_frames}")

    saved = 0
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if i in frame_indices:
            out_path = os.path.join(out_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(out_path, frame)
            if saved > max_frames:
                break
            saved += 1

    cap.release()
    print(f"完成！共保存 {saved} 帧到 {out_dir}/")

def get_IoU_prompt(video_path: Union[str, List[str]] = [], audio_path: str="", subtitle: str="", model_type: str='qwen-omni-rl') -> List:
    if model_type == 'qwen-omni-rl':
        return get_tam_frames_inputs_conversations_qwen_rl(video_path=video_path, audio_path=audio_path, subtitle=subtitle)
    elif model_type == 'qwen-omni':
        return get_tam_frames_inputs_conversations_qwen_omni(video_path=video_path, audio_path=audio_path, subtitle=subtitle)
    elif model_type == "qwen2_5_vl":
        return get_tam_frames_inputs_conversations_qwen_vl(video_path=video_path, subtitle=subtitle)
    else:
        raise ValueError(f"We didn't support the model type {model_type}")

def get_tam_frames_inputs_conversations_qwen_rl(video_path: Union[str, List[str]] = [], audio_path: str="", subtitle: str="") -> List:
    sys_prompt = random.choice(QUESTION_TEMPLATE)
    if isinstance(video_path, list):
        conversations = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                {"role": "user", "content":  [
                    {"type": "video", "video": video_path},
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": f"Subtitle content of this video: {subtitle}; \nPlease recognize all possible emotional states of the character."}
                ]},
            ]
    else:
        conversations = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                {"role": "user", "content":  [
                    {"type": "image", "image": video_path},
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": f"Subtitle content of this video: {subtitle}; As an expert in the field of emotions, please focus on the facial expressions, body movements, environment, subtitle content, etc., in the video to discern clues related to the emotions of the individual. Please recognize all possible emotional states of the character."}
                ]}
            ]

    return conversations

def get_tam_frames_inputs_conversations_qwen_omni(video_path: str, audio_path: str, subtitle: str) -> List:
    conversations = [
        {"role": "system", "content": [{"type": "text", "text": "You're an emotional expert, and answer me question."}]},
        {"role": "user", "content":  [
            {"type": "video", "video": video_path},
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": f"Subtitle content of this video: {subtitle}; As an expert in the field of emotions, please focus on the facial expressions, body movements, environment, subtitle content, etc., in the video to discern clues related to the emotions of the individual. Please recognize all possible emotional states of the character."}
        ]}
    ]

    return conversations

def get_tam_frames_inputs_conversations_qwen_vl(video_path: str, subtitle: str) -> List:
    conversations = [
        {"role": "system", "content": [{"type": "text", "text": "You're an emotional expert, and answer me question."}]},
        {"role": "user", "content":  [
            {"type": "video", "video": video_path},
            {"type": "text", "text": f"Subtitle content of this video: {subtitle}; As an expert in the field of emotions, please focus on the facial expressions, body movements, environment, subtitle content, etc., in the video to discern clues related to the emotions of the individual. Please recognize all possible emotional states of the character."}
        ]}
    ]

    return conversations