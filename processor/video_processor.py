"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import cv2
import decord
import numpy as np
import random as rnd
import random
from omegaconf import OmegaConf
from typing import Tuple
import ipdb

import torch
import os
from decord import VideoReader, cpu
from torchvision import transforms
from dataset.transforms_video import RandomResizedCropVideo, ToTensorVideo, NormalizeVideo
from processor.base_processor import BaseProcessor
import dataset.functional_video as F
from utils.registry import registry

# max video frames
MAX_INT = 2048
# 将 Decord 解码的视频帧自动转换为 PyTorch 张量（Tensor）
decord.bridge.set_bridge("torch")

## video -> sampled frames

def load_video_decord(video_path, n_frms=MAX_INT, height=-1, width=-1, sampling="uniform", return_msg=False):
    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_path, height=height, width=width)

    vlen = len(vr)
    start, end = 0, vlen

    n_frms_update = min(n_frms, vlen) # for vlen < n_frms, only read vlen

    if sampling == "uniform": # 均匀采样
        indices = np.arange(start, end, vlen / n_frms_update).astype(int).tolist()
    elif sampling == "headtail": # 前面随机采一半；后面随机采一半
        indices_h = sorted(rnd.sample(range(vlen // 2), n_frms_update // 2))
        indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms_update // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError

    #########################################
    ## for vlen < n_frms, pad into n_frms
    while len(indices) < n_frms:
        indices.append(indices[-1])
    #########################################

    # get_batch -> T, H, W, C
    temp_frms = vr.get_batch(indices) # 这块报错 [h264 @ 0xc97e880] mmco: unref short failure => 这通常的是视频本身的问题
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)

    if not return_msg:
        return frms

    fps = float(vr.get_avg_fps())
    sec = ", ".join([str(round(f / fps, 1)) for f in indices])
    msg = f"The video contains {len(indices)} frames sampled at {sec} seconds. "
    return frms, msg


def load_video(video_path: str,
               n_frms: int = -1,
               height: int = -1,
               width: int = -1,
               sampling: str = "uniform",
               return_msg: bool = False
               ) -> Tuple[torch.Tensor, str]:
    """Load and process a video file into a tensor.
    
    Args:
        video_path: Path to the video file.
        n_frms: Number of frames to sample. -1 means all frames.
        height: Target frame height. -1 means keep original.
        width: Target frame width. -1 means keep original.
        sampling: Sampling method ('uniform' or 'headtail').
        return_msg: Whether to return additional info message.
    
    Returns:
        A tensor of shape (C, T, H, W) and optionally an info message.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file {video_path}")

    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Handle case where video has no frames
    if vlen == 0:
        cap.release()
        raise RuntimeError(f"Video {video_path} has no frames")

    # Determine number of frames to sample
    n_frms_update = vlen if n_frms < 0 or n_frms > vlen else n_frms

    # Sample indices
    if sampling == "uniform":
        indices = np.linspace(0, vlen - 1, n_frms_update).astype(int).tolist()
    elif sampling == "headtail":
        half = n_frms_update // 2
        half = min(half, vlen // 2)  # Ensure we don't sample beyond half
        tail_count = n_frms_update - half
        tail_count = min(tail_count, vlen - vlen // 2)
        
        # Handle case where video is too short
        if vlen < 2:
            indices = [0] * n_frms_update
        else:
            indices_h = sorted(rnd.sample(range(vlen // 2), half))
            indices_t = sorted(rnd.sample(range(vlen // 2, vlen), tail_count))
            indices = indices_h + indices_t
    else:
        cap.release()
        raise NotImplementedError(f"Sampling method '{sampling}' is not implemented.")

    # Pad indices if needed (only if video is shorter than requested frames)
    while len(indices) < n_frms_update:
        indices.append(indices[-1] if indices else 0)

    # Determine default dimensions if not specified
    if height <= 0 or width <= 0:
        # Get one frame to check original dimensions
        ret, frame = cap.read()
        if ret:
            default_height, default_width = frame.shape[:2]
        else:
            default_height, default_width = 224, 224
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            # Create dummy frame with proper dimensions
            h = height if height > 0 else default_height
            w = width if width > 0 else default_width
            dummy = np.zeros((h, w, 3), dtype=np.uint8)
            frames.append(dummy)
            continue

        # Resize if needed
        if height > 0 and width > 0:
            frame = cv2.resize(frame, (width, height))
        
        # Convert color space
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    # Convert to tensor
    try:
        frames_np = np.stack(frames, axis=0)
        frames_tensor = torch.from_numpy(frames_np)
        frms = frames_tensor.permute(3, 0, 1, 2).float()
    except Exception as e:
        raise RuntimeError(f"Failed to convert frames to tensor: {str(e)}")

    # Prepare message if needed
    msg = ""
    if return_msg:
        times = [round(f / fps, 1) for f in indices]
        sec = ", ".join(map(str, times))
        msg = f"The video contains {len(indices)} frames sampled at {sec} seconds."

    return (frms, msg) if return_msg else frms

## 读取并采样人脸信息 [这个就不包括]
def load_face(face_npy, n_frms=MAX_INT, height=-1, width=-1, sampling="uniform", return_msg=False):
    
    faces = np.load(face_npy)
    faces = [cv2.resize(face, (width, height)) for face in faces] # [seqlen, 224, 224, 3]

    vlen = len(faces)
    start, end = 0, vlen

    n_frms_update = min(n_frms, vlen) # for vlen < n_frms, only read vlen

    if sampling == "uniform": # 均匀采样
        indices = np.arange(start, end, vlen / n_frms_update).astype(int).tolist()
    elif sampling == "headtail": # 前面随机采一半；后面随机采一半
        indices_h = sorted(rnd.sample(range(vlen // 2), n_frms_update // 2))
        indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms_update // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError

    #########################################
    ## for vlen < n_frms, pad into n_frms
    while len(indices) < n_frms:
        indices.append(indices[-1])
    #########################################

    # get_batch -> T, H, W, C
    temp_frms = np.array(faces)[indices]
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)

    if not return_msg:
        return frms

    msg = "We read faces in this time."
    return frms, msg

class AlproVideoBaseProcessor(BaseProcessor):
    def __init__(self, mean: float=None, std: float=None, n_frms: int=MAX_INT):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)
        # self.normalize = transforms.Normalize(mean, std)
        self.normalize = NormalizeVideo(mean, std)
        self.n_frms = n_frms
        
class ToUint8(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.to(torch.uint8)

    def __repr__(self):
        return self.__class__.__name__
    
class ToTHWC(object):
    """
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (C, T, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, H, W, C)
    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.permute(1, 2, 3, 0)

    def __repr__(self):
        return self.__class__.__name__
    
class ResizeVideo(object):
    def __init__(self, target_size, interpolation_mode="bilinear"):
        self.target_size = target_size
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (C, T, crop_size, crop_size)
        """
        return F.resize(clip, self.target_size, self.interpolation_mode)

    def __repr__(self):
        return self.__class__.__name__ + "(resize_size={0})".format(self.target_size)


# default: image_size=224; n_frms=8;
@registry.register_processor("alpro_video_train")
class AlproVideoTrainProcessor(AlproVideoBaseProcessor):
    def __init__(
        self,
        image_size=384,
        mean=None,
        std=None,
        min_scale=0.5,
        max_scale=1.0,
        n_frms=MAX_INT,
    ):
        super().__init__(mean=mean, std=std, n_frms=n_frms)

        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                # Video size is (C, T, H, W) -> 图像随机裁剪后，放缩到与输入图片相同尺度 (C, T, H, W)
                RandomResizedCropVideo(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation_mode="bicubic",
                ),
                ToTHWC(),  # C, T, H, W -> T, H, W, C
                ToUint8(),
                ToTensorVideo(),  # T, H, W, C -> C, T, H, W
                self.normalize, # 依赖于基类的图像 (mean, std) 处理器
            ]
        )

    def __call__(self, vpath):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: video clip after transforms. Size is (C, T, size, size).
        """
        clip = load_video(
            video_path=vpath,
            n_frms=self.n_frms,
            height=self.image_size,
            width=self.image_size,
            sampling="headtail",
        )

        return self.transform(clip)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()
        image_size = cfg.get("image_size", 256)
        mean = cfg.get("mean", None)
        std = cfg.get("std", None)
        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)
        n_frms = cfg.get("n_frms", MAX_INT)
        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
            n_frms=n_frms,
        )

"""
There is a clear distinction between video preprocessing in the training and evaluation stages. 
During training, random cropping and scaling are applied to perform data augmentation, which helps 
improve the model's generalization ability. In contrast, these random operations are removed during 
evaluation to maintain consistent input and ensure reliable results. Both stages apply normalization,
but the output in training is variable, whereas in evaluation it is consistent, enabling fair and reproducible 
assessment of model performance.
"""

@registry.register_processor("alpro_video_eval")
class AlproVideoEvalProcessor(AlproVideoBaseProcessor):
    def __init__(self, image_size=256, mean=None, std=None, n_frms=MAX_INT):
        super().__init__(mean=mean, std=std, n_frms=n_frms)

        self.image_size = image_size
        # self.n_frms = n_frms
        # Input video size is (C, T, H, W)
        self.transform = transforms.Compose(
            [   
                ## 在 eval 时候，删除了随机裁剪的操作，从而保证 eval 阶段的一致性
                # frames will be resized during decord loading.
                ToUint8(),  # C, T, H, W
                ToTHWC(),  # T, H, W, C
                ToTensorVideo(),  # C, T, H, W
                self.normalize,  # C, T, H, W
            ]
        )

    def __call__(self, vpath):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: raw video clip. Size is (C, T, size, size).
            torch.tensor: video clip after transforms. Size is (C, T, size, size).
        """
        clip = load_video(
            video_path=vpath,
            n_frms=self.n_frms,
            height=self.image_size,
            width=self.image_size,
        )

        # ipdb.set_trace()
        return clip, self.transform(clip)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()
        image_size = cfg.get("image_size", 256)
        mean = cfg.get("mean", None)
        std = cfg.get("std", None)
        n_frms = cfg.get("n_frms", MAX_INT)
        return cls(image_size=image_size, mean=mean, std=std, n_frms=n_frms)
