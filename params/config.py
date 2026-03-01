# *_*coding:utf-8 *_*
import os

## 所有涉及 transformers 的模型存储路径
AFFECTGPT_ROOT = './'
EMOTION_WHEEL_ROOT = './dataset/emotion_wheel'
RESULT_ROOT = os.path.join(AFFECTGPT_ROOT, 'output/results')


#######################
## 所有模型的存储路径
#######################
PATH_TO_LLM = {
    'Qwen25': '/root/autodl-tmp/EMER-Affect/Qwen2.5-7B-Instruct',
}

PATH_TO_VISUAL = {
    'CLIP_VIT_LARGE': '/root/autodl-tmp/EMER-Affect/clip-vit-large-patch14',

}

PATH_TO_AUDIO = {
    'HUBERT_LARGE':  '/root/autodl-tmp/EMER-Affect/chinese-hubert-large',
}


#######################
## 所有数据集的存储路径
#######################
# DATA_DIR = {
#     'MER2025OV':      '../dataset/mer2025-dataset',
#     'MERCaptionPlus': '../dataset/mer2025-dataset',
#     'OVMERD':         '../dataset/mer2025-dataset',
#     'MER2023':        '../dataset/mer2023-dataset-process',
#     'MER2024':        '../dataset/mer2024-dataset-process',
#     'IEMOCAPFour':    '../dataset/iemocap-process',
#     'CMUMOSI':        '../dataset/cmumosi-process',
#     'CMUMOSEI':       '../dataset/cmumosei-process', 
#     'SIMS':           '../dataset/sims-process',
#     'SIMSv2':         '../dataset/simsv2-process',
#     'MELD':           '../dataset/meld-process', 
# }
DATA_DIR = {
    "MER2024": "/root/autodl-tmp/dataset/MER2024",
}

PATH_TO_RAW_AUDIO = {
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'audio'),
}
PATH_TO_RAW_VIDEO = {
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'video'),

}
PATH_TO_RAW_FACE = {
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'openface_face'),
}
PATH_TO_TRANSCRIPTIONS = {
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'transcription_merge.csv'),
    # 'MER2025OV':  os.path.join(DATA_DIR['MER2025OV'], 'subtitle_chieng.csv'),
}
PATH_TO_LABEL = {
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'label-6way.npz'),
}


#######################
## store global values
#######################
DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
DEFAULT_AUDIO_PATCH_TOKEN = '<AudioHere>'
DEFAULT_FRAME_PATCH_TOKEN = '<FrameHere>'
DEFAULT_FACE_PATCH_TOKEN  = '<FaceHere>'
DEFAULT_MULTI_PATCH_TOKEN = '<MultiHere>'

few_shot = 16
DEFAULT_AUDIO_PATCH_TOKEN_ICL = ['<AudioHere1>', '<AudioHere2>', '<AudioHere3>', '<AudioHere4>', '<AudioHere5>', '<AudioHere6>', '<AudioHere7>', '<AudioHere8>', '<AudioHere9>', '<AudioHere10>', '<AudioHere11>', '<AudioHere12>', '<AudioHere13>', '<AudioHere14>', '<AudioHere15>', '<AudioHere16>']
DEFAULT_FRAME_PATCH_TOKEN_ICL = ['<FrameHere1>', '<FrameHere2>', '<FrameHere3>', '<FrameHere4>', '<FrameHere5>', '<FrameHere6>', '<FrameHere7>', '<FrameHere8>', '<FrameHere9>', '<FrameHere10>', '<FrameHere11>', '<FrameHere12>', '<FrameHere13>', '<FrameHere14>', '<FrameHere15>', '<FrameHere16>']
DEFAULT_FACE_PATCH_TOKEN_ICL = ['<FaceHere1>', '<FaceHere2>', '<FaceHere3>', '<FaceHere4>', '<FaceHere5>', '<FaceHere6>', '<FaceHere7>', '<FaceHere8>', '<FaceHere9>', '<FaceHere10>', '<FaceHere11>', '<FaceHere12>', '<FaceHere13>', '<FaceHere14>', '<FaceHere15>', '<FaceHere16>']
DEFAULT_MULTI_PATCH_TOKEN_ICL = ['<MultiHere1>', '<MultiHere2>', '<MultiHere3>', '<MultiHere4>', '<MultiHere5>', '<MultiHere6>', '<MultiHere7>', '<MultiHere8>', '<MultiHere9>', '<MultiHere10>', '<MultiHere11>', '<MultiHere12>', '<MultiHere13>', '<MultiHere14>', '<MultiHere15>', '<MultiHere16>']
IGNORE_INDEX = -100
