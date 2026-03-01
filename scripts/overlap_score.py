import numpy as np
import pandas as pd
import openai
import os
import pickle
from openai import OpenAI
from typing import Dict
from pydantic import ValidationError
from omegaconf import DictConfig, OmegaConf
import instructor
from tqdm import tqdm
import traceback
import json
import nltk
nltk.data.path.append('/root/autodl-tmp/nltk_data')
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
_ = wordnet.synsets("dog")

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate import meteor
from rouge_score import rouge_scorer
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import ipdb
import re

openai_api_and_url_list = [
    (
        "gpt-4o-mini",
        "",
        ""
    )
]

def get_openai_key():
    openai_api_and_url = openai_api_and_url_list[0]
    return openai_api_and_url


def get_completion(text=None, retry_total=10):
    """使用自定义API配置获取完成结果"""
    try:
        model, api_key, base_url = get_openai_key()
        
        client = instructor.patch(
            openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                max_retries=retry_total
            ),
            mode=instructor.Mode.MD_JSON
        )

        article_analysis = client.chat.completions.create(model=model, messages=text)
        return article_analysis.choices[0].message.content
    
    except ValidationError as e:
        for error in e.errors():
            if error['type'] == 'literal_error':
                invalid_value = error['ctx']['input_value']
                print(f"Invalid category value: {invalid_value}")
        raise e
    
    except Exception as e:
        print(f"API请求失败: {str(e)}")
        raise e

def func_clue_scoring(gt_reason, pred_reason):
    """
    evaluate clue overlap
    """
    prompt = f"""
                The following are the "true descriptions" and "predicted descriptions" of the characters. Please calculate the score of the "Prediction Description" according to the following steps. The score range is from 1 to 10. Ultimately, only the numerical value of the predicted score is output, along with the reason.
                1. Please summarize the descriptions of the characters' emotional states in "True Description"
                2. Please summarize the descriptions of the characters' emotional states in the "Prediction Description"
                3. Calculate the overlap degree between the "predicted description" and the "true description". The higher the overlap degree, the higher the returned score.
                4. The output format is: 'Predicted Score' : Predicted score; "Reason" : Reason
                Input:
                "True Description: {gt_reason}
                "Prediction Description": {pred_reason}
                Output

                """
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = get_completion(text=messages)
        
        return response
    except Exception as e:
        print(f"评分过程出错: {str(e)}")
        return None

def get_summarized_emotions(reason):
    """
    Get the emotion states for the emotion overlap
    """
    prompt = f"""
              Please infer the most likely emotional state of the characters in the video based on the following video description, and only output emotional words:
              Video description: Although he looks very happy, he is actually very anxious.
              Output result: Anxiety
              Video description: {reason}
              Output result:
              """
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = get_completion(text=messages)
        return response
    except Exception as e:
        print(f"评分过程出错: {str(e)}")
        return None

def func_label_scoring(gt_reason, pred_reason):
    """
    evaluate emotion overlap
    """
    prompt = f"""
                The following presents the "True Emotions" and "Predicted Emotions" of the characters. Please calculate the overlap degree between "Predicted Emotions" and "True Emotions". The higher the overlap degree, the higher the returned score. The score range is from 1 to 10. Ultimately, only the numerical value of the predicted score is output, along with the reason.
                The output format is: 'Predicted Score' : Predicted score; "Reason" : Reason
                Input:
                "True Emotions": {gt_reason}
                "Predicting Emotions": {pred_reason}
                Output:
                """
                
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = get_completion(text=messages)
        return response
    except Exception as e:
        print(f"评分过程出错: {str(e)}")
        return None

def func_LACE_scoring(explanation_text: str):
    """
    evaluate LACE score
    """
    prompt = f"""
As a fair evaluator,
your task is to assess a natural language explanation (explanation_text) for its internal logical coherence and clarity, and then assign a score.
You only need to evaluate whether the explanation itself is clear, reasonable, and self-consistent. There is no need to refer to any standard answer or external label.

Input Data:
Explanation: {explanation_text}

Evaluation Criteria

Logical Coherence:
Does the explanation present a clear, reasonable, and continuous reasoning process? Does it naturally build toward a conclusion without unnecessary detours or contradictions?

Structure and Clarity:
Does the explanation demonstrate good organization and separation of reasoning from conclusions? Is the presentation concise and easy to follow, making it straightforward to see how the reasoning supports the outcome?

Persuasiveness:
Does the explanation feel convincing and natural, with the conclusion clearly connected to the prior reasoning? Does the phrasing avoid vagueness, overgeneralization, or excessive speculation?

Scoring Scale (1-10)

10 = Extremely coherent and clear; reasoning is complete, focused, and strongly connected to the conclusion, with a well-structured presentation.
8-9 = Highly coherent; reasoning is clear and logical with good structure, though there may be minor shortcomings in conciseness or focus.
6-7 = Moderately coherent; reasoning is mostly sound but somewhat diffuse, with noticeable issues in structure, clarity, or precision of the conclusion.
4-5 = Weak coherence; reasoning and conclusion are loosely connected, overly verbose, or lack focus, making the logic less convincing.
2-3 = Very low coherence; reasoning is unclear, lacks structure, or contains significant breaks in logic.
1 = Completely incoherent; explanation is disorganized, illogical, or meaningless.

"Predicted Score" : <integer from 1 to 10>;
"Reason" : "<concise justification based on logical coherence, structural clarity, and persuasiveness>"

The output format is: 'Predicted Score' : Predicted score; "Reason" : Reason
"""
    messages = [{"role": "user", "content": prompt}]
    try:
        response = get_completion(text=messages)
        return response
    except Exception as e:
        print(f"评分过程出错: {str(e)}")
        return None

def func_MEER_scoring(explanation_text: str, ground_truth: str):
    prompt = f"""
You are an impartial evaluator for a multimodal emotion reasoning task.

Goal:
Evaluate how well a model-generated explanation demonstrates understanding of the emotional cues present in audio, video, and text, and how accurately these cues correspond to the ground-truth explanation. 
The focus is on the model's analysis of multimodal signals (facial expressions, body movements, tone of voice, background environment, dialogue content) and whether its reasoning stays relevant to the evidence.

Input Data:
Ground Truth Explanation:
{ground_truth}

Model Explanation:
{explanation_text}

Evaluation Aspects (provide a single holistic score from 1 to 10):
1. Multimodal Cue Analysis - Does the model identify and describe emotional cues from video, audio, and text (e.g., facial expressions, gestures, tone, environmental mood) rather than only abstract emotion labels?
2. Evidence Alignment - Are the cues and reasoning steps consistent with the ground-truth explanation? Does the model correctly reference or reject specific visual, acoustic, or textual details instead of inventing unsupported content?
3. Coherence and Focus - Is the explanation logically organized and focused on relevant multimodal evidence without drifting into unrelated or speculative descriptions?

Scoring Scale (1-10):
10 = Excellent: rich and accurate analysis of multimodal emotional cues with strong alignment to the ground truth and no irrelevant content.
8-9 = Strong: mostly accurate multimodal cue analysis with only minor omissions or slight wording issues.
6-7 = Moderate: captures some relevant cues but lacks depth, misses key signals, or shows occasional misalignment.
4-5 = Weak: superficial or incomplete analysis with noticeable irrelevant or incorrect details.
2-3 = Poor: little connection to multimodal evidence or frequent hallucinations.
1 = Very poor: completely irrelevant or incoherent explanation.

The output format is: 'Predicted Score' : Predicted score; "Reason" : Reason
"""
    messages = [{"role": "user", "content": prompt}]
    try:
        response = get_completion(text=messages)
        return response
    except Exception as e:
        print(f"评分过程出错: {str(e)}")
        return None

def func_UTAI_scoring(explanation_text: str):
    prompt = f""" 
Act as an expert evaluator simulating a domain-agnostic end-user who relies on a model's prediction and explanation to make decisions. Your goal is to assess how useful the explanation is for your understanding and trust.

Input Data:
- Explanation: {explanation_text}

Evaluation Criteria:  
Rate the explanation's usefulness based on how well it helps you, as a user:
1.  Understanding: Does the explanation clearly clarify WHY the model made this specific prediction? Does it demystify the outcome?
2.  Trust & Confidence: Does the explanation make you more or less confident in accepting the model's prediction? Does it feel like a "black box" or a justified decision?
3.  Actionability: Based on this explanation, could you imagine what to change in an input to alter the prediction? (e.g., "If I want a more positive result, I should use words like X instead of Y").
4.  Insight: Does the explanation provide non-obvious or meaningful insight into the features the model uses, beyond just listing words?

Scoring Scale:  
1-10 (integer scores only), where:
- 10: Exceptionally useful - greatly enhances understanding, trust, and provides clear actionable insights.
- 8-9: Very useful - clearly explains the rationale and builds confidence.
- 6-7: Moderately useful - provides some insight but may be vague or incomplete.
- 4-5: Slightly useful - offers minimal clarification or only states the obvious.
- 2-3: Not useful - confusing, misleading, or fails to justify the prediction.
- 1: Counterproductive - undermines trust or provides incorrect reasoning.

The output format is: 'Predicted Score' : Predicted score; "Reason" : Reason
"""
    messages = [{"role": "user", "content": prompt}]
    try:
        response = get_completion(text=messages)
        return response
    except Exception as e:
        print(f"评分过程出错: {str(e)}")
        return None
    
def extract_score(response_text):
    """
    从模型返回中提取 'Predicted Score' 后面的得分
    """
    match = re.search(r"Predicted Score['\"]?\s*[:：]?\s*(\d+)", response_text)
    if match:
        return int(match.group(1))
    else:
        print("无法提取得分，原始响应为：", response_text)
        return None
    
def process_single_item(key, value, answer_dict):
    if isinstance(value, list):
        pred_text = value[0] if len(value) > 0 else ","
    else:
        pred_text = str(value)
    
    clue_score = None
    emotion_score = None

    standard_text = answer_dict[key]['english']
    while clue_score is None:
        clue_reason = func_clue_scoring(gt_reason=standard_text, pred_reason=pred_text)
        clue_score = extract_score(clue_reason)

    while emotion_score is None:
        pred_emotion_states = get_summarized_emotions(pred_text)
        standard_emotion_states = get_summarized_emotions(standard_text)
        emotion_reason = func_label_scoring(gt_reason=standard_emotion_states, pred_reason=pred_emotion_states)
        emotion_score = extract_score(emotion_reason)

    return key, {
        'clue_reason': clue_reason,
        'emotion_reason': emotion_reason,
        'clue_score': clue_score,
        'emotion_score': emotion_score
    }

def process_our_metric_item(key, value, answer_dict):
    if isinstance(value, list):
        pred_text = value[0] if len(value) > 0 else ","
    else:
        pred_text = str(value)
    
    LACE_score = None 
    MEER_score = None
    UTAI_score = None

    gt_reason = answer_dict[key]['english']
    # ipdb.set_trace()
    while MEER_score is None:
        emotion_reason = func_MEER_scoring(explanation_text=pred_text, ground_truth=gt_reason)
        MEER_score = extract_score(emotion_reason)
        
    while LACE_score is None:
        LACE_reason = func_LACE_scoring(explanation_text=pred_text)
        LACE_score = extract_score(LACE_reason)
    
    while UTAI_score is None:
        UTAI_reason = func_UTAI_scoring(explanation_text=pred_text)
        UTAI_score = extract_score(UTAI_reason)
    
    return key, {
        'LACE_reason': LACE_reason,
        'UTAI_reason': UTAI_reason,
        'emotion_reason': emotion_reason,
        'LACE_score': LACE_score,
        'UTAI_score': UTAI_score,
        'MEER_score': MEER_score
    }
    
def calculate_overlab_clue_emotion(dir: str = "", cfg: Dict = None) -> None:
    dataset = cfg.dataset.name
    answer_file = cfg.dataset.path_dict[dataset]
    answer_path = os.path.join(answer_file, 'final-EMER-reason.csv')
    df = pd.read_csv(answer_path)
    answer_dict = df.set_index('name').to_dict('index')
    eval_dir = os.path.join(dir, f'results-ovmerd/{cfg.model_name}')
    if cfg.model_name in ['Chat-UniVi', 'LLaMA-VID', 'mPLUG-Owl', 'Otter', 'Qwen-Audio', 'SALMONN', 'Video-ChatGPT', 'Video-LLaVA', 'VideoChat', 'VideoChat2']:
        path = os.path.join(eval_dir, 'output-reason-7b-subtitle-eng.npz')
        data = np.load(path, allow_pickle=True)
        prediction_dict = data['name2reason'].item()
    elif cfg.model_name in ['AffectGPT', 'Emotion-LLaMA', 'Valley', 'Qwen-Omni', 'Qwen-Omni-Func2', 'Qwen-Omni-Func3', 'Qwen-Omni-Func3-SFT', 'Qwen-Omni-Func4',\
        'Qwen-Omni-Func4-SFT', 'Qwen-Omni-TW', 'Qwen-Omni-TW-SFT', 'Qwen-Omni-VAD', 'Qwen-Omni-VAD-Polarity', 'Qwen-Omni-VADTW-Polarity', 'Qwen2_5_VL', 'Qwen-Omni-offline', "Qwen-Omni-offline-fscore",\
        'Qwen-Omni-offline-fscore-coarse', 'SECap', 'OneLLM', 'TW_Cold', 'Qwen-Omni-offline-two']:
        pred_pkl_path = os.path.join(eval_dir, f'OV-MERD.pkl')
        with open(pred_pkl_path, 'rb') as f:
            prediction_dict = pickle.load(f)
    else:
        raise ValueError(f"Unsupported model name: {cfg.model_name}")

    response = {}
    # Model-based metrics
    clue_score_list, emotion_score_list = [], []

    with ProcessPoolExecutor(max_workers=24) as executor:
        futures = {
            executor.submit(process_single_item, key, value, answer_dict): key
            for key, value in prediction_dict.items()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                key, result = future.result()
                response[key] = result
                clue_score_list.append(result['clue_score'])
                emotion_score_list.append(result['emotion_score'])
            except Exception as e:
                print(f"[ERROR] Failed to process {futures[future]}: {e}")
                traceback.print_exc()

    # 保存 response.pkl
    response_pkl_path = os.path.join(dir, 'reason', f'{cfg.model_name}/{dataset}_response.pkl')
    os.makedirs(os.path.dirname(response_pkl_path), exist_ok=True)
    with open(response_pkl_path, 'wb') as f:
        pickle.dump(response, f)

    # 读取旧结果并更新
    if cfg.model_name.endswith('icl'):
        json_path = f'./experiment/eval_explanation/explanation_fewshot{cfg.model.few_shot}.json'
    else:
        json_path = f'./experiment/eval_explanation/explanation.json'

    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            results = json.load(f)
    else:
        results = {f'{cfg.model_name}-{cfg.dataset.name}': {}}

    if f'{cfg.model_name}-{cfg.dataset.name}' not in results:
        results[f'{cfg.model_name}-{cfg.dataset.name}'] = {}
        
    results[f'{cfg.model_name}-{cfg.dataset.name}']['Emotion score'] = sum(emotion_score_list) / len(emotion_score_list)
    results[f'{cfg.model_name}-{cfg.dataset.name}']['Clue score'] = sum(clue_score_list) / len(clue_score_list)

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)

def calculate_LACE_UTAI_MEER_metric(dir: str = "", cfg: Dict = None) -> None:
    dataset = cfg.dataset.name
    answer_file = cfg.dataset.path_dict[dataset]
    answer_path = os.path.join(answer_file, 'track2_train_ovmerd.csv')
    reason_path = os.path.join(answer_file, 'final-EMER-reason.csv')
    df_reason = pd.read_csv(reason_path)
    df_openset = pd.read_csv(answer_path)
    df = pd.merge(df_reason, df_openset, on='name', how='inner')
    answer_dict = df.set_index('name').to_dict('index')
    # ipdb.set_trace()
    eval_dir = os.path.join(dir, f'results-ovmerd/{cfg.model_name}')
    if cfg.model_name in ['Chat-UniVi', 'LLaMA-VID', 'mPLUG-Owl', 'Otter', 'Qwen-Audio', 'SALMONN', 'Video-ChatGPT', 'Video-LLaVA', 'VideoChat', 'VideoChat2']:
        path = os.path.join(eval_dir, 'output-reason-7b-subtitle-eng.npz')
        data = np.load(path, allow_pickle=True)
        prediction_dict = data['name2reason'].item()
    elif cfg.model_name in ['AffectGPT', 'Emotion-LLaMA', 'Valley', 'Qwen-Omni', 'Qwen-Omni-Func2', 'Qwen-Omni-Func3', 'Qwen-Omni-Func3-SFT', 'Qwen-Omni-Func4', 'Qwen-Omni-Func4-SFT', 'Qwen-Omni-TW',\
        'Qwen-Omni-TW-SFT', 'Qwen-Omni-VAD', 'Qwen-Omni-VAD-Polarity', 'Qwen-Omni-VADTW-Polarity', 'Qwen2_5_VL', 'Qwen-Omni-offline', "Qwen-Omni-offline-fscore", 'Qwen-Omni-offline-fscore-coarse', \
        'SECap', 'OneLLM', 'TW_Cold', 'Qwen-Omni-offline-two']:
        pred_pkl_path = os.path.join(eval_dir, f'OV-MERD.pkl')
        with open(pred_pkl_path, 'rb') as f:
            prediction_dict = pickle.load(f)
    else:
        raise ValueError(f"Unsupported model name: {cfg.model_name}")

    response = {}
    # Model-based metrics
    LACE_score_list, MEER_score_list, UTAI_score_list = [], [], []

    with ProcessPoolExecutor(max_workers=24) as executor:
        futures = {
            executor.submit(process_our_metric_item, key, value, answer_dict): key
            for key, value in prediction_dict.items()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                key, result = future.result()
                response[key] = result
                LACE_score_list.append(result['LACE_score'])
                MEER_score_list.append(result['MEER_score'])
                UTAI_score_list.append(result['UTAI_score'])
            except Exception as e:
                print(f"[ERROR] Failed to process {futures[future]}: {e}")
                traceback.print_exc()

    # 保存 response.pkl
    response_pkl_path = os.path.join(dir, 'reason', f'{cfg.model_name}/{dataset}_our_response.pkl')
    os.makedirs(os.path.dirname(response_pkl_path), exist_ok=True)
    with open(response_pkl_path, 'wb') as f:
        pickle.dump(response, f)

    # 读取旧结果并更新
    if cfg.model_name.endswith('icl'):
        json_path = f'./experiment/eval_explanation/explanation_fewshot{cfg.model.few_shot}.json'
    else:
        json_path = f'./experiment/eval_explanation/explanation_our.json'

    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            results = json.load(f)
    else:
        results = {f'{cfg.model_name}-{cfg.dataset.name}': {}}

    if f'{cfg.model_name}-{cfg.dataset.name}' not in results:
        results[f'{cfg.model_name}-{cfg.dataset.name}'] = {}
        
    results[f'{cfg.model_name}-{cfg.dataset.name}']['LACE score'] = sum(LACE_score_list) / len(LACE_score_list)
    results[f'{cfg.model_name}-{cfg.dataset.name}']['MEER score'] = sum(MEER_score_list) / len(MEER_score_list)
    results[f'{cfg.model_name}-{cfg.dataset.name}']['UTAI score'] = sum(UTAI_score_list) / len(UTAI_score_list)

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)