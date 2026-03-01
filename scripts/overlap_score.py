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
        "sk-JizBFGaB8TPayhbbHftnMi86Xs6gaDwNnqio54AFiC8BQaeU",
        "https://api.nuwaapi.com/v1"
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
    
    
    # cleaned_pred_text = re.sub(r'[^a-zA-Z0-9\s]', '', pred_text)
    # cleaned_standard_text = re.sub(r'[^a-zA-Z0-9\s]', '', standard_text)
    # tokenizer
    standard_text_tokens = nltk.word_tokenize(str(standard_text).lower())
    pred_tokens = nltk.word_tokenize(pred_text.lower())
    
    smoothie = SmoothingFunction().method4
    bleu1 = sentence_bleu([standard_text_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu4 = sentence_bleu([standard_text_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    
    # meteor_score = meteor([standard_text_tokens], pred_tokens)
    # meteor = safe_meteor_score(standard_text, pred_text)
    # meteor = None
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(standard_text, pred_text)
    rouge_l = rouge_scores['rougeL'].fmeasure


    return key, {
        'clue_reason': clue_reason,
        'emotion_reason': emotion_reason,
        'clue_score': clue_score,
        'emotion_score': emotion_score,
        'bleu1': bleu1,
        'bleu4': bleu4,
        # 'meteor': meteor_score,
        'rouge_l': rouge_l
    }

def calculate_overlab_clue_emotion(dir: str = "", cfg: Dict = None) -> None:
    dataset = cfg.dataset.name
    answer_file = cfg.dataset.path_dict[dataset]
    answer_path = os.path.join(answer_file, 'final-EMER-reason.csv')
    df = pd.read_csv(answer_path)
    answer_dict = df.set_index('name').to_dict('index')

    pred_pkl_path = os.path.join(dir, 'reason', f'{dataset}/{dataset}.pkl')
    with open(pred_pkl_path, 'rb') as f:
        prediction_dict = pickle.load(f)

    response = {}
    # Model-based metrics
    clue_score_list, emotion_score_list = [], []
    
    # Matching-based metrics
    bleu1_list, bleu4_list, meteor_list, rouge_l_list = [], [], [], []

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
                bleu1_list.append(result['bleu1'])
                bleu4_list.append(result['bleu4'])
                # meteor_list.append(result['meteor'])
                rouge_l_list.append(result['rouge_l'])
            except Exception as e:
                print(f"[ERROR] Failed to process {futures[future]}: {e}")
                traceback.print_exc()

    # 保存 response.pkl
    response_pkl_path = os.path.join(dir, 'reason', f'{dataset}/{dataset}_response.pkl')
    os.makedirs(os.path.dirname(response_pkl_path), exist_ok=True)
    with open(response_pkl_path, 'wb') as f:
        pickle.dump(response, f)

    # 读取旧结果并更新
    if cfg.model_name.endswith('icl'):
        json_path = f'./experiment/eval_{cfg.model_name}/result_fewshot{cfg.model.few_shot}.json'
    else:
        json_path = f'./experiment/eval_{cfg.model_name}/result.json'

    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            results = json.load(f)
    else:
        results = {f'{cfg.dataset.name}-Reason': {}}

    results[f'{cfg.dataset.name}-Reason']['Emotion score'] = sum(emotion_score_list) / len(emotion_score_list)
    results[f'{cfg.dataset.name}-Reason']['Clue score'] = sum(clue_score_list) / len(clue_score_list)
    results[f'{cfg.dataset.name}-Reason']['BLEU1'] = sum(bleu1_list) / len(bleu1_list)
    results[f'{cfg.dataset.name}-Reason']['BLEU4'] = sum(bleu4_list) / len(bleu4_list)
    # results[f'{cfg.dataset.name}-Reason']['METEOR'] = sum(meteor_list) / len(meteor_list)
    results[f'{cfg.dataset.name}-Reason']['ROUGE_L'] = sum(rouge_l_list) / len(rouge_l_list)

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    


