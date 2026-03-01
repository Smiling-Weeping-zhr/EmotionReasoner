import numpy as np
import json
import re
from typing import List

def get_prompt(eval_type: str=None) -> str:
    if eval_type == 'sentiment':
        prompt = """Please act as an expert in the field of emotions. \
            We provide a few words to describe the emotions of a character. \
            Please choose the most likely sentiment from the given candidates: [positive, negative, neutral] \
            Please direct output answer without analyzing process. \
            Input: [joyful]; Output: positive \
            Input: []; Output: neutral \
            Input: {}; Output: """
    elif eval_type == 'basic':
        prompt = """Please assume the role of an expert in the field of emotions. \
            We provide clues that may be related to the emotions of the characters. Based on the provided clues, please identify the emotional states of the main character. \
            The main character is the one with the most detailed clues. \
            Please separate different emotional categories with commas and output only the clearly identifiable emotional categories in a list format. \
            If none are identified, please output an empty list. \
            Input: We cannot recognize his emotional state; Output: [] \
            Input: His emotional state is happy, sad, and angry; Output: [happy, sad, angry] \
            Input: {}; Output: """
    elif eval_type == 'reason':
        prompt = """
        Please assume the role of an expert in the field of emotions. \
        We provide clues that may be related to the emotions of the characters. Based on the provided clues, please identify the emotional states of the main character. \
        The main character is the one with the most detailed clues. \
        Please separate different emotional categories with commas and output only the clearly identifiable emotional categories in a list format. \
        If none are identified, please output an empty list. \
        Input: We cannot recognize his emotional state; Output: [] \
        Input: His emotional state is happy, sad, and angry; Output: [happy, sad, angry] \
        Input: {reason}; Output: """
    else:
        raise ValueError("We didn't support the eval_type")

    return prompt

def prompt_func_extract_is_emo(words: List[str]=None) -> str:
    prompt = f"""You are a linguistics expert specializing in sentiment analysis. Your task is to determine whether a given word is an "emotion word".  
            Definition of emotion word: A word that expresses human feelings, emotions, or attitudes (e.g., happy, sad, angry, disgusted, excited, disappointed, hopeful, nervous, love, hate).  
            Non-emotion words: Neutral words describing objects, actions, or factual states (e.g., table, running, computer, learning, red, speed).  
            is_emotion: 1 = emotion word, 0 = non-emotion word

            For each input word, return a JSON object with the following fields:  
            [{{ "word": "the input word", "is_emotion": 1 or 0 }}]

            # Example
            Input: [happy, sad, apple]
            Output:
            [{{"word": "happy", "is_emotion": 1}}, {{"word": "sad", "is_emotion": 1}}, {{"word": "apple", "is_emotion": 0}}]

            # Now process your input words
            Input: {str(words)}
            Output:
            """
        
    return prompt

def extract_json_from_response(response_list: List[str]) -> List[dict]:
    """
    从模型返回列表中提取 JSON 对象
    """
    all_results = []
    for idx, r in enumerate(response_list):
        # 尝试匹配 ```json ... ```
        match = re.search(r'```json(.*?)```', r, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        else:
            # 如果没有 ```json```，尝试截取开头可能的 JSON
            split_keywords = ["explanation", "Explanation"]
            json_end = min([r.find(k) for k in split_keywords if r.find(k) != -1] + [len(r)])
            json_str = r[:json_end].strip()

        # 去掉首尾多余符号
        json_str = json_str.strip("` \n")

        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                all_results.extend(data)
            elif isinstance(data, dict):
                all_results.append(data)
        except json.JSONDecodeError as e:
            print(f"JSON decode error in response {idx}: {e}")
            continue
    return all_results