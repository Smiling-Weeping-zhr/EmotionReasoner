import re
import ipdb

# === 预编译正则，提高循环内多次调用效率 ===
ANS_TAG  = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)
THINK_TAG = re.compile(r"<think>\s*(.*?)\s*</think>", re.S)

def extract_answer(text: str) -> str:
    """和你原来 extract_answer 完全等价"""
    m = ANS_TAG.search(text)
    return m.group(1).strip() if m else ""

def extract_think(text: str) -> str:
    """提取 <think>...</think> 标签中的内容"""
    m = THINK_TAG.search(text)
    return m.group(1).strip() if m else ""

def parse_label_string(label_str):
    # 去掉 []
    items = label_str.strip("[]").split(",")
    # 去掉首尾空格和引号
    return [x.strip().strip("'").strip('"').lower() for x in items if x.strip()]

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
    
def offline_assessment(pred, ref_model=None, processor=None, generation_config=None):
    """
    Offline LLM-based assessment of tri-modal explanation quality for MER.
    """
    prompt = f"""You are an evaluator for multimodal emotion recognition (MER) model explanations. The model always receives three modalities as input:
            - Text (verbal content, linguistic cues, sentiment words) 
            - Video (facial expressions, gestures, visual affective cues) 
            - Audio (tone, pitch, prosody, acoustic emotion cues)
        Your task:
        1. Evaluate how well the generated explanation demonstrates:
        Understanding: clear and accurate interpretation of emotional cues across the three modalities;
        Balance: coherent integration of text, video, and audio evidence without over-relying on a single modality;
        Relevance: use of cues that directly and meaningfully support the predicted emotion;
        Insight: ability to capture subtle, non-obvious emotional or contextual nuances beyond surface descriptions.
        2. Provide ONE overall attention score (1-10), using the rubric:
        - 1-2 = Fails to interpret multimodal cues meaningfully; lacks relevance and insight.
        - 3-4 = Limited understanding; weak integration of modalities and minimal emotional depth.
        - 5-6 = Basic multimodal interpretation with moderate relevance but limited nuance or balance.
        - 7-8 = Clear and coherent multimodal reasoning with good relevance and some subtle insight.
        - 9 = Strong, well-balanced integration of modalities with meaningful and nuanced emotional analysis.
        - 10 = Fully integrated, highly insightful, and precisely reasoned multimodal explanation with excellent clarity and balance.
        3. DO NOT output reasoning or explanations, only the score.

        The output format is strictly:
        'Predicted Score' : <score>;

        Now evaluate the following output: {pred}"""
        
    input_format = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(input_format, add_generation_prompt=True, tokenize=False)
    prompt_inputs = processor(text=text, audio=None, images=None, videos=None, return_tensors="pt",
            padding=True, use_audio_in_video=False).to(ref_model.device)
    outputs = ref_model.generate(**prompt_inputs, generation_config=generation_config,
            use_audio_in_video=False, return_audio=False)
    response = processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    match = re.search(r'assistant\s*(.*)$', response, re.IGNORECASE)
    if match:
        response = match.group(1).strip()
    
    return response  

def coarse_assessment(reason, ref_model=None, processor=None, generation_config=None):
    """
    coarse assessment
    """
    prompt = f"""Please act as an expert in the field of emotions. \
            We provide a few words to describe the emotions of a character. \
            Please choose the most likely sentiment from the given candidates: [positive, negative, neutral] \
            Please direct output answer without analyzing process. \
            Input: [joyful]; Output: positive \
            Input: []; Output: neutral \
            Input: [sad]; Output: negative \
            Input: {reason}; Output: """
        
    input_format = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(input_format, add_generation_prompt=True, tokenize=False)
    prompt_inputs = processor(text=text, audio=None, images=None, videos=None, return_tensors="pt",
            padding=True, use_audio_in_video=False).to(ref_model.device)
    outputs = ref_model.generate(**prompt_inputs, generation_config=generation_config,
            use_audio_in_video=False, return_audio=False)
    response = processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    match = re.search(r'assistant\s*(.*)$', response, re.IGNORECASE)
    if match:
        response = match.group(1).strip()
    # ipdb.set_trace()
    return response    