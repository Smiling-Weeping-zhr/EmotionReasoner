import numpy as np

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
    