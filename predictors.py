from abc import ABC, abstractmethod
from typing import List, Dict, Callable
from liquid import Template

import utils
import tasks
import re

class GPT4Predictor(ABC):
    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def inference(self, ex, prompt):
        pass

class BinaryPredictor(GPT4Predictor):
    categories = ['No', 'Yes']

    def inference(self, ex, prompt):
        prompt = Template(prompt).render(text=ex['text'])
        response = utils.chatgpt(
            prompt, max_tokens=4, n=1, timeout=2, 
            temperature=self.opt['temperature'])[0]
        pred = 1 if response.strip().upper().startswith('YES') else 0
        return pred
    
class GSM8KPredictor(GPT4Predictor):
    def inference(self, ex, prompt):
        prompt_filled = Template(prompt).render(text=ex['text'])
        response = utils.chatgpt(
            prompt_filled, max_tokens=128, n=1,
            temperature=self.opt['temperature']
        )[0]
        return self.extract_answer(response)

    def extract_answer(self, response):
        match = re.search(r"(?i)answer\s*[:\-]?\s*(-?\d+\.?\d*)", response)
        if match:
            return match.group(1)
        nums = re.findall(r"-?\d+\.?\d*", response)
        return nums[-1] if nums else response.strip()
