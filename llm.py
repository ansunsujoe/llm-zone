from enum import Enum
from typing import Optional

import torch
from transformers import pipeline


class LLMType(str, Enum):
    DOLLY = "databricks/dolly-v2-3b"


class Model:
    def __init__(self, llm_type: LLMType):
        self.llm_type = llm_type
        self.generator = pipeline(
            model=llm_type.value,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

    def generate(self, prompt: str) -> str:
        response = self.generator(prompt)
        return response[0]["generated_text"]


class Evaluator:
    def __init__(self, model: Model):
        self.model = model

    def evaluate_bool(self, prompt: str, response: str):
        pass
