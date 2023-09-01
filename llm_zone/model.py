from enum import Enum

import torch
from transformers import pipeline


class LLMType(str, Enum):
    DOLLY = "databricks/dolly-v2-3b"


class LLM:
    def __init__(self, llm_type: LLMType, behavior: str | None = None) -> None:
        self.llm_type = llm_type
        self.behavior = behavior
        self.generator = pipeline(
            model=llm_type.value,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

    def set_behavior(self, behavior: str) -> None:
        self.behavior = behavior

    def generate(self, prompt: str) -> str:
        # Construct full prompt.
        full_prompt = self._construct_full_prompt(prompt)

        # Query the LLM.
        response = self.generator(full_prompt)
        return response[0]["generated_text"]

    def _construct_full_prompt(self, prompt: str) -> str:
        full_prompt = ""
        if self.behavior is not None:
            full_prompt += self.behavior + "\n"
        full_prompt += prompt
        return full_prompt