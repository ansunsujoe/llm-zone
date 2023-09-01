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


class EvaluationType(str, Enum):
    BOOLEAN = 1
    RATING_TO_TEN = 2


class LLMEvaluator:
    def __init__(self, llm_type: LLMType, output_type: EvaluationType) -> None:
        self.model = LLM(
            llm_type=llm_type, behavior="You are an unbiased evaluator."
        )
        self.output_type = output_type

    def evaluate(self, prompt: str, response: str) -> str:
        evaluation_prompt = (
            "Given the following prompt and response, "
            f"{self._get_output_format_string()}",
            f"\nPrompt: {prompt}\nResponse: {response}"
        )
        return self.model.generate(evaluation_prompt)

    def _get_output_format_string(self) -> str:
        if self.output_type == EvaluationType.BOOLEAN:
            return (
                "output True if the response is correct, "
                "and False if the response is not correct."
            )
        elif self.output_type == EvaluationType.RATING_TO_TEN:
            return (
                "rate the correctness of the response on a scale from 1 to 10."
            )
