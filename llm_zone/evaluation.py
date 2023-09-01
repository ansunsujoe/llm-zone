from enum import Enum

from llm_zone.model import LLM


class EvaluationType(str, Enum):
    BOOLEAN = 1
    RATING_TO_TEN = 2


class LLMEvaluator:
    def __init__(self, llm: LLM, output_type: EvaluationType) -> None:
        self.llm = llm
        self.llm.set_behavior("You are an unbiased evaluator.")
        self.output_type = output_type

    def evaluate(self, prompt: str, response: str) -> str:
        evaluation_prompt = (
            "Given the following prompt and response, "
            f"{self._get_output_format_string()}"
            f"\nPrompt: {prompt}\nResponse: {response}"
        )
        return self.llm.generate(evaluation_prompt)

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
