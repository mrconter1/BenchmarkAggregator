from benchmarks.base_benchmark import BaseBenchmark
import pandas as pd
import os

class ARCChallengeBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.id = "ARC-Challenge"
        self.data_url = "https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Challenge/validation-00000-of-00001.parquet"

    async def setup(self):
        await super().setup()
        file_path = os.path.join(self.temp_dir, "arc_challenge_validation.parquet")
        await self.download_file(self.data_url, file_path)
        self.data = pd.read_parquet(file_path)

    async def get_dataset(self) -> pd.DataFrame:
        return self.data

    def get_question(self, row: pd.Series) -> str:
        question = row['question']
        choices = row['choices']
        
        formatted_question = f"{question}\n\nOptions:\n"
        for label, text in zip(choices['label'], choices['text']):
            formatted_question += f"{label}. {text}\n"
        formatted_question += f"\nFinal answer should be the letter of the correct option."
        return formatted_question

    def get_correct_answer(self, row: pd.Series) -> str:
        return row['answerKey']

    def check_answer(self, model_answer: str, correct_answer: str) -> bool:
        return model_answer.strip().upper() == correct_answer.strip().upper()

    def construct_prompt(self, question: str) -> str:
        prompt = f"{question}\n\n"
        return self.append_answer_instruction(prompt)

    @staticmethod
    def append_answer_instruction(prompt: str) -> str:
        return prompt + "Please reason through the question and options carefully. After your reasoning, provide your answer as a single letter corresponding to the chosen option. For example: [answer]A[/answer]"