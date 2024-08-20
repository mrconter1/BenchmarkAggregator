from benchmarks.base_benchmark import BaseBenchmark
import pandas as pd
import os
import numpy as np 

class HellaSwagBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.id = "HellaSwag"
        self.data_url = "https://huggingface.co/api/datasets/Rowan/hellaswag/parquet/default/validation/0.parquet"

    async def setup(self):
        await super().setup()
        file_path = os.path.join(self.temp_dir, "hellaswag_validation.parquet")
        await self.download_file(self.data_url, file_path)
        self.data = pd.read_parquet(file_path)

    async def get_dataset(self) -> pd.DataFrame:
        return self.data

    def get_question(self, row: pd.Series) -> str:
        ctx = row['ctx']
        endings = row['endings']

        if isinstance(endings, np.ndarray):
            endings = endings.tolist()
        
        formatted_question = f"{ctx}\n\nOptions:\n"
        for i, ending in enumerate(endings):
            formatted_question += f"{i}. {ending}\n"
        formatted_question += f"\nFinal answer should be the number of the correct option."
        return formatted_question

    def get_correct_answer(self, row: pd.Series) -> str:
        return row['label']

    def check_answer(self, model_answer: str, correct_answer: str) -> bool:
        return model_answer.strip() == correct_answer.strip()

    def construct_prompt(self, question: str) -> str:
        prompt = f"{question}\n\n"
        return self.append_answer_instruction(prompt)

    @staticmethod
    def append_answer_instruction(prompt: str) -> str:
        return prompt + "Please reason through the context and options carefully. After your reasoning, provide your answer as a single number corresponding to the chosen option. For example: [answer]0[/answer]"