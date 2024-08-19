from benchmarks.base_benchmark import BaseBenchmark
import pandas as pd
import os

class MMULProBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.id = "MMLU-Pro"
        self.data_url = "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"

    async def get_dataset(self) -> pd.DataFrame:
        return pd.read_parquet(os.path.join(self.temp_dir, self.data_file)).head(100)

    def get_question(self, row: pd.Series) -> str:
        question = row['question']
        options = row['options']
        formatted_question = f"{question}\n\nOptions:\n"
        for i, option in enumerate(options):
            formatted_question += f"{chr(65 + i)}. {option}\n"
        formatted_question += f"Final answer should be the single letter you choose."
        return formatted_question

    def get_correct_answer(self, row: pd.Series) -> str:
        return row['answer']

    def check_answer(self, model_answer: str, correct_answer: str) -> bool:
        return model_answer.strip().upper() == correct_answer