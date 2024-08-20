import os
import pandas as pd
from benchmarks.base_benchmark import BaseBenchmark

class GSM8KBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.id = "GSM8K"
        self.data_url = "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet"

    async def get_dataset(self) -> pd.DataFrame:
        return pd.read_parquet(os.path.join(self.temp_dir, self.data_file))

    def get_question(self, row: pd.Series) -> str:
        return f"Solve the following grade school math problem step by step:\n\n{row['question']}\n\nSolution:"

    def get_correct_answer(self, row: pd.Series) -> str:
        return row['answer'].split('####')[-1].strip()

    def check_answer(self, model_answer: str, correct_answer: str) -> bool:
        try:
            model_value = float(model_answer)
            correct_value = float(correct_answer)
            return abs(model_value - correct_value) < 1e-6  # Allow for small floating-point differences
        except ValueError:
            return False  # If we can't convert to float, consider it incorrect