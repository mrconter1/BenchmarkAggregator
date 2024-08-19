import os
import pandas as pd
from typing import List
from benchmarks.base_benchmark import BaseBenchmark

class DROPBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.id = "DROP"
        self.data_url = "https://huggingface.co/datasets/ucinlp/drop/resolve/main/data/validation-00000-of-00001.parquet"

    async def get_dataset(self) -> pd.DataFrame:
        return pd.read_parquet(os.path.join(self.temp_dir, self.data_file)).head(100)

    def get_question(self, row: pd.Series) -> str:
        return f"""Read the following passage and answer the question:

Passage: {row['passage']}

Question: {row['question']}

Answer:"""

    def get_correct_answer(self, row: pd.Series) -> List[str]:
        return row['answers_spans']['spans']

    def check_answer(self, model_answer: str, correct_answers: List[str]) -> bool:
        return any(answer.lower() in model_answer.lower() for answer in correct_answers)