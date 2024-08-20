import csv
import random
from typing import List, Tuple
from huggingface_hub import hf_hub_download
from benchmarks.base_benchmark import BaseBenchmark
import pandas as pd

class GPQADiamondBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.id = "GPQA-Diamond"
        self.repo_id = "Idavidrein/gpqa"
        self.filename = "gpqa_diamond.csv"

    async def setup(self):
        await super().setup()
        self.data_file = hf_hub_download(repo_id=self.repo_id, filename=self.filename, repo_type="dataset")

    async def get_dataset(self) -> pd.DataFrame:
        with open(self.data_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return pd.DataFrame(list(reader))

    def get_question(self, row: pd.Series) -> Tuple[str, List[str]]:
        question = row['Question']
        correct_answer = row['Correct Answer']
        incorrect_answers = [
            row['Incorrect Answer 1'],
            row['Incorrect Answer 2'],
            row['Incorrect Answer 3']
        ]
        options = [correct_answer] + incorrect_answers
        shuffled_options = list(enumerate(options))
        random.shuffle(shuffled_options)
        
        formatted_question = f"{question}\n\nOptions:\n"
        for i, option in shuffled_options:
            formatted_question += f"- {option}\n"
        
        return formatted_question, [option for _, option in shuffled_options]

    def get_correct_answer(self, row: pd.Series) -> str:
        return row['Correct Answer']

    def check_answer(self, model_answer: str, correct_answer: str) -> bool:
        try:
            return model_answer.strip().lower() == correct_answer
        except (ValueError, IndexError):
            return False

    def process_question(self, model: str, client, row, progress_bar):
        question, self.shuffled_options = self.get_question(row)
        return super().process_question(model, client, row, progress_bar)