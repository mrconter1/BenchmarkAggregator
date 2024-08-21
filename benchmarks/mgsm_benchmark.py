import os
import pandas as pd
import git
import tempfile
import re
from benchmarks.base_benchmark import BaseBenchmark

class MGSMBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.id = "MGSM"
        self.repo_url = "https://huggingface.co/datasets/juletxara/mgsm"
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = os.path.join(self.temp_dir, "mgsm")
        self.repo = None

    async def setup(self):
        await super().setup()
        if not os.path.exists(self.repo_path):
            self.repo = git.Repo.clone_from(self.repo_url, self.repo_path)
        else:
            self.repo = git.Repo(self.repo_path)
            origin = self.repo.remotes.origin
            origin.pull()

    async def get_dataset(self) -> pd.DataFrame:
        all_questions = []
        for file in os.listdir(self.repo_path):
            if file.endswith('.tsv'):
                file_path = os.path.join(self.repo_path, file)
                df = pd.read_csv(file_path, sep='\t', names=['question', 'answer_number'], quoting=3)
                all_questions.append(df)
        
        return pd.concat(all_questions, ignore_index=True)

    def get_question(self, row: pd.Series) -> str:
        return row['question']

    def get_correct_answer(self, row: pd.Series) -> str:
        return str(row['answer_number'])

    def extract_number(self, text: str) -> float:
        # Keep only digits and decimal points
        cleaned_text = ''.join(char for char in text if char in '0123456789.')
        if cleaned_text:
            return float(cleaned_text)
        else:
            raise ValueError("No numeric value found in the answer")

    def check_answer(self, model_answer: str, correct_answer: str) -> bool:
        try:
            model_value = self.extract_number(model_answer)
            correct_value = float(correct_answer)
            return abs(model_value - correct_value) < 1e-6  # Allow for small floating-point differences
        except ValueError:
            return False  # If we can't extract or convert to float, consider it incorrect