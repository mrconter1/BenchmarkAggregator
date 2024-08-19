import os
import subprocess
import pandas as pd
import tempfile
import shutil
from api_handler import query_model

class MMULProBenchmark:
    def __init__(self):
        self.repo_url = "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro"
        self.temp_dir = None
        self.data_file = "data/test-00000-of-00001.parquet"

    def setup(self):
        self.temp_dir = tempfile.mkdtemp()
        self.clone_repo()

    def cleanup(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def clone_repo(self):
        subprocess.run(["git", "clone", self.repo_url, self.temp_dir])

    def run_benchmark(self, model, client):
        try:
            self.setup()
            
            # Load the dataset
            df = pd.read_parquet(os.path.join(self.temp_dir, self.data_file))
            
            correct_answers = 0
            total_questions = len(df)

            for _, row in df.iterrows():
                question = row['question']
                options = row['options']
                correct_answer = row['answer']

                # Construct the prompt
                prompt = f"{question}\n\nOptions:\n"
                for i, option in enumerate(options):
                    prompt += f"{chr(65 + i)}. {option}\n"
                prompt += "\nPlease provide the letter of the correct answer."

                # Query the model
                model_answer = query_model(client, model, prompt)

                # Check if the model's answer is correct
                if model_answer.strip().upper() == correct_answer:
                    correct_answers += 1

            score = correct_answers / total_questions
            return score
        finally:
            self.cleanup()