import os
import requests
import pandas as pd
from typing import List
from urllib.parse import urlparse
from benchmarks.base_benchmark import BaseBenchmark

class MMULProBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.id = "MMLU-Pro"
        self.data_url = "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"

    def setup(self):
        self.create_temp_dir()
        self.download_data()

    def download_data(self):
        parsed_url = urlparse(self.data_url)
        self.data_file = os.path.basename(parsed_url.path)
        local_path = os.path.join(self.temp_dir, self.data_file)
        self.download_file(self.data_url, local_path)

    def run(self, model: str, client) -> float:
        df = pd.read_parquet(os.path.join(self.temp_dir, self.data_file))
        correct_answers = 0
        total_questions = len(df)

        for _, row in df.iterrows():
            question = row['question']
            options = row['options']
            correct_answer = row['answer']

            prompt = self.construct_prompt(question, options)
            model_answer = self.query_model(client, model, prompt)

            if model_answer.strip().upper() == correct_answer:
                correct_answers += 1

        return correct_answers / total_questions

    def construct_prompt(self, question: str, options: List[str]) -> str:
        prompt = f"{question}\n\nOptions:\n"
        for i, option in enumerate(options):
            prompt += f"{chr(65 + i)}. {option}\n"
        prompt += "\nPlease provide the letter of the correct answer."
        return prompt

    def query_model(self, client, model: str, prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def cleanup(self):
        self.remove_temp_dir()

    @staticmethod
    def download_file(url: str, local_path: str):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(local_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"File downloaded successfully and saved to {local_path}")
        except Exception as e:
            print(f"An error occurred while downloading the file: {e}")
            raise