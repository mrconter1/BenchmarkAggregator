import os
import asyncio
import pandas as pd
from typing import List
from urllib.parse import urlparse
from benchmarks.base_benchmark import BaseBenchmark
from tqdm import tqdm

class GSM8KBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.id = "GSM8K"
        self.data_url = "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet"

    async def setup(self):
        self.create_temp_dir()
        await self.download_data()

    async def download_data(self):
        parsed_url = urlparse(self.data_url)
        self.data_file = os.path.basename(parsed_url.path)
        local_path = os.path.join(self.temp_dir, self.data_file)
        await self.download_file(self.data_url, local_path)

    async def run(self, model: str, client) -> float:
        df = pd.read_parquet(os.path.join(self.temp_dir, self.data_file)).head(101)
        total_questions = len(df)
        
        print(f"Starting GSM8K benchmark for model: {model}")
        print(f"Total questions: {total_questions}")
        
        progress_bar = tqdm(total=total_questions, desc="Progress", unit="question")
        
        tasks = []
        for _, row in df.iterrows():
            task = self.process_question(model, client, row, progress_bar)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        
        correct_answers = sum(results)
        final_score = correct_answers / total_questions
        
        progress_bar.close()
        print(f"\nFinal Score for {model} on GSM8K: {final_score:.2%}")
        
        return final_score

    async def process_question(self, model: str, client, row, progress_bar):
        question = row['question']
        correct_answer = row['answer'].split('####')[-1].strip()
        
        prompt = self.construct_prompt(question)
        
        model_response = await client.query_model(model, prompt)
        model_answer = self.parse_model_answer(model_response)
        
        is_correct = self.check_answer(model_answer, correct_answer)
        
        progress_bar.update(1)
        return int(is_correct)

    def construct_prompt(self, question: str) -> str:
        prompt = f"Solve the following grade school math problem step by step. At the end, provide the final numeric answer enclosed in [answer][/answer] tags.\n\nQuestion: {question}\n\nSolution:"
        return prompt

    def parse_model_answer(self, response: str) -> str:
        start_tag = "[answer]"
        end_tag = "[/answer]"
        start_index = response.find(start_tag)
        end_index = response.find(end_tag)
        if start_index != -1 and end_index != -1:
            return response[start_index + len(start_tag):end_index].strip()
        else:
            return response  # Return the full response if tags are not found

    def check_answer(self, model_answer: str, correct_answer: str) -> bool:
        try:
            model_value = float(model_answer)
            correct_value = float(correct_answer)
            return abs(model_value - correct_value) < 1e-6  # Allow for small floating-point differences
        except ValueError:
            return False  # If we can't convert to float, consider it incorrect

    async def cleanup(self):
        self.remove_temp_dir()