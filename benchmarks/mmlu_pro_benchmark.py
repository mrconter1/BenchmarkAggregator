import os
import asyncio
import pandas as pd
from typing import List
from urllib.parse import urlparse
from benchmarks.base_benchmark import BaseBenchmark
from tqdm import tqdm

class MMULProBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.id = "MMLU-Pro"
        self.data_url = "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"

    async def setup(self):
        self.create_temp_dir()
        await self.download_data()

    async def download_data(self):
        parsed_url = urlparse(self.data_url)
        self.data_file = os.path.basename(parsed_url.path)
        local_path = os.path.join(self.temp_dir, self.data_file)
        await self.download_file(self.data_url, local_path)

    async def run(self, model: str, client) -> float:
        df = pd.read_parquet(os.path.join(self.temp_dir, self.data_file)).head(100)
        total_questions = len(df)
        
        print(f"Starting MMLU-Pro benchmark for model: {model}")
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
        print(f"\nFinal Score for {model}: {final_score:.2%}")
        
        return final_score

    async def process_question(self, model: str, client, row, progress_bar):
        question = row['question']
        options = row['options']
        correct_answer = row['answer']
        prompt = self.construct_prompt(question, options)
        
        model_response = await client.query_model(model, prompt)
        model_answer = self.parse_model_answer(model_response)
        
        is_correct = model_answer.strip().upper() == correct_answer
        
        progress_bar.update(1)
        return int(is_correct)

    def construct_prompt(self, question: str, options: List[str]) -> str:
        prompt = f"{question}\n\nOptions:\n"
        for i, option in enumerate(options):
            prompt += f"{chr(65 + i)}. {option}\n"
        prompt += "\nInstructions: Please reason through the question and options. After your reasoning, provide the letter of the correct answer enclosed in [answer] tags. For example: [answer]A[/answer]"
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

    async def cleanup(self):
        self.remove_temp_dir()