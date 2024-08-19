import os
import asyncio
import pandas as pd
from typing import List, Dict
from urllib.parse import urlparse
from benchmarks.base_benchmark import BaseBenchmark
from tqdm import tqdm

class DROPBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.id = "DROP"
        self.data_url = "https://huggingface.co/datasets/ucinlp/drop/resolve/main/data/validation-00000-of-00001.parquet"

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
        
        print(f"Starting DROP benchmark for model: {model}")
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
        print(f"\nFinal Score for {model} on DROP: {final_score:.2%}")
        
        return final_score

    async def process_question(self, model: str, client, row, progress_bar):
        passage = row['passage']
        question = row['question']
        correct_answers = row['answers_spans']['spans']
        
        prompt = self.construct_prompt(passage, question)
        
        model_response = await client.query_model(model, prompt)
        model_answer = self.parse_model_answer(model_response)
        
        is_correct = self.check_answer(model_answer, correct_answers)
        
        progress_bar.update(1)
        return int(is_correct)

    def construct_prompt(self, passage: str, question: str) -> str:
        prompt = f"""Read the following passage and answer the question. Provide your answer within [answer][/answer] tags.

Passage: {passage}

Question: {question}

Answer:"""
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

    def check_answer(self, model_answer: str, correct_answers: List[str]) -> bool:
        return any(answer.lower() in model_answer.lower() for answer in correct_answers)

    async def cleanup(self):
        self.remove_temp_dir()