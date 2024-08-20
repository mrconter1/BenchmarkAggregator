from abc import ABC, abstractmethod
import os
import tempfile
import shutil
import aiofiles
import aiohttp
import pandas as pd
from tqdm import tqdm
import asyncio
from typing import List, Dict, Any
from urllib.parse import urlparse

class BaseBenchmark(ABC):
    def __init__(self):
        self.id = None
        self.temp_dir = None
        self.data_url = None
        self.data_file = None

    async def setup(self):
        self.create_temp_dir()
        await self.download_data()

    @abstractmethod
    async def get_dataset(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_question(self, row: pd.Series) -> str:
        pass

    @abstractmethod
    def get_correct_answer(self, row: pd.Series) -> Any:
        pass

    @abstractmethod
    def check_answer(self, model_answer: str, correct_answer: Any) -> bool:
        pass

    async def run(self, model: str, client, df: pd.DataFrame) -> float:
        total_questions = len(df)
        
        print(f"Starting {self.id} benchmark for model: {model}")
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
        print(f"\nFinal Score for {model} on {self.id}: {final_score:.2%}")
        
        return final_score

    async def process_question(self, model: str, client, row, progress_bar):
        question = self.get_question(row)
        prompt = self.construct_prompt(question)
        
        model_response = await client.query_model(model, prompt)
        model_answer = self.parse_model_answer(model_response)
        
        correct_answer = self.get_correct_answer(row)
        is_correct = self.check_answer(model_answer, correct_answer)
        
        progress_bar.update(1)
        return int(is_correct)

    def construct_prompt(self, question: str) -> str:
        prompt = f"{question}\n\n"
        return self.append_answer_instruction(prompt)

    def parse_model_answer(self, response: str) -> str:
        start_tag = "[answer]"
        end_tag = "[/answer]"
        start_index = response.find(start_tag)
        end_index = response.find(end_tag)
        if start_index != -1 and end_index != -1:
            return response[start_index + len(start_tag):end_index].strip()
        else:
            return response  # Return the full response if tags are not found

    @staticmethod
    def append_answer_instruction(prompt: str) -> str:
        return prompt + "Please reason through the question and options. After your reasoning, provide your answer enclosed in [answer] tags. For example: [answer]Your answer here[/answer]"

    async def cleanup(self):
        self.remove_temp_dir()

    def create_temp_dir(self):
        self.temp_dir = tempfile.mkdtemp()

    def remove_temp_dir(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    async def download_data(self):
        if not self.data_url:
            return
        parsed_url = urlparse(self.data_url)
        self.data_file = os.path.basename(parsed_url.path)
        local_path = os.path.join(self.temp_dir, self.data_file)
        await self.download_file(self.data_url, local_path)

    @staticmethod
    async def download_file(url: str, local_path: str):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    async with aiofiles.open(local_path, 'wb') as file:
                        while True:
                            chunk = await response.content.read(8192)
                            if not chunk:
                                break
                            await file.write(chunk)
            print(f"File downloaded successfully and saved to {local_path}")
        except Exception as e:
            print(f"An error occurred while downloading the file: {e}")
            raise