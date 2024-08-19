import asyncio
import csv
import random
from typing import List
from huggingface_hub import hf_hub_download
from benchmarks.base_benchmark import BaseBenchmark
from tqdm import tqdm

class GPQADiamondBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.id = "GPQA-Diamond"
        self.repo_id = "Idavidrein/gpqa"
        self.filename = "gpqa_diamond.csv"

    async def setup(self):
        self.create_temp_dir()
        await self.download_data()

    async def download_data(self):
        local_path = hf_hub_download(repo_id=self.repo_id, filename=self.filename, repo_type="dataset")
        self.data_file = local_path

    async def run(self, model: str, client) -> float:
        with open(self.data_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            questions = list(reader)

        total_questions = len(questions)
        
        print(f"Starting GPQA Diamond benchmark for model: {model}")
        print(f"Total questions: {total_questions}")
        
        progress_bar = tqdm(total=total_questions, desc="Progress", unit="question")
        
        tasks = []
        for row in questions:
            task = self.process_question(model, client, row, progress_bar)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        
        correct_answers = sum(results)
        final_score = correct_answers / total_questions
        
        progress_bar.close()
        print(f"\nFinal Score for {model} on GPQA Diamond: {final_score:.2%}")
        
        return final_score

    async def process_question(self, model: str, client, row, progress_bar):
        question = row['Question']
        correct_answer = row['Correct Answer']
        incorrect_answers = [
            row['Incorrect Answer 1'],
            row['Incorrect Answer 2'],
            row['Incorrect Answer 3']
        ]
        
        all_answers = [correct_answer] + incorrect_answers
        random.shuffle(all_answers)
        
        prompt = self.construct_prompt(question, all_answers)
        
        model_response = await client.query_model(model, prompt)
        model_answer = self.parse_model_answer(model_response)
        
        is_correct = self.check_answer(model_answer, correct_answer, all_answers)
        
        progress_bar.update(1)
        return int(is_correct)

    def construct_prompt(self, question: str, answers: List[str]) -> str:
        prompt = f"Question: {question}\n\nPlease choose the correct answer from the following options:\n\n"
        for i, answer in enumerate(answers, start=1):
            prompt += f"{i}. {answer}\n"
        prompt += "\nProvide your answer as the number corresponding to your chosen option."
        return prompt

    def parse_model_answer(self, response: str) -> str:
        # Extract the numeric choice from the model's response
        for word in response.split():
            if word.isdigit() and 1 <= int(word) <= 4:
                return int(word)
        return None

    def check_answer(self, model_answer: int, correct_answer: str, all_answers: List[str]) -> bool:
        if model_answer is None:
            return False
        return all_answers[model_answer - 1] == correct_answer

    async def cleanup(self):
        # No need to remove the file as it's managed by huggingface_hub
        pass