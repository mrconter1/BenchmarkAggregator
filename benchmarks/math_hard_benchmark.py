import os
import json
import asyncio
import aiofiles
import pandas as pd
from benchmarks.base_benchmark import BaseBenchmark

class MathHardBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.id = "MATH-Hard"
        self.base_url = "https://huggingface.co/datasets/lighteval/MATH-Hard/resolve/main/test/"
        self.subtests = [
            "algebra.jsonl",
            "counting_and_probability.jsonl",
            "geometry.jsonl",
            "intermediate_algebra.jsonl",
            "number_theory.jsonl",
            "prealgebra.jsonl",
            "precalculus.jsonl"
        ]
        self.BOXED_COMMAND = '\\boxed{'

    async def setup(self):
        await super().setup()
        await self.download_subtests()

    async def download_subtests(self):
        tasks = [self.download_file(self.base_url + subtest, os.path.join(self.temp_dir, subtest)) 
                 for subtest in self.subtests]
        await asyncio.gather(*tasks)

    async def get_dataset(self) -> pd.DataFrame:
        all_problems = []
        for subtest in self.subtests:
            file_path = os.path.join(self.temp_dir, subtest)
            async with aiofiles.open(file_path, mode='r') as file:
                content = await file.read()
                problems = json.loads(content)
                all_problems.extend(problems)
        return pd.DataFrame(all_problems)

    def get_question(self, row: pd.Series) -> str:
        return row['problem']

    def get_correct_answer(self, row: pd.Series) -> str:
        solution = row['solution']
        try:
            boxed_content = solution.split("\\boxed{")[1]
            if "}$" in boxed_content:
                boxed_content = boxed_content.split("}$")[0]
            elif "}." in boxed_content:
                boxed_content = boxed_content.split("}.")[0]
            return boxed_content
        except IndexError:
            return None

    def check_answer(self, model_answer: str, correct_answer: str) -> bool:
        return model_answer.strip().lower() == correct_answer.strip().lower()

    def construct_prompt(self, question: str) -> str:
        prompt = f"Solve the following mathematics problem:\n\n{question}\n\n"
        prompt += "Don't enclose the answer in parentheses and don't add units to the answer.\n"
        prompt += "But still ONLY WRITE the answer in LATEX script!\n"
        prompt += "Example: [answer]42[/answer]\n"
        prompt += "Don't add dollar signs before or after answer!\n"
        return self.append_answer_instruction(prompt)