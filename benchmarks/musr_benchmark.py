from benchmarks.base_benchmark import BaseBenchmark
import pandas as pd
import os
import ast

class MuSRBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.id = "MuSR"
        self.data_url = "https://huggingface.co/datasets/TAUR-Lab/MuSR/resolve/main/all.csv"

    async def get_dataset(self) -> pd.DataFrame:
        file_path = os.path.join(self.temp_dir, self.data_file)
        df = pd.read_csv(file_path)
        df['choices'] = df['choices'].apply(ast.literal_eval)
        return df

    def get_question(self, row: pd.Series) -> str:
        narrative = row['narrative']
        question = row['question']
        choices = row['choices']
        
        formatted_question = f"{narrative}\n\n{question}\n\nOptions:\n"
        for i, choice in enumerate(choices):
            formatted_question += f"{i+1}. {choice}\n"
        formatted_question += f"\nFinal answer should be the number of the correct option."
        return formatted_question

    def get_correct_answer(self, row: pd.Series) -> str:
        return str(row['answer_index'] + 1)  # Adding 1 because answer_index is 0-based

    def check_answer(self, model_answer: str, correct_answer: str) -> bool:
        return model_answer.strip() == correct_answer

    def construct_prompt(self, question: str) -> str:
        prompt = f"{question}\n\n"
        return self.append_answer_instruction(prompt)

    @staticmethod
    def append_answer_instruction(prompt: str) -> str:
        return prompt + "Please reason through the narrative and question carefully. After your reasoning, provide your answer as a single number corresponding to the chosen option. For example: [answer]1[/answer]"