import os
import requests
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
        
        print(f"Starting MMLU-Pro benchmark for model: {model}")
        print(f"Total questions: {total_questions}")
        
        progress_bar = tqdm(total=total_questions, desc="Progress", unit="question")
        
        for index, row in df.iterrows():
            question = row['question']
            options = row['options']
            correct_answer = row['answer']
            prompt = self.construct_prompt(question, options)
            model_response = self.query_model(client, model, prompt)
            model_answer = self.parse_model_answer(model_response)
            
            if model_answer.strip().upper() == correct_answer:
                correct_answers += 1
            
            current_score = correct_answers / (index + 1)
            questions_left = total_questions - (index + 1)
            
            progress_bar.set_postfix({
                "Score": f"{current_score:.2%}",
                "Correct": correct_answers,
                "Remaining": questions_left
            })
            progress_bar.update(1)
            
            if (index + 1) % 10 == 0 or index == total_questions - 1:
                print(f"\nStatus Update:")
                print(f"Model: {model}")
                print(f"Questions Processed: {index + 1}/{total_questions}")
                print(f"Current Score: {current_score:.2%}")
                print(f"Questions Remaining: {questions_left}")
                print("------------------------")
        
        progress_bar.close()
        final_score = correct_answers / total_questions
        print(f"\nFinal Score for {model}: {final_score:.2%}")
        
        return final_score

    def construct_prompt(self, question: str, options: List[str]) -> str:
        prompt = f"{question}\n\nOptions:\n"
        for i, option in enumerate(options):
            prompt += f"{chr(65 + i)}. {option}\n"
        prompt += "\nInstructions: Please reason through the question and options. After your reasoning, provide the letter of the correct answer enclosed in [answer] tags. For example: [answer]A[/answer]"
        return prompt

    def query_model(self, client, model: str, prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def parse_model_answer(self, response: str) -> str:
        start_tag = "[answer]"
        end_tag = "[/answer]"
        start_index = response.find(start_tag)
        end_index = response.find(end_tag)
        if start_index != -1 and end_index != -1:
            return response[start_index + len(start_tag):end_index].strip()
        else:
            return response  # Return the full response if tags are not found

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