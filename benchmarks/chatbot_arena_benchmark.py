import pandas as pd
import os
import git
import pickle
import tempfile
import shutil
import asyncio
from datetime import datetime
from benchmarks.base_benchmark import BaseBenchmark

class ChatbotArenaBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.id = "ChatbotArena"
        self.repo_url = "https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard"
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = os.path.join(self.temp_dir, "chatbot-arena-leaderboard")
        self.model_mapping = {
            "openai/gpt-3.5-turbo-0125": "gpt-3.5-turbo-0125",
            "openai/gpt-4o-mini-2024-07-18": "gpt-4o-mini-2024-07-18",
            "openai/gpt-4o-2024-08-06": "chatgpt-4o-latest-2024-08-08",
            "anthropic/claude-3-sonnet": "claude-3-sonnet-20240229",
            "anthropic/claude-3-opus": "claude-3-opus-20240229",
            "anthropic/claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
            "google/gemini-pro-1.5-exp": "gemini-1.5-pro-exp-0801",
            "meta-llama/llama-3.1-70b-instruct": "llama-3.1-70b-instruct",
            "meta-llama/llama-3.1-405b-instruct": "llama-3.1-405b-instruct",
            "mistralai/mistral-large": "mistral-large-2407"
        }
        self.repo = None

    async def setup(self):
        if not os.path.exists(self.repo_path):
            self.repo = git.Repo.clone_from(self.repo_url, self.repo_path)
        else:
            self.repo = git.Repo(self.repo_path)
            origin = self.repo.remotes.origin
            origin.pull()

    def get_latest_elo_file(self):
        elo_files = [f for f in os.listdir(self.repo_path) if f.startswith("elo_results_") and f.endswith(".pkl")]
        return max(elo_files, key=lambda x: datetime.strptime(x.split("_")[2].split(".")[0], "%Y%m%d"))

    async def get_dataset(self):
        latest_file = self.get_latest_elo_file()
        with open(os.path.join(self.repo_path, latest_file), 'rb') as f:
            data = pickle.load(f)
        
        full_elo_data = data['text']['full']['elo_rating_final']
        df = pd.DataFrame(list(full_elo_data.items()), columns=['Model', 'ELO'])
        return df

    def normalize_scores(self, df):
        min_score = df['ELO'].min()
        max_score = df['ELO'].max()
        df['normalized_score'] = (df['ELO'] - min_score) / (max_score - min_score)
        return df

    async def run(self, model: str, client, df: pd.DataFrame = None, samples: int = None) -> float:
        df = self.normalize_scores(df)
            
        df_sorted = df.sort_values('ELO', ascending=False)
        df_sorted['normalized_score'] = df_sorted['normalized_score'].round(4)

        # For debugging
        #print(tabulate(df_sorted, headers='keys', tablefmt='grid'))

        arena_model = self.model_mapping.get(model, model)
        if arena_model in df['Model'].values:
            normalized_score = df[df['Model'] == arena_model]['normalized_score'].values[0]
            rounded_normalized_score = round(normalized_score, 4)
            return rounded_normalized_score
        else:
            print(f"Warning: No data available for model {model}")
            return 0.0

    async def cleanup(self):
        if self.repo:
            self.repo.close()
        await asyncio.sleep(1)  # Give a moment for any pending operations to complete
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # These methods are not used but are required by the BaseBenchmark abstract class
    def get_question(self, row: pd.Series) -> str:
        pass

    def get_correct_answer(self, row: pd.Series) -> str:
        pass

    def check_answer(self, model_answer: str, correct_answer: str) -> bool:
        pass