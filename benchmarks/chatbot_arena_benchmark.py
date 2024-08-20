import pandas as pd
import os
import git
import pickle
import tempfile
import shutil
from benchmarks.base_benchmark import BaseBenchmark
from tabulate import tabulate

class ChatbotArenaBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.id = "ChatbotArena"
        self.repo_url = "https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard"
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = os.path.join(self.temp_dir, "chatbot-arena-leaderboard")
        self.model_mapping = {
            "anthropic/claude-3.5-sonnet": "claude-3.5-sonnet-20240620",
            "openai/gpt-4o-mini-2024-07-18": "gpt-4o-mini-2024-07-18",
            # Add more mappings as needed
        }

    async def setup(self):
        if not os.path.exists(self.repo_path):
            git.Repo.clone_from(self.repo_url, self.repo_path)
        else:
            repo = git.Repo(self.repo_path)
            origin = repo.remotes.origin
            origin.pull()

    def get_latest_elo_file(self):
        elo_files = [f for f in os.listdir(self.repo_path) if f.startswith("elo_results_") and f.endswith(".pkl")]
        return max(elo_files, key=lambda x: os.path.getmtime(os.path.join(self.repo_path, x)))

    async def get_dataset(self):
        latest_file = self.get_latest_elo_file()
        with open(os.path.join(self.repo_path, latest_file), 'rb') as f:
            data = pickle.load(f)
        
        full_elo_data = data['elo_rating_online']
        df = pd.DataFrame(list(full_elo_data.items()), columns=['Model', 'ELO'])
        return df

    def normalize_scores(self, df):
        min_score = df['ELO'].min()
        max_score = df['ELO'].max()
        df['normalized_score'] = (df['ELO'] - min_score) / (max_score - min_score)
        return df

    async def run(self, models, client, samples=None):
        await self.setup()
        df = await self.get_dataset()
        df = self.normalize_scores(df)

        # Sort the DataFrame by ELO score in descending order
        df_sorted = df.sort_values('ELO', ascending=False)
        
        # Print the table
        print("\nChatbot Arena ELO Leaderboard:")
        print(tabulate(df_sorted, headers='keys', tablefmt='pretty', showindex=False))
        print("\n")  # Add a newline for better readability
        
        results = {}
        for model in models:
            arena_model = self.model_mapping.get(model, model)
            if arena_model in df['Model'].values:
                elo_score = df[df['Model'] == arena_model]['ELO'].values[0]
                normalized_score = df[df['Model'] == arena_model]['normalized_score'].values[0]
                results[model] = {
                    'elo': elo_score,
                    'normalized_score': normalized_score
                }
            else:
                print(f"Warning: No data available for model {model}")
        
        return results

    async def cleanup(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    # These methods are not used but are required by the BaseBenchmark abstract class
    def get_question(self, row: pd.Series) -> str:
        pass

    def get_correct_answer(self, row: pd.Series) -> str:
        pass

    def check_answer(self, model_answer: str, correct_answer: str) -> bool:
        pass