import aiohttp
import pandas as pd
from io import StringIO
from tabulate import tabulate
from benchmarks.base_benchmark import BaseBenchmark

class LiveBenchCSVBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.id = "LiveBench"
        self.url = "https://livebench.ai/table_2024_07_26.csv"
        self.model_mapping = {
            "openai/gpt-3.5-turbo-0613": "gpt-3.5-turbo-0125",
            "openai/gpt-4o-mini-2024-07-18": "gpt-4o-mini-2024-07-18",
            "openai/gpt-4o-2024-08-06": "chatgpt-4o-latest",
            "anthropic/claude-3-sonnet": "claude-3-sonnet-20240229",
            "anthropic/claude-3-opus": "claude-3-opus-20240229",
            "anthropic/claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
            "meta-llama/llama-3.1-70b-instruct": "meta-llama-3.1-70b-instruct-turbo",
            "meta-llama/llama-3.1-405b-instruct": "meta-llama-3.1-405b-instruct-turbo",
            "mistralai/mistral-large": "mistral-large-2407"
        }

    async def setup(self):
        await super().setup()
        async with aiohttp.ClientSession() as session:
            async with session.get(self.url) as response:
                csv_content = await response.text()
        
        df = pd.read_csv(StringIO(csv_content))
        df['average_score'] = df.iloc[:, 1:].mean(axis=1)
        self.df = df[['model', 'average_score']]

    async def get_dataset(self) -> pd.DataFrame:
        return self.df

    def get_question(self, row: pd.Series) -> str:
        return f"What is the LiveBench score for {row['model']}?"

    def get_correct_answer(self, row: pd.Series) -> str:
        return str(row['average_score'])

    def check_answer(self, model_answer: str, correct_answer: str) -> bool:
        try:
            return abs(float(model_answer) - float(correct_answer)) < 0.01
        except ValueError:
            return False

    async def run(self, model: str, client, df: pd.DataFrame = None, samples: int = None) -> float:
        livebench_model = self.model_mapping.get(model, model)
        if livebench_model in df['model'].values:
            score = df[df['model'] == livebench_model]['average_score'].values[0]
            return score / 100.0  # Convert percentage to decimal
        else:
            print(f"Warning: No data available for model {model}")
            return 0.0