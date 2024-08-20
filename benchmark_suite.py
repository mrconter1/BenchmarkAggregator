import os
import importlib.util
import asyncio
import json
from typing import List, Dict
from benchmarks.base_benchmark import BaseBenchmark
from api_handler import get_openrouter_client, RateLimitedClient

class BenchmarkSuite:
    def __init__(self):
        self.all_benchmarks = self._discover_benchmarks()
        self.client = None
        self.benchmark_data = {}

    def _discover_benchmarks(self):
        discovered_benchmarks = {}
        benchmark_dir = os.path.join(os.path.dirname(__file__), 'benchmarks')
        
        for filename in os.listdir(benchmark_dir):
            if filename.endswith('.py') and filename != 'base_benchmark.py':
                module_name = filename[:-3]  # Remove .py extension
                module_path = os.path.join(benchmark_dir, filename)
                
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                for item_name in dir(module):
                    item = getattr(module, item_name)
                    if isinstance(item, type) and issubclass(item, BaseBenchmark) and item != BaseBenchmark:
                        discovered_benchmarks[item().id] = item
        
        return discovered_benchmarks

    async def run(self, models: List[str], benchmark_ids: List[str] = None, samples_per_benchmark: int = None) -> Dict[str, Dict[str, float]]:
        if benchmark_ids is None:
            benchmarks_to_run = self.all_benchmarks
        else:
            benchmarks_to_run = {bid: self.all_benchmarks[bid] for bid in benchmark_ids if bid in self.all_benchmarks}
            if len(benchmarks_to_run) != len(benchmark_ids):
                missing = set(benchmark_ids) - set(benchmarks_to_run.keys())
                print(f"Warning: The following benchmarks were not found: {missing}")

        openai_client = get_openrouter_client()
        self.client = RateLimitedClient(openai_client, rate_limit=10)

        # Load benchmark data once
        await self._load_benchmark_data(benchmarks_to_run, samples_per_benchmark)

        results = {model: {} for model in models}
        tasks = []

        for model in models:
            for benchmark_id, benchmark_class in benchmarks_to_run.items():
                task = asyncio.create_task(self._run_benchmark(model, benchmark_id, benchmark_class))
                tasks.append(task)

        benchmark_results = await asyncio.gather(*tasks)

        for model, benchmark_id, score in benchmark_results:
            results[model][benchmark_id] = score

        return results

    async def _load_benchmark_data(self, benchmarks_to_run, samples_per_benchmark):
        for benchmark_id, benchmark_class in benchmarks_to_run.items():
            if benchmark_id not in self.benchmark_data:
                benchmark = benchmark_class()
                await benchmark.setup()
                df = await benchmark.get_dataset()
                if (not benchmark.id == "ChatbotArena") and samples_per_benchmark is not None and samples_per_benchmark < len(df):
                    df = df.sample(n=samples_per_benchmark, random_state=42)
                self.benchmark_data[benchmark_id] = df
                await benchmark.cleanup()

    async def _run_benchmark(self, model: str, benchmark_id: str, benchmark_class):
        benchmark = benchmark_class()
        benchmark.df = self.benchmark_data[benchmark_id]  # Use pre-loaded data
        try:
            score = await benchmark.run(model, self.client, self.benchmark_data[benchmark_id])
            return model, benchmark_id, score
        finally:
            # No need to call cleanup here as we're not setting up each time
            pass

    def print_results(self, results: Dict[str, Dict[str, float]]):
        for model, benchmark_scores in results.items():
            print(f"Results for model: {model}")
            for benchmark_id, score in benchmark_scores.items():
                print(f"  {benchmark_id}: {score:.2%}")

    def save_results_to_json(self, results: Dict[str, Dict[str, float]], filename='data.json'):
        formatted_results = []
        for model, benchmark_scores in results.items():
            model_data = {
                "model": model,
                "benchmarks": [
                    {"name": benchmark_id, "score": score}
                    for benchmark_id, score in benchmark_scores.items()
                ]
            }
            formatted_results.append(model_data)
        
        with open(filename, 'w') as f:
            json.dump(formatted_results, f, indent=2)
        
        print(f"Results saved to {filename}")