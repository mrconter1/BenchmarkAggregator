import os
import importlib.util
import asyncio
import json
from typing import List, Dict, Any
from benchmarks.base_benchmark import BaseBenchmark
from api_handler import get_openrouter_client, RateLimitedClient
from model import Model

class BenchmarkSuite:
    def __init__(self):
        self.all_benchmarks = self._discover_benchmarks()
        self.client = None
        self.full_benchmark_data = {}
        self.sampled_benchmark_data = {}
        self.samples_per_benchmark = None

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

    async def run(self, models: List[Model], benchmark_ids: List[str] = None, samples_per_benchmark: int = None) -> Dict[str, Dict[str, Any]]:
        self.samples_per_benchmark = samples_per_benchmark
        benchmarks_to_run = {bid: self.all_benchmarks[bid] for bid in benchmark_ids if bid in self.all_benchmarks}
        if len(benchmarks_to_run) != len(benchmark_ids):
            missing = set(benchmark_ids) - set(benchmarks_to_run.keys())
            print(f"Warning: The following benchmarks were not found: {missing}")

        openai_client = get_openrouter_client()
        self.client = RateLimitedClient(openai_client, rate_limit=5)

        # Load benchmark data once
        await self._load_benchmark_data(benchmarks_to_run, samples_per_benchmark)

        results = {model.id: {"releaseDate": model.release_date, "benchmarks": {}} for model in models}
        tasks = []

        for model in models:
            for benchmark_id, benchmark_class in benchmarks_to_run.items():
                task = asyncio.create_task(self._run_benchmark(model, benchmark_id, benchmark_class))
                tasks.append(task)

        benchmark_results = await asyncio.gather(*tasks)

        for model, benchmark_id, score in benchmark_results:
            results[model.id]["benchmarks"][benchmark_id] = score

        return results

    async def _load_benchmark_data(self, benchmarks_to_run, samples_per_benchmark):
        for benchmark_id, benchmark_class in benchmarks_to_run.items():
            if benchmark_id not in self.full_benchmark_data:
                benchmark = benchmark_class()
                await benchmark.setup()
                df = await benchmark.get_dataset()
                self.full_benchmark_data[benchmark_id] = df
                
                if benchmark_id not in ["ChatbotArena", "LiveBench"] and samples_per_benchmark is not None and samples_per_benchmark < len(df):
                    sampled_df = df.sample(n=samples_per_benchmark, random_state=1337)
                else:
                    sampled_df = df
                
                self.sampled_benchmark_data[benchmark_id] = sampled_df
                await benchmark.cleanup()

    async def _run_benchmark(self, model: Model, benchmark_id: str, benchmark_class):
        benchmark = benchmark_class()
        benchmark.df = self.sampled_benchmark_data[benchmark_id]  # Use sampled data for running
        try:
            score = await benchmark.run(model.id, self.client, self.sampled_benchmark_data[benchmark_id])
            return model, benchmark_id, score
        finally:
            pass

    def print_results(self, results: Dict[str, Dict[str, Any]]):
        for model_id, model_data in results.items():
            print(f"Results for model: {model_id} (Release Date: {model_data['releaseDate']})")
            for benchmark_id, score in model_data['benchmarks'].items():
                print(f"  {benchmark_id}: {score:.2%}")

    def save_results_to_json(self, results: Dict[str, Dict[str, Any]], filename='data.json'):
        formatted_results = []
        for model_id, model_data in results.items():
            model_result = {
                "model": model_id,
                "releaseDate": model_data['releaseDate'],
                "benchmarks": []
            }
            for benchmark_id, score in model_data['benchmarks'].items():
                benchmark_info = {
                    "name": benchmark_id,
                    "score": round(score * 100, 2)
                }
                
                if benchmark_id not in ["ChatbotArena", "LiveBench"]:
                    total_samples = len(self.full_benchmark_data[benchmark_id])
                    drawn_samples = len(self.sampled_benchmark_data[benchmark_id])
                    benchmark_info["samplesDrawn"] = drawn_samples
                    benchmark_info["totalSamples"] = total_samples
                else:
                    benchmark_info["samplesDrawn"] ="N/A"
                    benchmark_info["totalSamples"] ="N/A"
                
                model_result["benchmarks"].append(benchmark_info)
            
            formatted_results.append(model_result)
        
        with open(filename, 'w') as f:
            json.dump(formatted_results, f, indent=2)
        
        print(f"Results saved to {filename}")