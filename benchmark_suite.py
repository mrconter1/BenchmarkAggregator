import os
import importlib.util
from typing import List, Dict
from benchmarks.base_benchmark import BaseBenchmark
from api_handler import get_openrouter_client

class BenchmarkSuite:
    def __init__(self):
        self.all_benchmarks = self._discover_benchmarks()
        self.client = get_openrouter_client()

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
                        instance = item()
                        discovered_benchmarks[instance.id] = instance
        
        return discovered_benchmarks

    def run(self, models: List[str], benchmark_ids: List[str] = None) -> Dict[str, Dict[str, float]]:
        if benchmark_ids is None:
            benchmarks_to_run = self.all_benchmarks
        else:
            benchmarks_to_run = {bid: self.all_benchmarks[bid] for bid in benchmark_ids if bid in self.all_benchmarks}
            if len(benchmarks_to_run) != len(benchmark_ids):
                missing = set(benchmark_ids) - set(benchmarks_to_run.keys())
                print(f"Warning: The following benchmarks were not found: {missing}")

        results = {model: {} for model in models}
        for model in models:
            for benchmark_id, benchmark in benchmarks_to_run.items():
                try:
                    benchmark.setup()
                    score = benchmark.run(model, self.client)
                    results[model][benchmark_id] = score
                finally:
                    benchmark.cleanup()
        return results

    def print_results(self, results: Dict[str, Dict[str, float]]):
        for model, benchmark_scores in results.items():
            print(f"Results for model: {model}")
            for benchmark_id, score in benchmark_scores.items():
                print(f"  {benchmark_id}: {score}")