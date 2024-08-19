from typing import List, Dict
import inspect
import benchmarks
from benchmarks.base_benchmark import BaseBenchmark

class BenchmarkSuite:
    def __init__(self):
        self.all_benchmarks = self._discover_benchmarks()

    def _discover_benchmarks(self):
        discovered_benchmarks = {}
        for name, obj in inspect.getmembers(benchmarks):
            if inspect.ismodule(obj):
                for class_name, class_obj in inspect.getmembers(obj):
                    if inspect.isclass(class_obj) and issubclass(class_obj, BaseBenchmark) and class_obj != BaseBenchmark:
                        instance = class_obj()
                        discovered_benchmarks[instance.id] = instance
        return discovered_benchmarks

    def run(self, models: List[str], client, benchmark_ids: List[str] = None) -> Dict[str, Dict[str, float]]:
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
                    score = benchmark.run(model, client)
                    results[model][benchmark_id] = score
                finally:
                    benchmark.cleanup()
        return results

    def print_results(self, results: Dict[str, Dict[str, float]]):
        for model, benchmark_scores in results.items():
            print(f"Results for model: {model}")
            for benchmark_id, score in benchmark_scores.items():
                print(f"  {benchmark_id}: {score}")