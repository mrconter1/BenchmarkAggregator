from api_handler import get_openrouter_client
from benchmark_suite import BenchmarkSuite

def main():
    models = ["openai/gpt-4o-mini-2024-07-18"]
    client = get_openrouter_client()

    suite = BenchmarkSuite()

    # Specify which benchmarks to run
    benchmark_ids = ["MMLU-Pro"]  # Use the benchmark ID here

    results = suite.run(models, client, benchmark_ids)
    suite.print_results(results)

if __name__ == "__main__":
    main()