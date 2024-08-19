import asyncio
from benchmark_suite import BenchmarkSuite

async def main():
    suite = BenchmarkSuite()

    # Specify which models to evaluate 
    models = ["openai/gpt-4o-mini-2024-07-18"]

    # Specify which benchmarks to run
    benchmark_ids = ["MMLU-Pro"]  # Use the benchmark ID here

    results = await suite.run(models, benchmark_ids)
    suite.print_results(results)

if __name__ == "__main__":
    asyncio.run(main())