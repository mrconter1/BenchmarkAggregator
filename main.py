import asyncio
from benchmark_suite import BenchmarkSuite

async def main():
    suite = BenchmarkSuite()

    # Specify which models to evaluate 
    models = [
        "openai/gpt-4o-mini-2024-07-18",
        "openai/gpt-4o-2024-08-06"
        "anthropic/claude-3.5-sonnet",
    ]

    # Specify which benchmarks to run
    benchmark_ids = [
        "MMLU-Pro",
        "GSM8K",
        "GPQA-Diamond",
        "DROP"
    ]

    # Run the benchmarks
    results = await suite.run(models, benchmark_ids)

    # Print the results
    suite.print_results(results)

if __name__ == "__main__":
    asyncio.run(main())