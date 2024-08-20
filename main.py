import asyncio
from benchmark_suite import BenchmarkSuite

async def main():
    suite = BenchmarkSuite()

    # Specify which models to evaluate 
    models = [
        "openai/gpt-4o-mini-2024-07-18",
        # Add other models as needed
    ]

    # Specify which benchmarks to run
    benchmark_ids = [
        "MMLU-Pro",
        "GSM8K",
        "GPQA-Diamond"
    ]

    # Specify the number of samples to draw from each benchmark
    samples_per_benchmark = 5

    # Run the benchmarks
    results = await suite.run(models, benchmark_ids, samples_per_benchmark)

    # Print the results
    suite.print_results(results)

if __name__ == "__main__":
    asyncio.run(main())