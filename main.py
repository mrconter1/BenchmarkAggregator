from benchmark_suite import BenchmarkSuite
from model import Model
import asyncio

async def main():
    suite = BenchmarkSuite()

    # Create Model instances using OpenRouter model ids and model release dates
    models = [
        Model("openai/gpt-3.5-turbo-0125", "2024-01-24"),
        Model("openai/gpt-4o-mini-2024-07-18", "2024-07-18"),
        Model("openai/gpt-4o-2024-08-06", "2024-08-06"),
        Model("anthropic/claude-3-sonnet", "2024-02-29"),
        Model("anthropic/claude-3.5-sonnet", "2024-06-20"),
        Model("meta-llama/llama-3.1-70b-instruct", "2024-07-23"),
        Model("meta-llama/llama-3.1-405b-instruct", "2024-07-23"),
        Model("mistralai/mistral-large", "2024-07-24")
    ]

    # Specify which benchmarks to run
    benchmark_ids = [
        "MMLU-Pro",
        "GPQA-Diamond",
        "ChatbotArena",
        "MATH-Hard",
        "MuSR",
        "ARC-Challenge",
        "HellaSwag",
        "LiveBench",
        "MGSM"
    ]

    # Specify the number of samples to draw from each benchmark
    samples_per_benchmark = 100

    # Run the benchmarks
    results = await suite.run(models, benchmark_ids, samples_per_benchmark)

    # Print the results
    suite.print_results(results)

    # Save results to JSON
    suite.save_results_to_json(results)

if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())