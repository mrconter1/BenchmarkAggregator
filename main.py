from benchmark_suite import BenchmarkSuite
import asyncio

async def main():
    suite = BenchmarkSuite()

    # Specify which models to evaluate 
    models = [
        "openai/gpt-3.5-turbo-0613",
        "openai/gpt-4o-mini-2024-07-18",
        "openai/gpt-4o-2024-08-06",
        "anthropic/claude-3-sonnet",
        "anthropic/claude-3-opus",
        "anthropic/claude-3.5-sonnet",
        "meta-llama/llama-3.1-70b-instruct",
        "meta-llama/llama-3.1-405b-instruct",
        "mistralai/mistral-large"
    ]

    # Specify which benchmarks to run
    benchmark_ids = [
        "MMLU-Pro",
        "GSM8K",
        "GPQA-Diamond",
        "ChatbotArena"
    ]

    # Specify the number of samples to draw from each benchmark
    samples_per_benchmark = 2

    # Run the benchmarks
    results = await suite.run(models, benchmark_ids, samples_per_benchmark)

    # Print the results
    suite.print_results(results)

    # Save results to JSON
    suite.save_results_to_json(results)

if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())