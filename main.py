from benchmark_suite import BenchmarkSuite
import asyncio

async def main():
    suite = BenchmarkSuite()

    # Specify which models to evaluate 
    models = [
        #"anthropic/claude-3.5-sonnet",
        "openai/gpt-4o-mini-2024-07-18",  # Added another model for testing
        # Add other models as needed
    ]

    # Specify which benchmarks to run
    benchmark_ids = [
        #"MMLU-Pro",
        #"GSM8K",
        #"GPQA-Diamond",
        "ChatbotArena"
    ]

    # Specify the number of samples to draw from each benchmark
    samples_per_benchmark = 5

    # Run the benchmarks
    results = await suite.run(models, benchmark_ids, samples_per_benchmark)

    # Print the results
    suite.print_results(results)

    # Save results to JSON
    suite.save_results_to_json(results)

if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())