import asyncio
from benchmark_suite import BenchmarkSuite

async def main():
    suite = BenchmarkSuite()

    # Specify which models to evaluate 
    models = [
        #"openai/gpt-3.5-turbo-0613"
        "openai/gpt-4o-mini-2024-07-18",
        #"openai/gpt-4o-2024-08-06"
        #"anthropic/claude-3-sonnet",
        #"anthropic/claude-3-opus",
        #"anthropic/claude-3.5-sonnet",
        #"google/gemini-pro-1.5-exp",
        #"meta-llama/llama-3.1-70b-instruct",
        #"meta-llama/llama-3.1-405b-instruct",
        #"mistralai/mistral-large",
    ]

    # Specify which benchmarks to run
    benchmark_ids = [
        "MMLU-Pro",
        "GSM8K",
    ]

    # Run the benchmarks
    results = await suite.run(models, benchmark_ids)

    # Print the results
    suite.print_results(results)

if __name__ == "__main__":
    asyncio.run(main())