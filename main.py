from api_handler import get_openrouter_client, query_model
from benchmarks.mmlu_pro_benchmark import MMULProBenchmark

def main():
    # Single model to test
    model = "openai/gpt-4o-mini-2024-07-18"

    # Initialize OpenRouter client
    client = get_openrouter_client()

    # Initialize benchmarks
    mmlu_pro = MMULProBenchmark()

    print(f"Running benchmark for {model}")
    
    # Run MMLU-Pro benchmark
    mmlu_pro_score = mmlu_pro.run_benchmark(model, client)
    print(f"MMLU-Pro score for {model}: {mmlu_pro_score}")

    # Add more benchmarks here as needed

if __name__ == "__main__":
    main()