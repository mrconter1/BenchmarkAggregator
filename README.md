# BenchmarkAggregator 🚀

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/yourusername/benchmarkaggregator/issues)

Rigorous, unbiased, and adaptable LLM evaluations across diverse AI benchmarks.

[View Leaderboard](https://benchmark-aggregator-lvss.vercel.app/) | [Features](#🌟-features) | [Benchmarks](#🏆-current-benchmarks) | [FAQ](#🤔-faq)

## 🎯 Introduction

The BenchmarkAggregator framework serves as a **central** hub, addressing the critical need for consistent model evaluation in the AI community. By providing comprehensive comparisons of Large Language Models (LLMs) across challenging, well-respected benchmarks in one unified location, it offers a holistic, fair, and scalable view of model performance. Our approach balances depth of evaluation with resource constraints, ensuring fair comparisons while maintaining practicality and accessibility from a single, authoritative source.

## 📊 Model Performance Overview

| Model | Average Score |
|-------|---------------|
| gpt-4o-2024-08-06 | 69.0 |
| claude-3.5-sonnet | 66.2 |
| gpt-4o-mini-2024-07-18 | 62.1 |
| mistral-large | 61.4 |
| llama-3.1-405b-instruct | 59.8 |
| llama-3.1-70b-instruct | 58.4 |
| claude-3-sonnet | 53.2 |
| gpt-3.5-turbo-0125 | 34.8 |

For detailed scores across all benchmarks, visit our [leaderboard](https://benchmark-aggregator-lvss.vercel.app/).

## 🌟 Features

1. 🏆 Incorporates top, most respected benchmarks in the AI community
2. 📊 Balanced evaluation using 100 randomly drawn samples per benchmark (adjustable)
3. 🔌 Quick and easy integration of new benchmarks and models (uses [OpenRouter](https://openrouter.ai/), making the addition of new models absolutely trivial)
4. 📈 Holistic performance view through score averaging across diverse tasks
5. ⚖️ Efficient approach balancing evaluation depth with resource constraints

## 🏆 Current Benchmarks
1. [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
2. [GPQA-Diamond](https://huggingface.co/datasets/Idavidrein/gpqa)
3. [ChatbotArena](https://chat.lmsys.org/) 
4. [MATH-Hard](https://huggingface.co/datasets/lighteval/MATH-Hard) 
5. [MuSR](https://huggingface.co/datasets/TAUR-Lab/MuSR) 
6. [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc) 
7. [HellaSwag](https://rowanzellers.com/hellaswag/) 
8. [LiveBench](https://livebench.ai/) 
9. [MGSM](https://huggingface.co/datasets/juletxara/mgsm) 

📖 [Learn more about each benchmark on our website](https://benchmark-aggregator-lvss.vercel.app/)

## 🤔 FAQ

<details>
<summary>Why not run all questions for each benchmark?</summary>
Running all questions for each benchmark would be cost-prohibitive. Our approach balances comprehensive evaluation with practical resource constraints.
</details>

<details>
<summary>How are benchmark samples chosen?</summary>
The samples are randomly drawn from the larger benchmark dataset. The same sample set is used for each model to ensure consistency and fair comparison across all evaluations.
</details>

<details>
<summary>Why are certain models like Claude 3 Opus and GPT-4 turbo absent?</summary>
These models are significantly more expensive to query compared to many others. Their absence is due to cost considerations in running the benchmarks.
</details>

<details>
<summary>How easy is it to add new benchmarks or models?</summary>
Adding new benchmarks or models is designed to be quick and efficient. For benchmarks, it can take only a few minutes to integrate an existing one. For models, we use [OpenRouter](https://openrouter.ai/), which covers basically all closed and open-source options. **To add a model, simply find its ID on the OpenRouter website and include it in our framework. This makes adding new models absolutely trivial!**
</details>

<details>
<summary>How are the scores from Chatbot Arena calculated?</summary>
The scores for Chatbot Arena are fetched directly from their website. These scores are then normalized against the values of other models in this benchmark.
</details>

👉 [View more FAQs on our website](https://benchmark-aggregator-lvss.vercel.app/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We're grateful to the creators and maintainers of the benchmark datasets used in this project, as well as to [OpenRouter](https://openrouter.ai/) for making model integration seamless.

---

<p align="center">
  Made with ❤️ by the AI community
</p>

<p align="center">
  <a href="https://benchmark-aggregator-lvss.vercel.app/">Website</a>
</p>