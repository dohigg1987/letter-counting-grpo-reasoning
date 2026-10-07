# Teaching an LLM to Reason: Letter Counting with GRPO

**Author:** Dennis O'Higgins

This project fine-tunes `Qwen2.5-3B-Instruct` with LoRA adapters and Group Relative Policy Optimization (GRPO) so the model counts the occurrences of a letter in a word by spelling the word letter by letter and keeping a running total.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dohigg1987/letter-counting-grpo-reasoning/blob/main/gen_ai_fundamentals_project_starter.ipynb)

## Contents
- `gen_ai_fundamentals_project_starter.ipynb`: the completed project notebook (LoRA configuration, chain-of-thought prompt baseline, five reward functions, GRPO training runs, reward plots, and before and after comparisons).

## Running
The notebook needs an NVIDIA GPU with at least 16 GB of memory (for example a T4) and the pinned package versions from the course `requirements.txt`, installed with `pip install -r requirements.txt --no-deps`.
