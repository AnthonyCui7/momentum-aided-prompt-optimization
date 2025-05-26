# Momentum-Aided Natural Language Gradient Descent for Prompt Optimization

[[Paper]](https://arxiv.org/abs/2410.19499) (NAACL SRW 2025)

Prompt optimization is crucial for improving output quality of Large Language Models (LLMs), but many existing methods are inefficient, requiring extensive computation and manual tuning. We propose **M**omentum-**A**ided **P**rompt **O**ptimization (MAPO), which builds on ProTeGi [(Pryzant et al., 2023)](https://arxiv.org/abs/2305.03495) by incorporating positive natural language "gradients" and a momentum-based memory mechanism to refine prompts while avoiding local minima and oscillations. It also employs beam search and an Upper Confidence Bound (UCB) algorithm for balanced candidate expansion and selection. MAPO achieves faster convergence time with fewer API calls and higher performance than ProTeGi, demonstrating it as a robust and scalable solution for automated prompt optimization in LLMs.

## Usage

Quickstart command:
```
time python main.py --task ethos --prompts prompts/ethos.md --data_dir data/ethos  --evaluator ucb
```

Change the dataset name in the quickstart command accordingly to run experiments on a different benchmark.
