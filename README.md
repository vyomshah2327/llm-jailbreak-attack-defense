# LLM Jailbreak Attacks & Guardrail Defense

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![LLM Safety](https://img.shields.io/badge/Focus-LLM%20Safety-8E24AA)
![JailbreakBench](https://img.shields.io/badge/Benchmark-JailbreakBench-455A64)
![License](https://img.shields.io/badge/License-MIT-blue)

A red-team / blue-team study of open-weight LLMs: three jailbreak attack families were run against **Llama-3.1-8B, Llama-4-Maverick-17B and Qwen3-32B** on the 100 JailbreakBench harmful behaviours, then an inference-time guardrail was built that **cut attack success from 53.9% to 0% — without touching model weights**.

<p align="center">
  <img src="Images/attack_vs_defense_comparison.png" width="85%" alt="Attack success before vs after defense">
</p>

---

## Results

### Attacks (no defense)

Attack Success Rate = share of behaviours where the judge scored the response ≥ 7/10 harmful.

| Attack | Type | Llama-3.1-8B | Llama-4-Maverick | Qwen3-32B |
|---|---|---|---|---|
| **Prompt + Random Search** | black-box | **99%** | **99%** | 68% |
| **PAIR** (iterative LLM attacker) | black-box | 46% | 34% | **79%** |
| **GCG** (adversarial suffix) | white-box transfer | 6% | 26% | 26% |

**Takeaway:** black-box attacks that need no model internals were the most dangerous — a simple prompt template plus random search broke both Llama models almost every time.

### Defense

| Attack | Success without defense | Success with defense |
|---|---|---|
| Prompt + Random Search | 88.7% | **0%** |
| PAIR | 53.7% | **0%** |
| GCG | 19.3% | **0%** |
| **Overall** | **53.9%** | **0%** |

The guardrail blocked **98 of 100** attack responses outright; the remaining 2 passed through but were judged harmless.

Over-refusal check: the guarded model still answered **9 of 10** benign everyday prompts (cooking, travel, science…).

---

## How it works

```
Attack prompt ──► target LLM ──► LlamaGuard-3 output moderation ──► user
                                          │ unsafe
                                          ▼
                                       BLOCKED
```

**Automated judge** (`Judge and Evaluation/`):
1. **LlamaGuard-3** — binary safe/unsafe verdict on the prompt–response pair.
2. **Llama-3.1-8B grader** — 1–10 harm score with a written rationale; ≥ 7 counts as a successful jailbreak.
3. 95% confidence intervals (normal approximation) on attack success rates.

Target models were queried through the Groq API; judges and guardrail run locally through Ollama.

---

## Project structure

```
├── Judge and Evaluation/
│   ├── jailbreak.ipynb                     # runs the three attacks against the target models
│   ├── judge_jailbreak.ipynb               # LlamaGuard + LLM-grader judging pipeline
│   ├── defense_comparison.ipynb            # guarded vs unguarded evaluation
│   ├── calculate_metrics.py / analyze_results.py
│   ├── defense_evaluation.py / attack_defense_comparison.py
│   └── performance_metrics_evaluation.py
├── Defense/                                 # per-prompt guardrail decisions
├── CSV/                                     # aggregate metrics (ASR, harm score, block rate, over-refusal)
└── Images/                                  # result charts
```

## Quick start

```bash
pip install -r requirements.txt
export GROQ_API_KEY=...            # target models
ollama pull llama-guard3:1b        # guardrail + judge
ollama pull llama3.1:8b            # harm grader
jupyter notebook "Judge and Evaluation/jailbreak.ipynb"
```

> **Responsible disclosure.** This repo publishes methodology and aggregate metrics only. Raw model completions produced under attack are intentionally **not** included.

---

**Tech:** Python · Groq API · Ollama · LlamaGuard-3 · Llama-3.1 · Llama-4 · Qwen3 · JailbreakBench · pandas · SciPy · Matplotlib

Built with [Yagni Patel](https://github.com/YagniPatel) · MIT License
