# ARC-AGI Master Development Hub (2019–2026) 🏛️

A central repository documenting and organizing the evolution of AI reasoning research, benchmarks, and model developments from **ARC-AGI-1** to **ARC-AGI-3**.

---

## ⏳ The Evolution Timeline (2019–2026)

### 🚀 2019: The Conceptual Launch
* **Thesis & Dataset**: François Chollet publishes his seminal paper *"On the Measure of Intelligence"* and releases the original benchmark dataset, **ARC-AGI-1**.
* **Initial Results**: Early deep learning systems achieve near 0% accuracy, proving resistant to standard gradient descent methods.

### 🧩 2020–2023: The Program Synthesis Era
* **2020 Kaggle Competition**: The first formal Kaggle ARC Challenge is hosted. Solo developer **"icecuber"** wins with a 20% success rate.
* **Methodology**: Relying on hand-built Domain-Specific Languages (DSLs) and brute-force symbolic program synthesis.
* **LLM Stagnation**: For three years, Large Language Models (LLMs) remain stuck below 5% accuracy. Top symbolic methods slowly reach ~33% by 2023.

### ⚡ 2024: The Test-Time Compute Breakthrough
* **ARC Prize Launch**: Mike Knoop and François Chollet launch the ARC Prize Foundation with a $1,000,000+ bounty for open-source AI researchers.
* **Inference Adaptation**: Researchers shift focus from parameter scaling to test-time adaptation (inference compute). Systems powered by GPT-4o reach >50% accuracy by generating and validating thousands of program candidates against prompt examples.
* **OpenAI o3 Peak**: OpenAI's reasoning model `o3` achieves 87.5% on a semi-private evaluation set via high-compute test-time training (172x standard compute resources).

### 📈 2025: Scaling the Challenge (ARC-AGI-2)
* **ARC-AGI-2 Release**: Launched in March 2025 as frontier models approach human baselines on the original test set.
* **Granular Complexity**: Introduces over 1,000 highly granular, novel tasks specifically designed to stress-test abstract reasoning and generalization.
* **Benchmark Shift**: While human subjects maintain near 100% scores, AI model performance drops significantly. NVIDIA's **NVARC** team wins the 2025 prize with 24% accuracy using a 4B parameter model combined with synthetic dataset training.

### 🎮 2026: Interactive Agentic Reasoning (ARC-AGI-3)
* **ARC-AGI-3 Paradigm Shift**: In early 2026, the ARC Prize Foundation transitions from single-turn static grid puzzles to an interactive reasoning paradigm.
* **ARCEngine Environments**: Introduces ~100 completely novel, unseen game environments powered by the open-source **ARCEngine**.
* **Autonomous Agent Benchmark**: Autonomous AI agents are dropped into dynamic environments without initial instructions and must explore, intuit operational rules, and execute multi-step plans. Major frontier labs now report ARC-AGI-3 scores on model cards to demonstrate agentic reasoning capabilities.

---

## 📁 Repository Folder Structure

```
ARC_Prize_2024/
├── README.md                      # Central Documentation & Evolution Timeline
├── CodeStructureSummary.md        # Technical breakdown of architecture and solver logic
├── the_architects.pdf             # Core research methodology paper
│
├── legacy/                        # (2020–2023) Program Synthesis & Early Deep Learning
│   ├── experiments/               # Early neural (CNN/LSTM, RL, RNN, Autoencoder) experiments
│   ├── literature/                # Vision Transformer research papers & ViT scripts
│   └── notebooks/                 # Historical Google Colab & local research notebooks
│
├── arc-prize-2024-main/           # (2024) ARC-AGI-1 Test-Time Compute & DSL Solvers
│   └── plain/                     # Core Kaggle submission framework & DSL solvers
│
├── ARC AGI 2/                     # (2025) ARC-AGI-2 High-Granularity Benchmark
│   └── README.md                  # ARC-AGI-2 overview, 4B synthetic training setup
│
└── ARC AGI 3/                     # (2026) ARC-AGI-3 Interactive ARCEngine Benchmark
    └── README.md                  # ARC-AGI-3 environment specifications & agentic solvers
```

---

## 🛠 Sub-Project Quick Start

- **`arc-prize-2024-main/`**: Contains the primary test-time compute submission framework for the 2024 ARC Prize competition.
- **`ARC AGI 2/`**: Development environment for 2025 granular task generation and synthetic training pipelines.
- **`ARC AGI 3/`**: Workspace for ARCEngine interactive game environments and autonomous agentic exploration algorithms.
- **`legacy/`**: Archive of early heuristic, RNN, RL, and symbolic program synthesis experiments.
