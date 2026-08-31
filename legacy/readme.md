# ARC-AGI Historical Research & Legacy Archive (2020–2023) 📚

This directory contains historical code, exploratory deep learning models, reinforcement learning experiments, and research literature developed during early iterations of the **ARC-AGI** challenge.

---

## 📁 Organized Archive Structure

```
legacy/
├── README.md                 # Detailed catalog of historical models & research assets
├── LICENSE                   # Open-source license documentation
│
├── experiments/              # Early neural & symbolic model prototypes
│   ├── LSTM/                 # Attention-augmented Multi-head Conv-LSTM solvers
│   ├── RL/                   # Reinforcement Learning (Stable-Baselines3) solvers & PDFs
│   ├── RNN/                  # Recurrent Neural Network grid transformation scripts
│   └── sample/               # Autoencoder CNNs, GPT-2 pattern engines, ML ensembles
│
├── literature/               # Key research papers & literature surveys
│   ├── ViT for ARC Challenges.pdf
│   ├── Vision Transformer with Sparse Scan Prior.pdf
│   ├── A Survey of Vision Transformers in Autonomous Driving.pdf
│   └── vision_transformer.py # Custom PyTorch Vision Transformer implementation
│
└── notebooks/                # Historical Google Colab & local exploration notebooks
    ├── arc-prize-2024-colab/ # Primary Colab training checkpoints & pipelines
    ├── arc-prize-2024-colab2/# Secondary Colab experimentation environment
    └── arc-prize-2024-local/ # Local PyTorch/Jupyter experimentation scripts
```

---

## 🔬 Summary of Legacy Approaches

### 1. Neural Architecture Experiments (`experiments/`)
* **Convolutional-LSTM with Multi-Head Attention (`experiments/LSTM/`)**:
  * Implements `Attention_multihead_CNLSTM.py` to capture spatial grid features across time steps.
* **Deep Reinforcement Learning (`experiments/RL/`)**:
  * Leverages `sb3.py` (PPO/DQN via Stable-Baselines3) to treat grid transformations as MDP environment actions.
* **Autoencoder & GPT-2 Pattern Engine (`experiments/sample/`)**:
  * `arc_2024_ae_based_cnn_solver.py`: Autoencoder-based convolutional solver for grid reconstruction.
  * `gpt2_pattern_engine.py`: Early experiment converting 2D grid arrays into 1D token sequences for GPT-2 autoregressive training.

### 2. Literature & Vision Transformers (`literature/`)
* **ViT for ARC**: Exploration of Vision Transformers for zero-shot grid pattern recognition and token reduction techniques (`vision_transformer.py`).

### 3. Notebook Environments (`notebooks/`)
* Contains historical checkpoint managers, data parsers, and exploratory visualization tools.
