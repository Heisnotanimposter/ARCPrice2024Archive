# ARC-AGI-3 (2026) Interactive Agentic Benchmark Hub 🎮

Welcome to the **ARC-AGI-3** project module. This directory contains the agentic architecture, world modeling engines, and **ARCEngine** environment wrappers designed for the **2026 Interactive Agentic Reasoning Benchmark**.

---

## 🕹️ Benchmark Overview

In early 2026, the ARC Prize Foundation introduced **ARC-AGI-3**, fundamentally shifting the frontier from single-turn grid transformation puzzles to **interactive dynamic environments**.

### Key Characteristics
* **ARCEngine Framework**: Built on top of the open-source **ARCEngine**, offering ~100 completely novel, unseen 2D interactive environments.
* **Uninstructed Exploration**: AI agents are placed into game environments with zero natural language instructions or explicit rule sets.
* **Autonomous Rule Discovery**: Agents must observe state changes $\Delta S = f(S, A)$, intuit environmental physics and win conditions, and plan multi-step strategies.
* **Public Model Card Standard**: Major AI research labs now benchmark autonomous agents on ARC-AGI-3 to demonstrate general agentic intelligence.

---

## 🧠 Open-Source Agentic Architectures

### 1. Active Inference & World Model Engine (`agents/world_model.py`)
* **State Transition Hypothesis**: Maintains a probabilistic set of hypotheses regarding environmental mechanics (collision physics, item interactions, score triggers).
* **Curiosity-Driven Exploration**: Action selection prioritizes actions that minimize uncertainty in the internal world model (Active Inference).

### 2. Neural-Symbolic MCTS Planner (`agents/mcts_planner.py`)
* **Hierarchical Planning**: Uses Monte Carlo Tree Search (MCTS) guided by a LLM/VLM policy prior to search for multi-step goal paths.
* **Backtracking & Replanning**: If an environment action yields unexpected transitions, the planner immediately updates its world model hypotheses and replans.

---

## 📁 Codebase Architecture

```
ARC AGI 3/
├── README.md                     # Comprehensive architecture guide
├── configs/                      # Search, exploration & agent parameters
│   └── agent_config.json         # MCTS rollout limit, temperature & curiosity parameters
├── environments/                 # ARCEngine environment wrappers
│   ├── arc_engine_env.py         # Open-source ARCEngine Gym/PettingZoo API wrapper
│   └── game_loader.py            # Environment loader for 100 benchmark games
└── agents/                       # Autonomous agent modules
    ├── rule_intuition_agent.py   # Rule intuition & hypothesis tracking agent
    ├── world_model.py            # Neural-symbolic environment transition model
    ├── mcts_planner.py           # Monte Carlo Tree Search action planner
    └── runner.py                 # Multi-environment evaluation harness
```

---

## 🚀 Open-Sourced Core Implementation

Below are the core agentic script templates provided in `environments/` and `agents/`:

### 1. ARCEngine Wrapper (`environments/arc_engine_env.py`)
Provides `reset()`, `step(action)`, and `render()` hooks to interface seamlessly with standard Python reinforcement learning and agentic frameworks.

### 2. Rule Intuition Agent (`agents/rule_intuition_agent.py`)
Explores unseen game environments, builds hypotheses on state transitions, and chooses actions to maximize goal achievement.

---

## ⚙️ Quick Start

```bash
# 1. Test ARCEngine Environment Interface
python environments/arc_engine_env.py --game_id 42

# 2. Run Autonomous Rule Intuition Agent
python agents/runner.py --agent_type rule_intuition --num_games 10 --max_steps 500

# 3. Evaluate Neural-Symbolic MCTS Planner
python agents/runner.py --agent_type mcts_vlm --mcts_simulations 200
```
