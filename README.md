# Conditional Inference for Financial Order Routing Using Energy-Based Models

[![Open in Colab - Synthetic Data](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brumoema/ising-conditional-routing/blob/main/experiments/synthetic_data.ipynb)
[![Open in Colab - Real Data](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brumoema/ising-conditional-routing/blob/main/experiments/real_data.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A novel approach to Smart Order Routing (SOR) using THRML, a JAX-based library for probabilistic graphical models, to model venue correlations as an Ising energy-based model.**

---

## 📖 Overview

In multi-venue financial markets, traders must decide where to route orders without full visibility into execution quality across all venues. Traditional multi-armed bandit approaches treat venues as independent, missing valuable cross-venue correlation signals.

This research proposes modeling the multi-venue system as an **Ising Energy-Based Model (EBM)**, where:
- Each venue is represented as a node (spin) in a graph
- Edges capture pairwise correlations between venues
- Node biases represent each venue's individual tendency toward favorable/unfavorable outcomes

By leveraging **conditional (clamped) sampling** with [THRML](https://github.com/extropic-ai/thrml), our agent observes partial market state and infers the best routing decision based on learned correlations.

---

## 🚀 Key Results

### Synthetic Data (N=3 venues, K=1 context)

#### Fixed Context Mode

| Scenario | Ctx-ε-Greedy | Ctx-Thompson | THRML | THRML Benefit |
|----------|--------------|--------------|-------|---------------|
| IID Venues | 0.00 | 0.00 | 0.00 | Tie (Optimal) |
| Correlated | 206.40 | 5.99 | **2.21** | **Win** |
| Regime Shift | 2520.47 | 3555.48 | **2154.07** | **Win (-39%)** |

#### Random Context Mode

| Scenario | Ctx-ε-Greedy | Ctx-Thompson | THRML | THRML Benefit |
|----------|--------------|--------------|-------|---------------|
| IID Venues | 664.12 | 12.84 | **0.96** | **Win** |
| Correlated | 1897.93 | **1416.11** | 1418.24 | Competitive |
| Regime Shift | 2052.62 | 2596.88 | **1444.48** | **Win (-44%)** |

### Real Cryptocurrency Data (Coinbase, Kraken, Bitstamp)

| Context Mode | Ctx-ε-Greedy | Ctx-Thompson | THRML |
|--------------|--------------|--------------|-------|
| Fixed | 4823.86 | 4532.97 | **3914.27** |
| Random | 5141.93 | 5000.64 | **3775.45** |

**THRML achieves 14-25% regret reduction** compared to state-of-the-art contextual bandit approaches.

---

## 🏗️ Project Structure

```
.
├── index.qmd                      # Main research article (Quarto)
├── experiments/
│   ├── synthetic_data.ipynb       # Synthetic experiments notebook
│   └── real_data.ipynb            # Real cryptocurrency data notebook
├── docs/                          # Rendered HTML (GitHub Pages)
│   └── index.html                 # Published article
├── _quarto.yml                    # Quarto configuration
├── styles.css                     # Custom styling
├── LICENSE                        # MIT License
└── README.md                      # This file
```

---

## 🔬 Methodology

### Agents Compared

| Agent | Description |
|-------|-------------|
| **Contextual ε-Greedy** | Maintains context-specific success/count statistics with ε=0.1 exploration |
| **Contextual Thompson Sampling** | Uses Beta-distributed posteriors conditioned on context |
| **THRML** | Leverages Ising model correlations with clamped Gibbs sampling for conditional inference |

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_venues` | 3 | Number of trading venues |
| `n_steps` | 10,000 | Steps per experiment run |
| `n_seeds` | 200 | Independent runs for statistical significance |
| `discount_factor` | 0.995 | Forgetting factor for non-stationary adaptation |
| `learning_rate` | 0.05 | THRML learning rate |
| `coupling_decay` | 0.995 | Decay factor for edge weights |
| `steps_per_sample` | 4 | Gibbs sampling thinning parameter |

---

## 🛠️ Installation & Usage

### Prerequisites

- Python 3.10+
- JAX with GPU support (recommended)
- [THRML](https://github.com/extropic-ai/thrml) ≥ 0.1.3

### Run on Google Colab (Recommended)

The easiest way to run the experiments is via Google Colab:

1. **Synthetic Data**: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brumoema/ising-conditional-routing/blob/main/experiments/synthetic_data.ipynb)

2. **Real Data**: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brumoema/ising-conditional-routing/blob/main/experiments/real_data.ipynb)

### Local Installation

```bash
# Clone the repository
git clone https://github.com/brumoema/ising-conditional-routing.git
cd ising-conditional-routing

# Create virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install jax jaxlib thrml matplotlib pandas ccxt

# Run Jupyter
jupyter notebook experiments/synthetic_data.ipynb
```

---

## 📊 Experiment Scenarios

### Synthetic Data

1. **IID Venues**: No correlation between venues (correlation_weight=0.0)
2. **Correlated Venues**: Positive correlations between venues (correlation_weight=0.4)
3. **Regime Shift**: Correlations exist, and venue biases change mid-experiment (step 5000)

### Context Modes

- **Fixed Context**: Always observe Venue 0's outcome as context
- **Random Context**: Randomly select which venue provides context each step

### Real Data

- **Source**: BTC/USD trades from Coinbase, Kraken, and Bitstamp
- **Window**: Rolling 10,000-second window of recent trades
- **Processing**: Time-bucketed with argmax-based winner labeling

---

## 📚 References

1. **THRML Documentation**: [https://docs.thrml.ai/](https://docs.thrml.ai/)
2. **THRML Repository**: [https://github.com/extropic-ai/thrml](https://github.com/extropic-ai/thrml)
3. **JAX**: [https://github.com/google/jax](https://github.com/google/jax)
4. Ising, E. (1925). "Beitrag zur Theorie des Ferromagnetismus." *Zeitschrift für Physik*, 31(1), 253-258.
5. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
6. Agrawal, S., & Goyal, N. (2013). "Thompson Sampling for Contextual Bandits with Linear Payoffs." *ICML*.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Bruna Moema**

---

## 🙏 Acknowledgments

- [Extropic AI](https://www.extropic.ai/) for developing THRML
- The JAX team at Google for the high-performance computing framework
