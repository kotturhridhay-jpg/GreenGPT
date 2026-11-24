# ShaunGPT — Understanding Intelligence, One Token at a Time. 🤖

### A GPT-Style Language Model Built from Scratch (NumPy & PyTorch Variants)

ShaunGPT is an educational implementation of a Generative Pre-trained Transformer (GPT)-style language model. This project explores the core architecture and mathematical principles behind modern transformer-based language models across three progressive iterations:

1. **`ShaunPy.py`**: Built from scratch using **Pure Python and NumPy**, implementing manual forward passes, analytic backward passes, custom embeddings, multi-head attention, layer normalization, and an Adam optimizer.
2. **`ShaunTorch.py`**: Ported to **PyTorch**, leveraging autograd, optimized tensor operations, and a detailed technical reporting engine.
3. **`StarkGPT.py`**: Upgraded to a **modern transformer architecture** incorporating **RMSNorm**, **Rotary Position Embeddings (RoPE)**, **PyTorch's Causal SDPA (Scaled Dot-Product Attention)**, and the **SwiGLU feed-forward network**.

The models are trained on the Tiny Shakespeare dataset to generate character-level text autoregressively.

---

## 🎯 Project Objectives

- Explore the fundamental decoder-only Transformer architecture.
- Understand tensor operations, automatic differentiation, and manual gradient backpropagation (NumPy variant).
- Implement causal self-attention and multi-head attention mechanisms.
- Transition from educational NumPy building blocks to efficient PyTorch modules.
- Experiment with advanced architectural components like **RMSNorm**, **RoPE**, and **SwiGLU**.
- Analyze parameter statistics, memory footprints, and training dynamics via built-in technical reports.

---

## 🧠 Model Architecture Variants

| Component | NumPy Version (`ShaunPy.py`) | PyTorch Version (`ShaunTorch.py`) | Modern Version (`StarkGPT.py`) |
|---|---|---|---|
| **Framework** | Pure Python & NumPy | PyTorch (`torch.float64`) | PyTorch (`torch.float64`) |
| **Execution Device** | CPU | CPU | CPU |
| **Normalization** | Standard LayerNorm (Manual) | `nn.LayerNorm` | **RMSNorm** (Root Mean Square) |
| **Position Encoding** | Sinusoidal (Additive) | Sinusoidal (Additive) | **RoPE** (Rotary on $Q$ and $K$) |
| **Attention Mechanism** | Manual Causal Multi-Head | Manual Multi-Head | **Causal SDPA** (`F.scaled_dot_product_attention`) |
| **Feed-Forward Network**| Linear $\rightarrow$ ReLU $\rightarrow$ Linear | Linear $\rightarrow$ ReLU $\rightarrow$ Linear | **SwiGLU** ($W_3(\text{SiLU}(W_1 x) \odot W_2 x)$) |
| **Transformer Blocks** | 4 | 4 | 4 |
| **Embedding Dimension**| 320 | 320 | 320 |
| **Attention Heads** | 5 (Head Dim: 64) | 5 (Head Dim: 64) | 5 (Head Dim: 64) |
| **Context Length** | 256 | 256 | 256 |
| **Optimizer & Loss** | Adam + Cross-Entropy | Adam (`weight_decay=0.01`) + Cross-Entropy | Adam (`weight_decay=0.01`) + Cross-Entropy |

---

## ⚙️ Core Components & Features

- **Character-Level Tokenization:** Dynamically builds a vocabulary from unique characters in the dataset.
- **Rotary Position Embeddings (RoPE):** Encodes absolute and relative positions by rotating query and key vectors in complex planes (in `StarkGPT.py`).
- **RMSNorm:** Normalizes activations using root-mean-square scaling without mean-centering or learnable biases, improving training stability and efficiency.
- **Causal Self-Attention & SDPA:** Prevents information leakage from future tokens using causal masking or PyTorch's optimized scaled dot-product attention.
- **SwiGLU Feed-Forward Network:** Replaces standard ReLU layers with a Gated Linear Unit variant using the SiLU activation function.
- **Autoregressive Generation:** Generates text character-by-character using temperature scaling and top-$k$ sampling.
- **Technical Reporting Engine:** Automatically prints out model identification, architecture breakdown, parameter counts, memory usage, dataset splits, training configuration, and performance metrics upon completion.

---

## 🏗️ Architecture Overview (`StarkGPT.py`)

```text
                       Input Text
                           |
                           v
                     Character IDs
                           |
                           v
                    Token Embeddings
                           |
                           v
              +----------------------------+
              |    Transformer Block 1     |
              |                            |
              |    RMSNorm                 |
              |    Q/K/V Projections       |
              |    Rotary Position (RoPE)  |
              |    Causal SDPA Attention   |
              |    Residual Connection     |
              |    RMSNorm                 |
              |    SwiGLU Feed-Forward     |
              |    Residual Connection     |
              +----------------------------+
                           |
                           v
              +----------------------------+
              |    Transformer Block 2-4   |
              +----------------------------+
                           |
                           v
                     Final RMSNorm
                           |
                           v
                  Output Linear Projection
                           |
                           v
                  Character Probabilities
                           |
                           v
                 Next-Character Prediction
                           |
                           v
                   Autoregressive Loop
                           |
                           v
                    Generated Text
```

---

## 📚 Dataset & Training Configuration

ShaunGPT uses the **Tiny Shakespeare** corpus (`input.txt`), processed into contiguous sequence chunks.

- **Dataset Split:** 90% Training / 10% Validation
- **Default Training Steps:** 3,000 steps
- **Batch Size:** 32 sequences
- **Context Length:** 256 tokens
- **Learning Rate:** $3 \times 10^{-4}$
- **Dropout Rate:** 0.20
- **Random Seed:** 42 (for complete reproducibility)

---

## 🚀 Usage

Make sure you have `input.txt` (Tiny Shakespeare dataset) in the same directory as your scripts, then run any of the implementations:

```bash
# Run the NumPy educational implementation
python ShaunPy.py

# Run the PyTorch port
python ShaunTorch.py

# Run the modern architecture variant (RMSNorm + RoPE + SwiGLU)
python StarkGPT.py
```

You can also pass custom CLI arguments:
```bash
python StarkGPT.py --steps 5000 --batch-size 64 --seq-len 256
```

---

**Understanding Intelligence, One Token at a Time.** 🤖