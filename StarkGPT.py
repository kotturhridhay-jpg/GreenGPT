from pathlib import Path
import argparse
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================= Configuration =========================

SEED = 42
MAX_CHARS = None
DEFAULT_STEPS = 3000
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEQ_LEN = 256
DEFAULT_DIM_MODEL = 320
DEFAULT_N_HEADS = 5
DEFAULT_BLOCKS = 4
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_SAMPLE_EVERY = 1000
DEFAULT_GENERATE_TOKENS = 150
DEFAULT_DROPOUT_RATE = 0.20
DEFAULT_EVAL_EVERY = 250
DEFAULT_EVAL_BATCHES = 10
DEFAULT_VAL_FRACTION = 0.10

# Preserve the execution settings of your uploaded ShaunTorch file.
DEVICE = torch.device("cpu")
MODEL_DTYPE = torch.float64


# ========================= Basic building blocks =========================

class LinearLEGO(nn.Module):
    """Linear layer with the small-normal initialization used by ShaunGPT."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


class EmbeddingsLEGO(nn.Module):
    """Maps integer token IDs to learned token vectors."""

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings(token_ids.long())


class DropoutLEGO(nn.Module):
    """Dropout that is active only while the model is training."""

    def __init__(self, dropout_rate: float):
        super().__init__()
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in the range [0, 1).")
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(inputs)


# ========================= RMS normalization =========================

class RMSNormLEGO(nn.Module):
    """Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm does not subtract the mean and has no learned
    bias. It normalizes by the root-mean-square value of the final dimension
    and applies one learned scale parameter per feature.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        if dim <= 0:
            raise ValueError("RMSNorm dimension must be positive.")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Compute the mean square in the input dtype, preserving float64 mode.
        mean_square = inputs.pow(2).mean(dim=-1, keepdim=True)
        normalized = inputs * torch.rsqrt(mean_square + self.eps)
        return normalized * self.weight


# ========================= Rotary position embeddings =========================

class RotaryPositionEmbeddingLEGO(nn.Module):
    """Applies RoPE to query and key tensors shaped (B, H, T, head_dim).

    Rotary embeddings encode token position by rotating pairs of dimensions
    in Q and K. Values are not rotated. This replaces the old additive
    sinusoidal positional-encoding layer.
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError(
                "RoPE requires a positive, even attention head dimension.")

        inverse_frequencies = 1.0 / (
            base ** (torch.arange(0, head_dim, 2,
                     dtype=torch.float64) / head_dim)
        )
        self.register_buffer("inverse_frequencies", inverse_frequencies)
        self.head_dim = head_dim

    def _rotate(self, tensor: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor) -> torch.Tensor:
        even_features = tensor[..., 0::2]
        odd_features = tensor[..., 1::2]

        rotated_even = even_features * cos - odd_features * sin
        rotated_odd = even_features * sin + odd_features * cos

        # Interleave each rotated even/odd pair back into the head dimension.
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

    def forward(self, queries: torch.Tensor,
                keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = queries.shape[-2]
        positions = torch.arange(
            seq_len,
            device=queries.device,
            dtype=self.inverse_frequencies.dtype,
        )
        angles = torch.outer(positions, self.inverse_frequencies)

        # Shapes become (1, 1, sequence_length, head_dim / 2).
        cos = angles.cos().to(dtype=queries.dtype)[None, None, :, :]
        sin = angles.sin().to(dtype=queries.dtype)[None, None, :, :]

        return self._rotate(queries, cos, sin), self._rotate(keys, cos, sin)


# ========================= Causal multi-head SDPA attention =========================

class MultiHeadAttentionLEGO(nn.Module):
    """Multi-head causal self-attention with Q/K RoPE and output projection."""

    def __init__(self, dim_model: int, n_heads: int, dropout_rate: float):
        super().__init__()
        if n_heads <= 0 or dim_model % n_heads != 0:
            raise ValueError("dim_model must be divisible by n_heads.")

        self.dim_model = dim_model
        self.n_heads = n_heads
        self.head_dim = dim_model // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                "The head dimension must be even for rotary position embeddings."
            )

        # The diagram's W_Q, W_K and W_V projections.
        self.w_q = LinearLEGO(dim_model, dim_model, bias=False)
        self.w_k = LinearLEGO(dim_model, dim_model, bias=False)
        self.w_v = LinearLEGO(dim_model, dim_model, bias=False)

        self.rope = RotaryPositionEmbeddingLEGO(self.head_dim)

        # The diagram's W_O projection after causal attention.
        self.w_o = LinearLEGO(dim_model, dim_model, bias=False)
        self.attention_dropout = dropout_rate
        self.output_dropout = DropoutLEGO(dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = inputs.shape

        # Project the normalized input into Q, K and V.
        queries = self.w_q(inputs)
        keys = self.w_k(inputs)
        values = self.w_v(inputs)

        # Split model dimension into independent attention heads.
        queries = queries.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)
        keys = keys.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)
        values = values.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)

        # Apply rotary position embeddings to Q and K only.
        queries, keys = self.rope(queries, keys)

        # This is PyTorch's fused-SDPA entry point. On supported CUDA hardware
        # and dtypes it may dispatch to FlashAttention; on this CPU setup it
        # safely dispatches to the CPU math implementation.
        dropout_p = self.attention_dropout if self.training else 0.0
        if hasattr(F, "scaled_dot_product_attention"):
            attended = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
            )
        else:
            # Compatibility fallback for older PyTorch releases.
            scores = (queries @ keys.transpose(-2, -1)) / \
                math.sqrt(self.head_dim)
            future_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=inputs.device,
                           dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(future_mask, float("-inf"))
            attention_weights = torch.softmax(scores, dim=-1)
            attention_weights = F.dropout(
                attention_weights, p=dropout_p, training=self.training
            )
            attended = attention_weights @ values

        # Merge heads back into the model dimension.
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.dim_model
        )

        # Apply W_O and dropout before the attention residual addition.
        return self.output_dropout(self.w_o(attended))


# ========================= SwiGLU feed-forward network =========================

class SwiGLULEGO(nn.Module):
    """The diagram's W3(SiLU(W1 x) * W2 x) feed-forward sublayer."""

    def __init__(self, dim_model: int, hidden_dim: int, dropout_rate: float):
        super().__init__()
        self.w1 = LinearLEGO(dim_model, hidden_dim, bias=False)
        self.w2 = LinearLEGO(dim_model, hidden_dim, bias=False)
        self.w3 = LinearLEGO(hidden_dim, dim_model, bias=False)
        self.dropout = DropoutLEGO(dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Two parallel projections of the same normalized input.
        gate = F.silu(self.w1(inputs))
        value = self.w2(inputs)

        # Elementwise gated product, followed by the W3 output projection.
        hidden = gate * value
        return self.dropout(self.w3(hidden))


# ========================= Transformer decoder block =========================

class TransformerBlockLEGO(nn.Module):
    """Pre-normalized block with two residual additions.

    1. x = x + Attention(RMSNorm(x))
    2. x = x + SwiGLU(RMSNorm(x))
    """

    def __init__(self, dim_model: int, n_heads: int,
                 dropout_rate: float, ffn_multiplier: int = 4):
        super().__init__()
        hidden_dim = ffn_multiplier * dim_model

        self.attention_norm = RMSNormLEGO(dim_model)
        self.attention = MultiHeadAttentionLEGO(
            dim_model, n_heads, dropout_rate
        )

        self.ffn_norm = RMSNormLEGO(dim_model)
        self.swiglu = SwiGLULEGO(
            dim_model, hidden_dim, dropout_rate
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # First residual path: input + attention output.
        inputs = inputs + self.attention(self.attention_norm(inputs))

        # Second residual path: attention result + SwiGLU output.
        inputs = inputs + self.swiglu(self.ffn_norm(inputs))
        return inputs


# ========================= Cross-entropy loss =========================

class CrossEntropyWithSoftmaxLEGO(nn.Module):
    """Next-token cross entropy. F.cross_entropy includes the log-softmax."""

    def forward(self, logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3:
            raise ValueError(
                "logits must have shape (batch, time, vocabulary).")

        batch_size, seq_len, vocab_size = logits.shape
        if labels.numel() != batch_size * seq_len:
            raise ValueError(
                "labels do not match logits batch/time dimensions.")

        labels = labels.long().reshape(-1)
        if torch.any(labels < 0) or torch.any(labels >= vocab_size):
            raise ValueError("A label index is outside the vocabulary.")

        return F.cross_entropy(logits.reshape(-1, vocab_size), labels)


# ========================= Complete GPT model =========================

class GPTLEGO(nn.Module):
    """Character-level GPT using the supplied RMSNorm/RoPE/SwiGLU diagram."""

    def __init__(self, vocab_size: int, blocks: int, dim_model: int,
                 n_heads: int, dropout_rate: float = DEFAULT_DROPOUT_RATE,
                 max_seq_len: int = 5000):
        super().__init__()
        if vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1.")
        if blocks <= 0:
            raise ValueError("blocks must be positive.")
        if dim_model <= 0 or n_heads <= 0 or dim_model % n_heads != 0:
            raise ValueError(
                "dim_model must be positive and divisible by n_heads.")
        if (dim_model // n_heads) % 2 != 0:
            raise ValueError("dim_model // n_heads must be even for RoPE.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")

        self.vocab_size = vocab_size
        self.dim_model = dim_model
        self.n_heads = n_heads
        self.n_blocks = blocks
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate

        # The diagram's input token embedding. There is no additive positional
        # encoding here; RoPE injects position into Q and K inside attention.
        self.embedding_decoder = EmbeddingsLEGO(vocab_size, dim_model)

        self.decoder_blocks = nn.ModuleList([
            TransformerBlockLEGO(
                dim_model=dim_model,
                n_heads=n_heads,
                dropout_rate=dropout_rate,
            )
            for _ in range(blocks)
        ])

        # The diagram's final RMSNorm -> Linear -> Output Logits sequence.
        self.final_rmsnorm = RMSNormLEGO(dim_model)
        self.output_linear = LinearLEGO(dim_model, vocab_size, bias=False)
        self.loss_function = CrossEntropyWithSoftmaxLEGO()

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        if input_tokens.ndim != 2:
            raise ValueError("input_tokens must have shape (batch, sequence).")
        if input_tokens.shape[1] > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {input_tokens.shape[1]} exceeds "
                f"max_seq_len={self.max_seq_len}."
            )

        x = self.embedding_decoder(input_tokens)
        for block in self.decoder_blocks:
            x = block(x)

        x = self.final_rmsnorm(x)
        return self.output_linear(x)

    @torch.no_grad()
    def generate(self, start_tokens: list[int], max_new_tokens: int,
                 seq_len: int, temperature: float = 1.0,
                 top_k: int | None = None) -> list[int]:
        """Autoregressively generate character IDs from a starting prompt."""
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")
        if seq_len <= 0 or max_new_tokens < 0:
            raise ValueError(
                "seq_len must be positive and max_new_tokens non-negative.")
        if not start_tokens:
            raise ValueError("start_tokens must contain at least one token.")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided.")

        generated = list(map(int, start_tokens))
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device

        try:
            for _ in range(max_new_tokens):
                context = generated[-seq_len:]
                input_tensor = torch.tensor(
                    [context], dtype=torch.long, device=device
                )

                logits = self(input_tensor)
                next_logits = logits[0, -1, :] / temperature

                if top_k is not None:
                    k = min(top_k, next_logits.numel())
                    keep_indices = torch.topk(next_logits, k).indices
                    filtered_logits = torch.full_like(
                        next_logits, float("-inf"))
                    filtered_logits[keep_indices] = next_logits[keep_indices]
                    next_logits = filtered_logits

                probabilities = torch.softmax(next_logits, dim=-1)
                next_token = int(torch.multinomial(
                    probabilities, num_samples=1).item())
                generated.append(next_token)
        finally:
            self.train(was_training)

        return generated


# ========================= Dataset loading =========================

def load_shakespeare(data_path: str | Path | None = None,
                     max_chars: int | None = MAX_CHARS):
    """Load local UTF-8 text and build a character-level vocabulary."""
    if data_path is None:
        data_path = Path(__file__).resolve().with_name("input.txt")

    data_path = Path(data_path).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Could not find dataset: {data_path}\n"
            "Place input.txt beside this script or pass --data PATH."
        )

    text = data_path.read_text(encoding="utf-8")
    if not text:
        raise ValueError(f"Dataset is empty: {data_path}")
    if max_chars is not None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive or None.")
        text = text[:max_chars]

    chars = sorted(set(text))
    char_to_int = {character: index for index, character in enumerate(chars)}
    int_to_char = {index: character for character,
                   index in char_to_int.items()}
    encoded = torch.tensor([char_to_int[character]
                           for character in text], dtype=torch.long)

    if len(encoded) < 2:
        raise ValueError("Dataset needs at least two characters.")

    return encoded, len(chars), char_to_int, int_to_char, text


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int):
    """Create random input sequences and their one-character-shifted targets."""
    if len(data) <= seq_len:
        raise ValueError(
            f"Dataset split must be longer than seq_len={seq_len}.")

    offsets = torch.randint(0, len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in offsets.tolist()])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in offsets.tolist()])
    return x, y


# ========================= Evaluation =========================

def estimate_loss(model: GPTLEGO, train_data: torch.Tensor,
                  val_data: torch.Tensor, batch_size: int,
                  seq_len: int, eval_batches: int = 10) -> dict[str, float]:
    """Estimate mean training and validation loss without updating weights."""
    results = {}
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for split_name, split_data in (("train", train_data), ("val", val_data)):
                losses = []
                for _ in range(eval_batches):
                    x_batch, y_batch = get_batch(
                        split_data, batch_size, seq_len)
                    logits = model(x_batch.to(DEVICE))
                    losses.append(model.loss_function(
                        logits, y_batch.to(DEVICE)))
                results[split_name] = torch.stack(losses).mean().item()
    finally:
        model.train(was_training)

    return results


# ========================= Technical report =========================

def print_technical_report(model: GPTLEGO, args, data: torch.Tensor,
                           train_data: torch.Tensor, val_data: torch.Tensor,
                           vocab_size: int, initial_losses: dict,
                           final_losses: dict, elapsed_time: float,
                           data_path: Path):
    parameters = list(model.parameters())
    total_parameters = sum(parameter.numel() for parameter in parameters)
    trainable_parameters = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    parameter_memory = sum(
        parameter.numel() * parameter.element_size() for parameter in parameters
    )
    buffer_memory = sum(
        buffer.numel() * buffer.element_size() for buffer in model.buffers()
    )

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    head_dim = args.dim_model // args.n_heads
    ffn_dim = 4 * args.dim_model

    print("\n" + "=" * 72)
    print("                              SHAUNGPT")
    print("                 RMSNorm + RoPE + SwiGLU Architecture")
    print("=" * 72)

    print("\n---------- 01. MODEL IDENTIFICATION ----------")
    print("Model Name             : ShaunGPT")
    print("Architecture           : Decoder-Only Transformer")
    print("Model Type             : Character-Level Language Model")
    print(f"Framework              : PyTorch {torch.__version__}")
    print(f"Execution Device       : {str(device).upper()}")
    print(f"Numerical Precision    : {dtype}")

    print("\n---------- 02. MODEL ARCHITECTURE ----------")
    print(f"Transformer Blocks     : {args.blocks}")
    print(f"Embedding Dimension    : {args.dim_model}")
    print(f"Attention Heads        : {args.n_heads}")
    print(f"Head Dimension         : {head_dim}")
    print(f"Context Length         : {args.seq_len}")
    print(f"Vocabulary Size        : {vocab_size}")
    print(f"SwiGLU Hidden Dimension: {ffn_dim}")
    print("Normalization          : RMSNorm")
    print("Position Method        : RoPE on Queries and Keys")
    print("Attention              : Causal Multi-Head SDPA")
    print("Attention API          : torch.nn.functional.scaled_dot_product_attention")
    print("Attention Backend      : PyTorch-selected backend (CPU math on this run)")
    print("Feed-Forward Network   : SwiGLU")
    print("SwiGLU Activation      : SiLU(x) multiplied by parallel linear gate")
    print("Residual Connections   : Two per decoder block")
    print("Output Head            : RMSNorm -> Linear -> Logits")

    print("\n---------- 03. MODEL PARAMETERS ----------")
    print(f"Total Parameters       : {total_parameters:,}")
    print(f"Trainable Parameters   : {trainable_parameters:,}")
    print(
        f"Non-Trainable Params   : {total_parameters - trainable_parameters:,}")
    print(f"Parameter Memory       : {parameter_memory / (1024 ** 2):.2f} MiB")
    print(f"Buffer Memory          : {buffer_memory / (1024 ** 2):.2f} MiB")

    print("\n---------- 04. DATASET INFORMATION ----------")
    print("Dataset                : Tiny Shakespeare / supplied text file")
    print(f"Dataset Source         : {data_path}")
    print(f"Characters Loaded      : {len(data):,}")
    print(f"Unique Characters      : {vocab_size}")
    print(f"Training Characters    : {len(train_data):,}")
    print(f"Validation Characters  : {len(val_data):,}")
    print(f"Training Split         : {1.0 - args.val_fraction:.0%}")
    print(f"Validation Split       : {args.val_fraction:.0%}")

    print("\n---------- 05. TRAINING CONFIGURATION ----------")
    print(f"Training Steps         : {args.steps:,}")
    print(f"Batch Size             : {args.batch_size}")
    print(f"Sequence Length        : {args.seq_len}")
    print(f"Learning Rate          : {args.learning_rate}")
    print("Optimizer              : Adam")
    print("Loss Function          : Cross Entropy")
    print(f"Dropout Rate           : {args.dropout:.2f}")
    print("Gradient Clipping      : 1.0")
    print(f"Random Seed            : {SEED}")
    print(f"Evaluation Interval    : {args.eval_every}")
    print(f"Evaluation Batches     : {args.eval_batches}")

    print("\n---------- 06. TRAINING PERFORMANCE ----------")
    print(f"Initial Train Loss     : {initial_losses['train']:.4f}")
    print(f"Initial Validation Loss: {initial_losses['val']:.4f}")
    print(f"Final Train Loss       : {final_losses['train']:.4f}")
    print(f"Final Validation Loss  : {final_losses['val']:.4f}")
    print(f"Training Duration      : {elapsed_time:.2f} seconds")

    print("\n---------- 07. COMPUTATIONAL DETAILS ----------")
    print("Execution Backend      : PyTorch")
    print(f"Device                 : {device}")
    print(f"Parameter Data Type    : {dtype}")
    print("GPU Acceleration       : Disabled (CPU configuration)")
    print("Flash Kernel Note      : SDPA uses the CPU math backend on this machine.")

    print("\n" + "=" * 72)
    print("                       END OF TECHNICAL REPORT")
    print("=" * 72 + "\n")


# ========================= Main training program =========================

def main():
    parser = argparse.ArgumentParser(
        description="Train ShaunGPT using RMSNorm, RoPE, causal SDPA and SwiGLU."
    )
    parser.add_argument("--data", type=Path, default=None,
                        help="Path to input.txt (default: beside this script).")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--sample-every", type=int, default=DEFAULT_SAMPLE_EVERY,
                        help="Generate text every N steps; 0 disables samples.")
    parser.add_argument("--max-chars", type=int, default=0,
                        help="Maximum characters; 0 loads the complete file.")
    parser.add_argument("--val-fraction", type=float,
                        default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--eval-batches", type=int,
                        default=DEFAULT_EVAL_BATCHES)
    parser.add_argument("--dim-model", type=int, default=DEFAULT_DIM_MODEL)
    parser.add_argument("--n-heads", type=int, default=DEFAULT_N_HEADS)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--learning-rate", type=float,
                        default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT_RATE)
    args = parser.parse_args()

    run_start_time = time.perf_counter()

    if args.steps <= 0 or args.batch_size <= 0 or args.seq_len <= 0:
        parser.error("--steps, --batch-size and --seq-len must be positive.")
    if args.sample_every < 0:
        parser.error("--sample-every cannot be negative.")
    if args.eval_every <= 0 or args.eval_batches <= 0:
        parser.error("--eval-every and --eval-batches must be positive.")
    if not 0.0 < args.val_fraction < 0.5:
        parser.error(
            "--val-fraction must be greater than 0 and less than 0.5.")
    if args.dim_model <= 0 or args.n_heads <= 0 or args.dim_model % args.n_heads != 0:
        parser.error(
            "--dim-model must be positive and divisible by --n-heads.")
    if (args.dim_model // args.n_heads) % 2 != 0:
        parser.error(
            "--dim-model / --n-heads must produce an even head dimension for RoPE.")
    if args.blocks <= 0 or args.learning_rate <= 0:
        parser.error("--blocks and --learning-rate must be positive.")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in the range [0, 1).")

    torch.manual_seed(SEED)
    max_chars = None if args.max_chars == 0 else args.max_chars
    data, vocab_size, char_to_int, int_to_char, _ = load_shakespeare(
        args.data, max_chars
    )

    split_index = int(len(data) * (1.0 - args.val_fraction))
    train_data = data[:split_index]
    val_data = data[split_index:]
    if len(train_data) <= args.seq_len or len(val_data) <= args.seq_len:
        parser.error(
            "Both training and validation splits must be longer than --seq-len.")

    data_path = (
        Path(args.data).expanduser().resolve()
        if args.data
        else Path(__file__).resolve().with_name("input.txt")
    )

    print(
        f"Dataset: {len(data):,} characters | vocabulary: {vocab_size} unique characters")
    print(f"Source : {data_path}")
    print(
        f"Split  : {len(train_data):,} train ({1 - args.val_fraction:.0%}) | "
        f"{len(val_data):,} validation ({args.val_fraction:.0%})"
    )
    print(
        "Building ShaunGPT: "
        f"{args.blocks} block(s), d_model={args.dim_model}, "
        f"heads={args.n_heads}, head_dim={args.dim_model // args.n_heads}, "
        f"context={args.seq_len}"
    )
    print("Architecture: Embedding -> [RMSNorm -> RoPE/SDPA -> Residual -> "
          "RMSNorm -> SwiGLU -> Residual] x blocks -> RMSNorm -> Linear\n")

    model = GPTLEGO(
        vocab_size=vocab_size,
        blocks=args.blocks,
        dim_model=args.dim_model,
        n_heads=args.n_heads,
        dropout_rate=args.dropout,
        max_seq_len=max(5000, args.seq_len),
    ).to(device=DEVICE, dtype=MODEL_DTYPE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )

    print(
        f"Training for {args.steps:,} steps on {DEVICE} ({MODEL_DTYPE})...\n")

    initial_losses = estimate_loss(
        model, train_data, val_data, args.batch_size,
        args.seq_len, args.eval_batches
    )
    final_losses = initial_losses.copy()

    for step in range(1, args.steps + 1):
        model.train()
        x_batch, y_batch = get_batch(train_data, args.batch_size, args.seq_len)
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        train_loss = model.loss_function(logits, y_batch)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % 50 == 0 or step == args.steps:
            print(
                f"Step [{step:04d}/{args.steps:04d}] | "
                f"training loss: {train_loss.item():.4f}"
            )

        should_evaluate = (
            step == 1 or step % args.eval_every == 0 or step == args.steps
        )
        if should_evaluate:
            final_losses = estimate_loss(
                model, train_data, val_data, args.batch_size,
                args.seq_len, args.eval_batches
            )
            print(
                f"  Evaluation | train loss: {final_losses['train']:.4f} | "
                f"validation loss: {final_losses['val']:.4f}"
            )

        if args.sample_every and (
            step % args.sample_every == 0 or step == args.steps
        ):
            prompt = "First Citizen:"
            if all(character in char_to_int for character in prompt):
                prompt_tokens = [char_to_int[character]
                                 for character in prompt]
            else:
                prompt_tokens = [int(data[0].item())]

            generated_tokens = model.generate(
                prompt_tokens,
                max_new_tokens=DEFAULT_GENERATE_TOKENS,
                seq_len=args.seq_len,
                temperature=0.8,
                top_k=min(20, vocab_size),
            )
            generated_text = "".join(
                int_to_char[token] for token in generated_tokens)
            print("\n--- ShaunGPT sample ---")
            print(generated_text)
            print("-----------------------\n")

    print("ShaunGPT training successfully finished!")
    elapsed_time = time.perf_counter() - run_start_time

    print_technical_report(
        model=model,
        args=args,
        data=data,
        train_data=train_data,
        val_data=val_data,
        vocab_size=vocab_size,
        initial_losses=initial_losses,
        final_losses=final_losses,
        elapsed_time=elapsed_time,
        data_path=data_path,
    )


if __name__ == "__main__":
    main()
