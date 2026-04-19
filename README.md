# transformer-pro

A clean, modular PyTorch implementation of the Transformer architecture.

Built from the ground up through [transformer-from-scratch](https://github.com/), then refactored into an importable package. All the math is the same — the structure is production-style.

## Install

```bash
pip install -r requirements.txt
```

No `setup.py` yet — run everything from the project root so `transformer/` is importable.

## Quickstart

```python
from transformer import Transformer, TransformerConfig

cfg   = TransformerConfig(vocab_size=1000, d_model=128, num_heads=4, max_seq_len=512)
model = Transformer(cfg, num_layers=2)

src    = torch.randint(0, cfg.vocab_size, (2, 10))  # (B, T_enc)
tgt    = torch.randint(0, cfg.vocab_size, (2, 8))   # (B, T_dec)
logits = model(src, tgt)                             # (B, T_dec, vocab_size)
```

## Package layout

```
transformer/
├── config.py      TransformerConfig dataclass — all hyperparameters in one place
├── embedding.py   InputEmbedding (token + positional), sinusoidal_pe()
├── attention.py   scaled_dot_product(), MultiHeadAttention, causal_mask()
├── layers.py      FeedForward, AddNorm
├── encoder.py     EncoderLayer, TransformerEncoder
├── decoder.py     DecoderLayer, TransformerDecoder
└── model.py       Transformer (encoder + decoder + logit head)
```

## API reference

### `TransformerConfig`

```python
TransformerConfig(
    vocab_size  = 1000,   # Vocabulary size
    d_model     = 8,      # Embedding / hidden dimension
    num_heads   = 2,      # Attention heads — must divide d_model evenly
    ff_dim      = None,   # FFN inner dim; defaults to d_model * 4
    max_seq_len = 100,    # Max sequence length for positional embeddings
    dropout     = 0.0,    # Placeholder — not wired in yet
)
```

`cfg.head_dim` is a computed property: `d_model // num_heads`.

---

### `InputEmbedding(cfg)`
Token embedding + learned positional embedding.
`forward(token_ids: (B, T)) → (B, T, d_model)`

### `sinusoidal_pe(max_seq_len, d_model) → Tensor`
Fixed sinusoidal encoding (not learned). Returns `(max_seq_len, d_model)`.

---

### `MultiHeadAttention(cfg)`
Multi-head self or cross attention with manual Q/K/V projections.
`forward(query, key, value, mask=None) → (output, weights)`
- `output`:  `(B, T_q, d_model)`
- `weights`: `(B, num_heads, T_q, T_kv)`

### `causal_mask(seq_len, device=None) → Tensor`
Upper-triangular bool mask for autoregressive decoding. `True` = masked.
Shape: `(seq_len, seq_len)`

---

### `FeedForward(cfg)`
Position-wise FFN: expand → ReLU → contract.
`forward(x: (B, T, d_model)) → (B, T, d_model)`

### `AddNorm(cfg)`
Residual connection + LayerNorm.
`forward(x, sublayer_output) → (B, T, d_model)`

---

### `EncoderLayer(cfg)`
Self-attention → Add&Norm → FFN → Add&Norm.
`forward(x: (B, T, d_model)) → (B, T, d_model)`

### `TransformerEncoder(cfg, num_layers=1)`
Input embedding + N stacked `EncoderLayer`s.
`forward(token_ids: (B, T)) → (B, T, d_model)`

---

### `DecoderLayer(cfg)`
Masked self-attention → Add&Norm → cross-attention → Add&Norm → FFN → Add&Norm.
`forward(decoder_x: (B, T_dec, d_model), encoder_output: (B, T_enc, d_model)) → (B, T_dec, d_model)`

### `TransformerDecoder(cfg, num_layers=1)`
Target embedding + N stacked `DecoderLayer`s.
`forward(tgt_token_ids: (B, T_dec), encoder_output: (B, T_enc, d_model)) → (B, T_dec, d_model)`

---

### `Transformer(cfg, num_layers=1)`
Full encoder-decoder with logit projection head.
`forward(src_ids: (B, T_enc), tgt_ids: (B, T_dec)) → logits (B, T_dec, vocab_size)`

## Tests

```bash
pytest tests/
```

Shape contracts for every module are in `tests/test_shapes.py`. All assertions reference the input → output shapes from the original experimental phases.

## Architecture decisions

**Manual Q/K/V projections in `MultiHeadAttention`** — `nn.MultiheadAttention` is used in phases 7–8 for convenience, but the pro package implements projections explicitly so the attention internals stay readable. `scaled_dot_product()` is separated from `MultiHeadAttention` so it can be tested and reused independently.

**`AddNorm` is a shared class** — both encoder and decoder use residual + LayerNorm. Extracting it avoids duplicating identical code in both files.

**`causal_mask` is generated inside `DecoderLayer.forward()`** — you never pass it manually. The mask shape matches `T_dec` at runtime so it works for any sequence length.

**`TransformerConfig` owns all hyperparameters** — no magic numbers inside modules. Scaling up means changing one dataclass, not hunting across files.

## Shape reference

See `utils/shapes.md`.

---

For the step-by-step derivation of every component, see [transformer-from-scratch](https://github.com/mutassimalzeem/transformer-from-scratch.git).
