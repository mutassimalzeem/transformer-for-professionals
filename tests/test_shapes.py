"""Shape regression tests — every module, every shape contract.

Run from the project root:
    pytest tests/

All assertions mirror the shape tracking from the experimental phases,
so failures are immediately traceable back to the phase that caught
the same shape first.
"""

import pytest
import torch

from transformer import (
    TransformerConfig,
    InputEmbedding,
    sinusoidal_pe,
    MultiHeadAttention,
    causal_mask,
    FeedForward,
    AddNorm,
    EncoderLayer,
    TransformerEncoder,
    DecoderLayer,
    TransformerDecoder,
    Transformer,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return TransformerConfig(vocab_size=1000, d_model=8, num_heads=2, max_seq_len=100)

@pytest.fixture
def B():   return 2
@pytest.fixture
def T():   return 6
@pytest.fixture
def T_enc(): return 7
@pytest.fixture
def T_dec(): return 5


# ── Config ───────────────────────────────────────────────────────────────────

def test_config_head_dim():
    cfg = TransformerConfig(d_model=8, num_heads=2)
    assert cfg.head_dim == 4

def test_config_ff_dim_default():
    cfg = TransformerConfig(d_model=8)
    assert cfg.ff_dim == 32   # d_model * 4

def test_config_invalid_heads():
    with pytest.raises(AssertionError):
        TransformerConfig(d_model=8, num_heads=3)   # 8 % 3 != 0


# ── Positional encoding ──────────────────────────────────────────────────────

def test_sinusoidal_pe_shape(cfg):
    pe = sinusoidal_pe(cfg.max_seq_len, cfg.d_model)
    assert pe.shape == (cfg.max_seq_len, cfg.d_model)   # Phase 2: (T, d_model)

def test_sinusoidal_pe_deterministic(cfg):
    pe1 = sinusoidal_pe(cfg.max_seq_len, cfg.d_model)
    pe2 = sinusoidal_pe(cfg.max_seq_len, cfg.d_model)
    assert torch.allclose(pe1, pe2)   # Not learned — same result every call


# ── Input embedding ──────────────────────────────────────────────────────────

def test_input_embedding_shape(cfg, B, T):
    model = InputEmbedding(cfg)
    ids   = torch.randint(0, cfg.vocab_size, (B, T))
    out   = model(ids)
    assert out.shape == (B, T, cfg.d_model)   # Phase 1–2: (B, T, d_model)


# ── Attention ────────────────────────────────────────────────────────────────

def test_self_attention_output_shape(cfg, B, T):
    mha    = MultiHeadAttention(cfg)
    x      = torch.randn(B, T, cfg.d_model)
    out, w = mha(x, x, x)
    assert out.shape == (B, T, cfg.d_model)            # Phase 5: (B, T, d_model)
    assert w.shape   == (B, cfg.num_heads, T, T)        # (B, h, T, T)

def test_cross_attention_output_shape(cfg, B, T_enc, T_dec):
    mha       = MultiHeadAttention(cfg)
    dec_x     = torch.randn(B, T_dec, cfg.d_model)
    enc_out   = torch.randn(B, T_enc, cfg.d_model)
    out, w    = mha(dec_x, enc_out, enc_out)
    assert out.shape == (B, T_dec, cfg.d_model)             # Phase 8, task 2
    assert w.shape   == (B, cfg.num_heads, T_dec, T_enc)

def test_causal_mask_shape(T):
    mask = causal_mask(T)
    assert mask.shape == (T, T)

def test_causal_mask_upper_triangular(T):
    mask = causal_mask(T)
    # Position 0 should attend to nothing future → mask[0, 1:] all True
    assert mask[0, 0].item() == False   # Can attend to itself
    assert mask[0, 1].item() == True    # Cannot attend to future
    assert mask[T-1, T-1].item() == False   # Last token sees itself


# ── Layers ───────────────────────────────────────────────────────────────────

def test_feedforward_shape(cfg, B, T):
    ffn = FeedForward(cfg)
    x   = torch.randn(B, T, cfg.d_model)
    out = ffn(x)
    assert out.shape == (B, T, cfg.d_model)   # Phase 6: (B, T, d_model)

def test_addnorm_shape(cfg, B, T):
    addnorm = AddNorm(cfg)
    x       = torch.randn(B, T, cfg.d_model)
    sub_out = torch.randn(B, T, cfg.d_model)
    out     = addnorm(x, sub_out)
    assert out.shape == (B, T, cfg.d_model)


# ── Encoder ──────────────────────────────────────────────────────────────────

def test_encoder_layer_shape(cfg, B, T):
    layer = EncoderLayer(cfg)
    x     = torch.randn(B, T, cfg.d_model)
    out   = layer(x)
    assert out.shape == (B, T, cfg.d_model)

def test_transformer_encoder_shape(cfg, B, T):
    encoder = TransformerEncoder(cfg, num_layers=1)
    ids     = torch.randint(0, cfg.vocab_size, (B, T))
    out     = encoder(ids)
    assert out.shape == (B, T, cfg.d_model)   # Phase 7: (B, T, d_model)

@pytest.mark.parametrize("num_layers", [1, 2, 4])
def test_encoder_stacked_layers(cfg, B, T, num_layers):
    encoder = TransformerEncoder(cfg, num_layers=num_layers)
    ids     = torch.randint(0, cfg.vocab_size, (B, T))
    out     = encoder(ids)
    assert out.shape == (B, T, cfg.d_model)   # Shape unchanged regardless of depth


# ── Decoder ──────────────────────────────────────────────────────────────────

def test_decoder_layer_shape(cfg, B, T_enc, T_dec):
    layer      = DecoderLayer(cfg)
    dec_x      = torch.randn(B, T_dec, cfg.d_model)
    enc_out    = torch.randn(B, T_enc, cfg.d_model)
    out        = layer(dec_x, enc_out)
    assert out.shape == (B, T_dec, cfg.d_model)   # Phase 8: (B, T_dec, d_model)

def test_transformer_decoder_shape(cfg, B, T_enc, T_dec):
    decoder = TransformerDecoder(cfg, num_layers=1)
    enc_out = torch.randn(B, T_enc, cfg.d_model)
    tgt_ids = torch.randint(0, cfg.vocab_size, (B, T_dec))
    out     = decoder(tgt_ids, enc_out)
    assert out.shape == (B, T_dec, cfg.d_model)


# ── Full model ───────────────────────────────────────────────────────────────

def test_transformer_output_shape(cfg, B, T_enc, T_dec):
    model  = Transformer(cfg, num_layers=1)
    src    = torch.randint(0, cfg.vocab_size, (B, T_enc))
    tgt    = torch.randint(0, cfg.vocab_size, (B, T_dec))
    logits = model(src, tgt)
    assert logits.shape == (B, T_dec, cfg.vocab_size)   # (B, T_dec, vocab_size)

def test_transformer_forward_no_crash(cfg):
    """Smoke test — just make sure nothing errors end-to-end."""
    model  = Transformer(cfg, num_layers=2)
    src    = torch.randint(0, cfg.vocab_size, (1, 10))
    tgt    = torch.randint(0, cfg.vocab_size, (1, 8))
    logits = model(src, tgt)
    assert logits is not None
