"""End-to-end forward pass tests for encoder, decoder, and full model.

Run: pytest tests/test_forward.py -v
"""

import torch
import torch.nn as nn

from transformer import (
    EncoderDecoderTransformer,
    TransformerConfig,
    TransformerDecoder,
    TransformerEncoder,
)

CFG = TransformerConfig(vocab_size=50, d_model=8, num_heads=2, ff_dim=32, max_seq_len=20)


def test_encoder_forward():
    encoder = TransformerEncoder(CFG, n_layers=2)
    src     = torch.randint(0, 50, (2, 7))
    out     = encoder(src)
    assert out.shape == (2, 7, 8)


def test_decoder_forward():
    encoder = TransformerEncoder(CFG, n_layers=2)
    decoder = TransformerDecoder(CFG, n_layers=2)
    src     = torch.randint(0, 50, (2, 7))
    tgt     = torch.randint(0, 50, (2, 5))
    enc_out = encoder(src)
    dec_out = decoder(tgt, enc_out)
    assert dec_out.shape == (2, 5, 8)


def test_full_model_forward():
    model  = EncoderDecoderTransformer(CFG, n_layers=2)
    src    = torch.randint(0, 50, (2, 7))
    tgt    = torch.randint(0, 50, (2, 5))
    logits = model(src, tgt)
    assert logits.shape == (2, 5, 50)    # (B, T_dec, vocab_size)


def test_logits_are_finite():
    """No NaNs or Infs from a freshly initialized model."""
    model  = EncoderDecoderTransformer(CFG, n_layers=2)
    src    = torch.randint(0, 50, (2, 7))
    tgt    = torch.randint(0, 50, (2, 5))
    logits = model(src, tgt)
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"


def test_backward_pass():
    """Gradients flow to all parameters — catches any broken detach or missing connection."""
    model     = EncoderDecoderTransformer(CFG, n_layers=2)
    src       = torch.randint(0, 50, (2, 7))
    tgt       = torch.randint(0, 50, (2, 5))
    logits    = model(src, tgt)                          # (2, 5, 50)

    # Cross-entropy loss over the sequence
    loss      = nn.CrossEntropyLoss()(
        logits.reshape(-1, CFG.vocab_size),
        tgt.reshape(-1)
    )
    loss.backward()

    no_grad = [
        name for name, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not no_grad, f"Parameters with no gradient: {no_grad}"
