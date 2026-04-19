import math
import torch
import torch.nn as nn

from .config import TransformerConfig


def scaled_dot_product(q: torch.Tensor,
                        k: torch.Tensor,
                        v: torch.Tensor,
                        mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Core attention math, separated from the projection machinery.

    Args:
        q, k, v: (B, num_heads, T, head_dim)
        mask:    (T, T) bool — True positions are masked (set to -inf)

    Returns:
        output:  (B, num_heads, T, head_dim)
        weights: (B, num_heads, T, T)
    """
    head_dim = q.shape[-1]

    scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)     # (B, h, T, T)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)                     # (B, h, T, T)
    output  = weights @ v                                       # (B, h, T, head_dim)

    return output, weights


class MultiHeadAttention(nn.Module):
    """Multi-head self or cross attention.

    Implements phases 3–5 (similarity scores → softmax → weighted sum → concat → project)
    as a single reusable module. Uses manual Q/K/V projections so the internals
    stay visible — wrapping nn.MultiheadAttention would hide too much.

    Shape contract:
        q:      (B, T_q,   d_model)
        k, v:   (B, T_kv,  d_model)   — same as q for self-attention
        output: (B, T_q,   d_model)
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim  = cfg.head_dim
        self.d_model   = cfg.d_model

        # Individual projections (not fused) to match the phase-4 derivation
        self.Wq = nn.Linear(cfg.d_model, cfg.d_model)
        self.Wk = nn.Linear(cfg.d_model, cfg.d_model)
        self.Wv = nn.Linear(cfg.d_model, cfg.d_model)
        self.W0 = nn.Linear(cfg.d_model, cfg.d_model)  # Output projection (phase 5 concat)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) → (B, num_heads, T, head_dim)"""
        B, T, _ = x.shape
        return x.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _concat_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, num_heads, T, head_dim) → (B, T, d_model)"""
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().reshape(B, T, self.d_model)

    def forward(self,
                query: torch.Tensor,
                key:   torch.Tensor,
                value: torch.Tensor,
                mask:  torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output:  (B, T_q, d_model)
            weights: (B, num_heads, T_q, T_kv)
        """
        q = self._split_heads(self.Wq(query))   # (B, h, T_q,  head_dim)
        k = self._split_heads(self.Wk(key))     # (B, h, T_kv, head_dim)
        v = self._split_heads(self.Wv(value))   # (B, h, T_kv, head_dim)

        attn_out, weights = scaled_dot_product(q, k, v, mask)  # (B, h, T_q, head_dim)

        output = self.W0(self._concat_heads(attn_out))          # (B, T_q, d_model)

        return output, weights


def causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """Upper-triangular bool mask — True = masked (future positions).

    Shape: (seq_len, seq_len). Pass directly to MultiHeadAttention as mask=.
    """
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()


if __name__ == "__main__":
    cfg = TransformerConfig()
    mha = MultiHeadAttention(cfg)

    x   = torch.randn(2, 6, cfg.d_model)   # (B=2, T=6, d_model=8)
    out, w = mha(x, x, x)
    print("Self-attention output shape:", out.shape)     # (2, 6, 8)
    print("Attention weights shape:    ", w.shape)       # (2, 2, 6, 6)

    # Cross-attention: decoder (T=5) attending to encoder (T=7)
    dec = torch.randn(2, 5, cfg.d_model)
    enc = torch.randn(2, 7, cfg.d_model)
    out_cross, w_cross = mha(dec, enc, enc)
    print("Cross-attention output shape:", out_cross.shape)  # (2, 5, 8)
    print("Cross-attention weights shape:", w_cross.shape)   # (2, 2, 5, 7)
