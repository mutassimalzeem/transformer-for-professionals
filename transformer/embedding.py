import numpy as np
import torch
import torch.nn as nn

from .config import TransformerConfig


def sinusoidal_pe(max_seq_len: int, d_model: int) -> torch.Tensor:
    """Vectorized sinusoidal positional encoding (Vaswani et al. 2017).

    Returns a (max_seq_len, d_model) tensor. Even columns = sin, odd = cos.
    Not learned — fixed math, same result every call.
    """
    positions = np.arange(max_seq_len)[:, np.newaxis]           # (T, 1)
    dims      = np.arange(0, d_model, 2)[np.newaxis, :]         # (1, d_model//2)

    angles        = positions / (10000 ** (dims / d_model))     # (T, d_model//2)
    PE            = np.zeros((max_seq_len, d_model))
    PE[:, 0::2]   = np.sin(angles)
    PE[:, 1::2]   = np.cos(angles)

    return torch.tensor(PE, dtype=torch.float32)


class InputEmbedding(nn.Module):
    """Token embedding + learned positional embedding.

    Shape contract:
        input:  token_ids  (B, T)
        output: x          (B, T, d_model)
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.max_seq_len, cfg.d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0).expand(B, T)
        return self.token_emb(token_ids) + self.pos_emb(positions)   # (B, T, d_model)


if __name__ == "__main__":
    cfg   = TransformerConfig()
    model = InputEmbedding(cfg)
    ids   = torch.randint(0, cfg.vocab_size, (2, 6))
    out   = model(ids)
    print("Input  shape:", ids.shape)   # (2, 6)
    print("Output shape:", out.shape)   # (2, 6, 8)
