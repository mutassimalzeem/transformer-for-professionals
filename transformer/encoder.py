import torch
import torch.nn as nn

from .config import TransformerConfig
from .attention import MultiHeadAttention
from .layers import FeedForward, AddNorm
from .embedding import InputEmbedding


class EncoderLayer(nn.Module):
    """One encoder block: multi-head self-attention → Add&Norm → FFN → Add&Norm.

    This is the clean consolidation of phase 7's EncoderBlock, with the
    Add&Norm sublayer extracted into its own reusable class.

    Shape contract:
        input:  x  (B, T, d_model)
        output:    (B, T, d_model)
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg)
        self.add_norm1 = AddNorm(cfg)
        self.ffn       = FeedForward(cfg)
        self.add_norm2 = AddNorm(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x)      # Self-attention (no mask — encoder sees all)
        x = self.add_norm1(x, attn_out)             # Add & Norm 1

        ff_out = self.ffn(x)                        # Feed-forward
        x = self.add_norm2(x, ff_out)               # Add & Norm 2

        return x                                    # (B, T, d_model)


class TransformerEncoder(nn.Module):
    """Full encoder: embedding + positional encoding + N stacked encoder layers.

    num_layers=1 matches the single-block setup from phase 7.
    Increase for real workloads.

    Shape contract:
        input:  token_ids  (B, T)
        output:            (B, T, d_model)
    """

    def __init__(self, cfg: TransformerConfig, num_layers: int = 1):
        super().__init__()
        self.embedding = InputEmbedding(cfg)
        self.layers    = nn.ModuleList([EncoderLayer(cfg) for _ in range(num_layers)])

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)       # (B, T, d_model)

        for layer in self.layers:
            x = layer(x)                   # (B, T, d_model) — same shape through every block

        return x


if __name__ == "__main__":
    cfg     = TransformerConfig()
    encoder = TransformerEncoder(cfg, num_layers=1)

    batch = torch.randint(0, cfg.vocab_size, (2, 6))   # (B=2, T=6)
    out   = encoder(batch)

    print("Input  shape:", batch.shape)  # (2, 6)
    print("Output shape:", out.shape)    # (2, 6, 8)
