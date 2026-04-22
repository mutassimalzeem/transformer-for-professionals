import torch
import torch.nn as nn

from .config import TransformerConfig


class FeedForward(nn.Module):
    """Position-wise feed-forward network (phase 6).

    Expand → ReLU → contract. The paper uses d_model * 4 for the inner dim,
    which is wired into TransformerConfig.ff_dim.

    Shape contract: (B, T, d_model) → (B, T, d_model)
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ff_dim),  # Expand: d_model → d_model * 4
            nn.ReLU(),
            nn.Linear(cfg.ff_dim, cfg.d_model),  # Contract: d_model * 4 → d_model
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AddNorm(nn.Module):
    """Residual connection + LayerNorm (the "Add & Norm" sublayer from phases 6).

    Used after both the attention sublayer and the FFN sublayer inside
    each encoder and decoder block. Shared here so neither imports from
    the other.

    Shape contract: x, sublayer_output (B, T, d_model) → (B, T, d_model)
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        return self.norm(x + sublayer_output)


if __name__ == "__main__":
    cfg = TransformerConfig()
    x   = torch.randn(2, 6, cfg.d_model)

    ffn     = FeedForward(cfg)
    addnorm = AddNorm(cfg)

    ff_out  = ffn(x)
    normed  = addnorm(x, ff_out)

    print("FFN output shape:    ", ff_out.shape)  # (2, 6, 8)
    print("AddNorm output shape:", normed.shape)  # (2, 6, 8)
