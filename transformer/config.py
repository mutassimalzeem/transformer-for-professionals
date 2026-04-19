from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Single source of truth for all transformer hyperparameters.

    Defaults match the toy setup used throughout the experimental phases
    (d_model=8, 2 heads, seq_len up to 100). Scale up for real tasks.
    """

    vocab_size: int = 1000
    d_model: int = 8            # Embedding / hidden dimension
    num_heads: int = 2          # Must evenly divide d_model
    ff_dim: int = None          # FFN inner dim; defaults to d_model * 4
    max_seq_len: int = 100      # Max positional encoding length
    dropout: float = 0.0        # Not wired in yet — placeholder for training

    def __post_init__(self):
        if self.ff_dim is None:
            self.ff_dim = self.d_model * 4

        assert self.d_model % self.num_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        )

    @property
    def head_dim(self) -> int:
        """Dimension per attention head: d_model // num_heads."""
        return self.d_model // self.num_heads
