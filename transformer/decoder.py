import torch
import torch.nn as nn

from .config import TransformerConfig
from .attention import MultiHeadAttention, causal_mask
from .layers import FeedForward, AddNorm
from .embedding import InputEmbedding


class DecoderLayer(nn.Module):
    """One decoder block: masked self-attn → Add&Norm → cross-attn → Add&Norm → FFN → Add&Norm.

    Consolidates phase 8's MaskedSelfAttention + CrossAttention into a single
    nn.Module. The causal mask is generated on-the-fly from the decoder sequence
    length so you never have to pass it manually.

    Shape contract:
        decoder_x:      (B, T_dec, d_model)
        encoder_output: (B, T_enc, d_model)
        output:         (B, T_dec, d_model)
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(cfg)     # Phase 8, task 1
        self.add_norm1        = AddNorm(cfg)

        self.cross_attn       = MultiHeadAttention(cfg)     # Phase 8, task 2
        self.add_norm2        = AddNorm(cfg)

        self.ffn              = FeedForward(cfg)
        self.add_norm3        = AddNorm(cfg)

    def forward(self,
                decoder_x: torch.Tensor,
                encoder_output: torch.Tensor) -> torch.Tensor:

        T_dec  = decoder_x.shape[1]
        mask   = causal_mask(T_dec, device=decoder_x.device)   # (T_dec, T_dec)

        # 1. Masked self-attention — decoder attends to its own past tokens only
        self_out, _ = self.masked_self_attn(decoder_x, decoder_x, decoder_x, mask)
        x = self.add_norm1(decoder_x, self_out)                 # (B, T_dec, d_model)

        # 2. Cross-attention — decoder queries the full encoder output
        # query = decoder, key/value = encoder
        cross_out, _ = self.cross_attn(x, encoder_output, encoder_output)
        x = self.add_norm2(x, cross_out)                        # (B, T_dec, d_model)

        # 3. Feed-forward
        ff_out = self.ffn(x)
        x = self.add_norm3(x, ff_out)                           # (B, T_dec, d_model)

        return x


class TransformerDecoder(nn.Module):
    """Full decoder: target embedding + N stacked decoder layers.

    Uses a separate InputEmbedding for the target vocabulary (which can differ
    from the source vocab size — e.g. English → French translation).

    Shape contract:
        tgt_token_ids:  (B, T_dec)
        encoder_output: (B, T_enc, d_model)
        output:         (B, T_dec, d_model)
    """

    def __init__(self, cfg: TransformerConfig, num_layers: int = 1):
        super().__init__()
        self.embedding = InputEmbedding(cfg)
        self.layers    = nn.ModuleList([DecoderLayer(cfg) for _ in range(num_layers)])

    def forward(self,
                tgt_token_ids: torch.Tensor,
                encoder_output: torch.Tensor) -> torch.Tensor:

        x = self.embedding(tgt_token_ids)           # (B, T_dec, d_model)

        for layer in self.layers:
            x = layer(x, encoder_output)            # (B, T_dec, d_model)

        return x


if __name__ == "__main__":
    cfg     = TransformerConfig()
    decoder = TransformerDecoder(cfg, num_layers=1)

    enc_out   = torch.randn(2, 7, cfg.d_model)           # Fake encoder output (B=2, T_enc=7)
    tgt_batch = torch.randint(0, cfg.vocab_size, (2, 5)) # Target tokens       (B=2, T_dec=5)

    out = decoder(tgt_batch, enc_out)

    print("Encoder output shape:", enc_out.shape)   # (2, 7, 8)
    print("Target token shape:  ", tgt_batch.shape) # (2, 5)
    print("Decoder output shape:", out.shape)        # (2, 5, 8)
