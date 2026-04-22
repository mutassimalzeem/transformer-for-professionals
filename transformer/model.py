import torch
import torch.nn as nn

from .config import TransformerConfig
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class Transformer(nn.Module):
    """Full encoder-decoder transformer.

    Wires TransformerEncoder + TransformerDecoder together and adds a final
    linear projection from d_model → vocab_size (the logit head).

    This is the "next step" from the roadmap — full encoder-decoder integration.

    Shape contract:
        src_ids:  (B, T_enc)    Source token ids
        tgt_ids:  (B, T_dec)    Target token ids (teacher-forced during training)
        output:   (B, T_dec, vocab_size)   Raw logits — pass through softmax for probs
    """

    def __init__(self, cfg: TransformerConfig, num_layers: int = 1):
        super().__init__()
        self.encoder    = TransformerEncoder(cfg, num_layers)
        self.decoder    = TransformerDecoder(cfg, num_layers)
        self.output_proj = nn.Linear(cfg.d_model, cfg.vocab_size)   # Logit head

    def forward(self,
                src_ids: torch.Tensor,
                tgt_ids: torch.Tensor) -> torch.Tensor:

        encoder_output = self.encoder(src_ids)                  # (B, T_enc, d_model)
        decoder_output = self.decoder(tgt_ids, encoder_output)  # (B, T_dec, d_model)
        logits         = self.output_proj(decoder_output)        # (B, T_dec, vocab_size)

        return logits


if __name__ == "__main__":
    cfg   = TransformerConfig()
    model = Transformer(cfg, num_layers=1)

    src = torch.randint(0, cfg.vocab_size, (2, 7))   # (B=2, T_enc=7)
    tgt = torch.randint(0, cfg.vocab_size, (2, 5))   # (B=2, T_dec=5)

    logits = model(src, tgt)

    print("Source shape:", src.shape)     # (2, 7)
    print("Target shape:", tgt.shape)     # (2, 5)
    print("Logits shape:", logits.shape)  # (2, 5, 1000)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
