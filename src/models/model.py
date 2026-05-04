import torch
import torch.nn as nn


class SoundDefectModel(nn.Module):
    def __init__(
        self,
        encoder,
        num_sounds: int,
        freeze_encoder: bool = True,
        encoder_name: str = "unknown",
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = encoder
        self.encoder_name = encoder_name

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        hidden_size = self.encoder.config.hidden_size

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_sounds),
        )

    def forward(self, audio, attention_mask=None):
        out = self.encoder(input_values=audio, attention_mask=attention_mask)
        h = out.last_hidden_state

        if attention_mask is not None and hasattr(self.encoder, "_get_feature_vector_attention_mask"):
            feat_mask = self.encoder._get_feature_vector_attention_mask(
                h.shape[1], attention_mask
            )
            feat_mask = feat_mask.unsqueeze(-1).float()
            h_masked = h * feat_mask
            emb = h_masked.sum(dim=1) / feat_mask.sum(dim=1).clamp(min=1e-6)
        else:
            emb = h.mean(dim=1)

        logits = self.head(emb)
        p_defect = torch.sigmoid(logits)

        return p_defect, logits