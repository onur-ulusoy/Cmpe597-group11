import torch
import torch.nn as nn


class MemeClassificationModel(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=512, dropout=0.3):
        """
        Binary classifier for meme-caption type classification.

        Input:
            image embedding: 768 dim
            text embedding : 768 dim
            concatenated   : 1536 dim

        Output:
            one logit for binary classification.
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, img_emb, text_emb):
        x = torch.cat([img_emb, text_emb], dim=-1)
        logits = self.mlp(x)
        return logits