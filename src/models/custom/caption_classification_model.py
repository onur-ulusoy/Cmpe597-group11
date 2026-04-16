import torch
import torch.nn as nn

class MemeClassificationModel(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=512, dropout=0.3):
        """
        classification model that fuses image and text embeddings.
        standard input_dim is 768 (image) + 768 (text) = 1536.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, img_emb, text_emb):
        # Concatenate image and text embeddings
        # shape: [batch, 1536]
        x = torch.cat([img_emb, text_emb], dim=-1)
        
        # Pass through MLP to get logits
        # shape: [batch, 1]
        logits = self.mlp(x)
        return logits
