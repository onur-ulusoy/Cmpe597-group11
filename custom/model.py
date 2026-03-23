import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Shallow Image Encoder ---
class SimpleImageEncoder(nn.Module):
    def __init__(self, feat_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        
        # A simple 4-layer CNN. 
        # We increase channels slowly: 3 -> 32 -> 64 -> 128 -> 256
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 224 -> 112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 112 -> 56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 56 -> 28

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # 28 -> 1 (Global Average Pooling)
        )
        
        # Projection Head
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, feat_dim)
        )

        # IMPORTANT: Initialize weights for scratch training
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.proj(x)
        return x

# --- 2. Text Encoder (Keep it, but ensure initialization) ---
class TextEncoderGRU(nn.Module):
    def __init__(self, vocab_size, pad_idx, word_dim=256, hidden_dim=256, num_layers=1, dropout=0.1, feat_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, word_dim, padding_idx=pad_idx)
        self.input_norm = nn.LayerNorm(word_dim)
        
        self.gru = nn.GRU(
            input_size=word_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        pooled_dim = hidden_dim * 2
        
        self.proj = nn.Sequential(
            nn.Linear(pooled_dim, pooled_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pooled_dim, feat_dim)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.02)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        x = self.input_norm(x)
        
        out, _ = self.gru(x)
        
        # Mean Pooling masked
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        
        return self.proj(pooled)

# --- 3. Main Model ---
class Type1MatchingModel(nn.Module):
    def __init__(self, vocab_size, pad_idx, feat_dim=256, **kwargs):
        super().__init__()
        self.image_encoder = SimpleImageEncoder(feat_dim=feat_dim)
        self.text_encoder = TextEncoderGRU(vocab_size, pad_idx, feat_dim=feat_dim)
        
        # Learnable temperature, initialized to ~2.65
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def forward(self, images, caption_ids, caption_mask):
        img_emb = F.normalize(self.image_encoder(images), dim=-1)
        txt_emb = F.normalize(self.text_encoder(caption_ids, caption_mask), dim=-1)
        
        return {
            "image_emb": img_emb,
            "text_emb": txt_emb,
            "logit_scale": self.logit_scale.exp().clamp(max=100)
        }
    
    # Helper for evaluation
    def encode_image(self, images, normalize=True):
        feats = self.image_encoder(images)
        return F.normalize(feats, dim=-1) if normalize else feats

    def encode_text(self, input_ids, mask, normalize=True):
        feats = self.text_encoder(input_ids, mask)
        return F.normalize(feats, dim=-1) if normalize else feats
