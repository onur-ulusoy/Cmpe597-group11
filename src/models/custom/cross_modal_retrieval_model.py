import torch
import torch.nn as nn
import torch.nn.functional as F

class SymmetricContrastiveLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, image_emb, text_emb, logit_scale):
        logits_per_image = logit_scale * (image_emb @ text_emb.T)
        logits_per_text = logits_per_image.T

        targets = torch.arange(image_emb.size(0), device=image_emb.device)

        loss_i = F.cross_entropy(
            logits_per_image,
            targets,
            label_smoothing=self.label_smoothing,
        )
        loss_t = F.cross_entropy(
            logits_per_text,
            targets,
            label_smoothing=self.label_smoothing,
        )

        loss = 0.5 * (loss_i + loss_t)

        return loss, {
            "loss_contrastive": loss.detach().item(),
        }

class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout2d(dropout)

        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = F.gelu(out)
        return out

class ImageEncoderCNN(nn.Module):
    def __init__(self, feat_dim: int = 256, dropout: float = 0.05):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64, stride=1, dropout=dropout),
            ResidualBlock(64, 64, stride=1, dropout=dropout),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, dropout=dropout),
            ResidualBlock(128, 128, stride=1, dropout=dropout),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, dropout=dropout),
            ResidualBlock(256, 256, stride=1, dropout=dropout),
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2, dropout=dropout),
            ResidualBlock(512, 512, stride=1, dropout=dropout),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, feat_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.proj(x)
        return x

class TextEncoderGRU(nn.Module):
    def __init__(
        self, vocab_size: int, pad_idx: int, word_dim: int = 256, 
        hidden_dim: int = 256, num_layers: int = 1, dropout: float = 0.15, feat_dim: int = 256,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, word_dim, padding_idx=pad_idx)
        self.embedding_dropout = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(word_dim)

        self.gru = nn.GRU(
            input_size=word_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=0.0,
        )

        pooled_dim = hidden_dim * 2 * 2
        self.proj = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, pooled_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pooled_dim, feat_dim),
        )

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        x = self.input_norm(x)
        x = self.embedding_dropout(x)

        out, _ = self.gru(x)
        mask = attention_mask.unsqueeze(-1).float()
        mean_pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        masked_out = out.masked_fill(mask == 0, float("-inf"))
        max_pooled = masked_out.max(dim=1).values
        max_pooled[max_pooled == float("-inf")] = 0.0

        pooled = torch.cat([mean_pooled, max_pooled], dim=-1)
        pooled = self.proj(pooled)
        return pooled

class FeatureFusion(nn.Module):
    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.LayerNorm(feat_dim)
        )

    def forward(self, img_feat, title_feat):
        fused = torch.cat([img_feat, title_feat], dim=-1)
        return self.proj(fused)

class MatchingModel(nn.Module):
    def __init__(
        self, vocab_size: int, pad_idx: int, model_type: str = "type1",
        feat_dim: int = 256, word_dim: int = 256, text_hidden_dim: int = 256,
        text_num_layers: int = 1, text_dropout: float = 0.15, image_dropout: float = 0.05,
    ):
        super().__init__()
        self.model_type = model_type
        self.image_encoder = ImageEncoderCNN(feat_dim=feat_dim, dropout=image_dropout)
        
        self.caption_encoder = TextEncoderGRU(
            vocab_size=vocab_size, pad_idx=pad_idx, word_dim=word_dim,
            hidden_dim=text_hidden_dim, num_layers=text_num_layers,
            dropout=text_dropout, feat_dim=feat_dim,
        )

        if self.model_type == "type2":
            self.title_encoder = TextEncoderGRU(
                vocab_size=vocab_size, pad_idx=pad_idx, word_dim=word_dim,
                hidden_dim=text_hidden_dim, num_layers=text_num_layers,
                dropout=text_dropout, feat_dim=feat_dim,
            )
            self.fusion = FeatureFusion(feat_dim=feat_dim)

        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def encode_meme(self, images, title_ids=None, title_mask=None, normalize: bool = True):
        feat = self.image_encoder(images)
        
        if self.model_type == "type2" and title_ids is not None:
            title_feat = self.title_encoder(title_ids, title_mask)
            feat = self.fusion(feat, title_feat)
            
        if normalize:
            feat = F.normalize(feat, dim=-1)
        return feat

    def encode_caption(self, caption_ids, caption_mask, normalize: bool = True):
        feats = self.caption_encoder(caption_ids, caption_mask)
        if normalize:
            feats = F.normalize(feats, dim=-1)
        return feats

    def forward(self, images, title_ids, title_mask, caption_ids, caption_mask):
        meme_emb = self.encode_meme(images, title_ids, title_mask, normalize=True)
        caption_emb = self.encode_caption(caption_ids, caption_mask, normalize=True)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        return {
            "image_emb": meme_emb,
            "text_emb": caption_emb,
            "logit_scale": logit_scale,
        }