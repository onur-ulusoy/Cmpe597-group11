import torch
import torch.nn.functional as F


def total_loss(image_emb, text_emb, logit_scale, label_smoothing=0.0):
    """
    Symmetric contrastive loss for image/text retrieval.

    image_emb: [B, D], normalized meme/query embeddings
    text_emb:  [B, D], normalized caption embeddings
    logit_scale: scalar temperature scale
    """
    logits_per_image = logit_scale * (image_emb @ text_emb.T)
    logits_per_text = logits_per_image.T

    targets = torch.arange(image_emb.size(0), device=image_emb.device)

    loss_i = F.cross_entropy(
        logits_per_image,
        targets,
        label_smoothing=label_smoothing,
    )

    loss_t = F.cross_entropy(
        logits_per_text,
        targets,
        label_smoothing=label_smoothing,
    )

    loss = 0.5 * (loss_i + loss_t)

    parts = {
        "loss_contrastive": float(loss.detach().cpu().item()),
        "loss_image_to_text": float(loss_i.detach().cpu().item()),
        "loss_text_to_image": float(loss_t.detach().cpu().item()),
    }

    return loss, parts
