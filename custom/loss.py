import torch
import torch.nn.functional as F

def symmetric_contrastive_loss(
    image_emb,
    text_emb,
    logit_scale,
    label_smoothing: float = 0.0,
):
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

    return 0.5 * (loss_i + loss_t)


def total_loss(
    image_emb,
    text_emb,
    logit_scale,
    label_smoothing: float = 0.0,
):
    # Wrapper kept to maintain output dictionary format for your logging
    loss = symmetric_contrastive_loss(
        image_emb=image_emb,
        text_emb=text_emb,
        logit_scale=logit_scale,
        label_smoothing=label_smoothing,
    )

    return loss, {
        "loss_contrastive": loss.detach().item(),
    }