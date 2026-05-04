import torch
import torch.nn.functional as F


def noisy_or_prob(p_defect: torch.Tensor, sounds_mask: torch.Tensor, eps: float = 1e-6):
    if p_defect.dim() == 1:
        p_defect = p_defect.unsqueeze(0)

    if sounds_mask.dim() == 1:
        sounds_mask = sounds_mask.unsqueeze(0)

    p = p_defect * sounds_mask
    p = torch.clamp(p, min=0.0, max=1.0)

    p_bad = 1.0 - torch.prod(1.0 - p, dim=1)
    p_bad = torch.clamp(p_bad, min=eps, max=1.0 - eps)

    return p_bad


def loss_fn(p_defect, sounds_mask, y_bad, lambda_sparse: float = 0.02):
    p_bad = noisy_or_prob(p_defect, sounds_mask)

    p_bad = torch.nan_to_num(p_bad, nan=0.5, posinf=1.0, neginf=0.0)
    y_bad = torch.clamp(y_bad, 0.0, 1.0)

    bce = F.binary_cross_entropy(p_bad, y_bad)

    sparse = (p_defect * sounds_mask).mean()
    sparse = torch.nan_to_num(sparse, nan=0.0, posinf=1.0, neginf=0.0)

    loss = bce + lambda_sparse * sparse

    return loss, {
        "bce": bce.item(),
        "sparse": sparse.item(),
    }