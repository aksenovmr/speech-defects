import torch

from src.models.losses import loss_fn


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def run_epoch(
    model,
    loader,
    device,
    train: bool,
    optimizer=None,
    lambda_sparse: float = 0.01,
    grad_clip: float = 1.0,
):
    model.train(train)
    total_loss = 0.0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        audio = batch["audio"]
        sounds_mask = batch["sounds_mask"]
        y_bad = batch["y_bad"]

        attn = (audio != 0).long()

        p_defect, _ = model(audio, attention_mask=attn)
        loss, _ = loss_fn(
            p_defect=p_defect,
            sounds_mask=sounds_mask,
            y_bad=y_bad,
            lambda_sparse=lambda_sparse,
        )

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)