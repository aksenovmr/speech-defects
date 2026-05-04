import numpy as np
import torch


@torch.no_grad()
def collect_good_sound_probs(model, df_good, sound_to_idx, device, sr=16000, max_seconds=10.0):
    from src.data.twister_dataset import TwisterDataset

    dataset = TwisterDataset(
        df=df_good,
        sr=sr,
        max_len_sec=max_seconds,
        train=False,
        return_metadata=False,
    )

    all_probs = []

    for i in range(len(dataset)):
        sample = dataset[i]

        audio = sample["audio"].unsqueeze(0).to(device)
        sounds_mask = sample["sounds_mask"].to(device)

        attn = (audio != 0).long()
        p_defect, _ = model(audio, attention_mask=attn)
        p_defect = p_defect[0].detach().cpu().numpy()
        sounds_mask = sounds_mask.detach().cpu().numpy()

        active_probs = p_defect[sounds_mask > 0.5]
        if len(active_probs) > 0:
            all_probs.extend(active_probs.tolist())

    return np.array(all_probs, dtype=float)


def tune_thr_sound(
    model,
    df_good,
    sound_to_idx,
    device,
    sr=16000,
    max_seconds=10.0,
    quantile=0.90,
    fallback=0.25,
):
    probs = collect_good_sound_probs(
        model=model,
        df_good=df_good,
        sound_to_idx=sound_to_idx,
        device=device,
        sr=sr,
        max_seconds=max_seconds,
    )

    if len(probs) == 0:
        return float(fallback), {
            "num_probs": 0,
            "quantile": quantile,
            "used_fallback": True,
        }

    thr_sound = float(np.quantile(probs, quantile))

    return thr_sound, {
        "num_probs": int(len(probs)),
        "quantile": float(quantile),
        "used_fallback": False,
        "min": float(np.min(probs)),
        "mean": float(np.mean(probs)),
        "max": float(np.max(probs)),
    }