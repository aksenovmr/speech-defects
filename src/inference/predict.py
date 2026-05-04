import torch

from src.data.dataset import load_audio, pad_or_crop
from src.models.losses import noisy_or_prob
from src.inference.utils import build_sounds_mask, validate_expected_sounds
from src.data.labeling import parse_sounds_from_filename


@torch.no_grad()
def predict_file(
    model,
    path: str,
    sound_to_idx: dict[str, int],
    expected_sounds: list[str],
    device: torch.device,
    sr: int = 16000,
    max_seconds: float = 10.0,
    thr_bad: float | None = None,
    thr_sound: float | None = None,
):
    model.eval()

    validate_expected_sounds(expected_sounds, sound_to_idx)

    x = load_audio(path, sr=sr)
    x = pad_or_crop(x, max_len=int(sr * max_seconds), train=False)

    audio = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
    attn = (audio != 0).long()

    p_defect, _ = model(audio, attention_mask=attn)
    p_defect = p_defect[0]  # (V,)

    sounds_mask = build_sounds_mask(expected_sounds, sound_to_idx).unsqueeze(0).to(device)
    p_bad = noisy_or_prob(p_defect.unsqueeze(0), sounds_mask)[0].item()

    items = []
    for sound, idx in sound_to_idx.items():
        prob = float(p_defect[idx].item())
        items.append((sound, prob))

    items.sort(key=lambda x: x[1], reverse=True)

    checked_items = [(s, p) for s, p in items if s in expected_sounds]

    flagged_sounds = None
    if thr_sound is not None:
        flagged_sounds = [s for s, p in checked_items if p >= thr_sound]

    is_bad = None
    if thr_bad is not None:
        is_bad = bool(p_bad >= thr_bad)

    return {
        "file": path,
        "expected_sounds": expected_sounds,
        "p_bad": float(p_bad),
        "is_bad": is_bad,
        "all_sounds": items,
        "checked_sounds": checked_items,
        "flagged_sounds": flagged_sounds,
    }


@torch.no_grad()
def predict_dataset_file(
    model,
    path: str,
    sound_to_idx: dict[str, int],
    device: torch.device,
    sr: int = 16000,
    max_seconds: float = 10.0,
    thr_bad: float | None = None,
    thr_sound: float | None = None,
):
    expected_sounds = parse_sounds_from_filename(path)

    return predict_file(
        model=model,
        path=path,
        sound_to_idx=sound_to_idx,
        expected_sounds=expected_sounds,
        device=device,
        sr=sr,
        max_seconds=max_seconds,
        thr_bad=thr_bad,
        thr_sound=thr_sound,
    )