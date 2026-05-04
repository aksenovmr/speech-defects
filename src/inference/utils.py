import torch


def build_sounds_mask(expected_sounds: list[str], sound_to_idx: dict[str, int]) -> torch.Tensor:
    mask = torch.zeros(len(sound_to_idx), dtype=torch.float32)

    for sound in expected_sounds:
        if sound in sound_to_idx:
            mask[sound_to_idx[sound]] = 1.0

    return mask


def validate_expected_sounds(expected_sounds: list[str], sound_to_idx: dict[str, int]):
    unknown = [s for s in expected_sounds if s not in sound_to_idx]
    if unknown:
        raise ValueError(
            f"Неизвестные звуки: {unknown}. Допустимые: {sorted(sound_to_idx.keys())}"
        )