from pathlib import Path
import pandas as pd


def parse_sounds_from_filename(path: str) -> list[str]:
    name = Path(path).stem

    if "__" not in name:
        return []

    labels_part = name.split("__")[-1]
    labels_part = labels_part.replace(" ", "").replace("-", "").replace("_", "")

    return list(labels_part)


def build_dataframe(good_dir: str, bad_dir: str) -> pd.DataFrame:
    good_files = sorted(Path(good_dir).rglob("*.wav"))
    bad_files = sorted(Path(bad_dir).rglob("*.wav"))

    good_df = pd.DataFrame({
        "path": [str(p) for p in good_files],
        "is_good": [1] * len(good_files),
    })

    bad_df = pd.DataFrame({
        "path": [str(p) for p in bad_files],
        "is_good": [0] * len(bad_files),
    })

    df = pd.concat([good_df, bad_df], ignore_index=True)

    df["filename"] = df["path"].apply(lambda x: Path(x).name)
    df["sounds"] = df["path"].apply(parse_sounds_from_filename)

    return df


def build_label_vocab(df: pd.DataFrame):
    all_sounds = sorted({sound for sounds in df["sounds"] for sound in sounds})
    sound_to_idx = {sound: idx for idx, sound in enumerate(all_sounds)}
    idx_to_sound = {idx: sound for sound, idx in sound_to_idx.items()}

    return all_sounds, sound_to_idx, idx_to_sound


def sounds_to_multihot(sounds: list[str], sound_to_idx: dict[str, int]) -> list[int]:
    vector = [0] * len(sound_to_idx)

    for sound in sounds:
        if sound in sound_to_idx:
            vector[sound_to_idx[sound]] = 1

    return vector