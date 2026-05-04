import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

from src.data.labeling import sounds_to_multihot
import soundfile as sf


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    errors = []
    try:
        x, file_sr = sf.read(path, always_2d=False)

        if x.ndim > 1:
            x = np.mean(x, axis=1)

        x = x.astype(np.float32)

        if file_sr != sr:
            x = librosa.resample(x, orig_sr=file_sr, target_sr=sr)

        return x.astype(np.float32)
    except Exception as e:
        errors.append(f"soundfile: {repr(e)}")

    try:
        x, _ = librosa.load(path, sr=sr, mono=True)
        return x.astype(np.float32)
    except Exception as e:
        errors.append(f"librosa: {repr(e)}")

    raise RuntimeError(
        "Не удалось прочитать аудиофайл. "
        "Проверьте, что файл не повреждён и действительно является WAV/поддерживаемым аудио. "
        f"Подробности: {' | '.join(errors)}"
    )


def pad_or_crop(x: np.ndarray, max_len: int, train: bool = False) -> np.ndarray:
    if len(x) == 0:
        return np.zeros((max_len,), dtype=np.float32)

    if len(x) > max_len:
        if train:
            start = np.random.randint(0, len(x) - max_len + 1)
        else:
            start = (len(x) - max_len) // 2
        return x[start:start + max_len]

    if len(x) < max_len:
        return np.pad(x, (0, max_len - len(x)), mode="constant")

    return x


class SoundDataset(Dataset):
    def __init__(
        self,
        df,
        sound_to_idx: dict[str, int],
        sr: int = 16000,
        max_len_sec: int = 10,
        train: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.sound_to_idx = sound_to_idx
        self.sr = sr
        self.max_len_sec = max_len_sec
        self.max_len = int(sr * max_len_sec)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio = load_audio(path=row["path"], sr=self.sr)
        audio = pad_or_crop(audio, max_len=self.max_len, train=self.train)

        label = sounds_to_multihot(row["sounds"], self.sound_to_idx)

        return {
            "audio": torch.tensor(audio, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "is_good": torch.tensor(row["is_good"], dtype=torch.float32),
            "path": row["path"],
            "sounds": row["sounds"],
        }