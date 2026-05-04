import torch
from torch.utils.data import Dataset

from src.data.dataset import load_audio, pad_or_crop


class TwisterDataset(Dataset):
    def __init__(
        self,
        df,
        sr: int = 16000,
        max_len_sec: int = 10,
        train: bool = False,
        return_metadata: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.sr = sr
        self.max_len_sec = max_len_sec
        self.max_len = int(sr * max_len_sec)
        self.train = train
        self.return_metadata = return_metadata

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = load_audio(path=row["path"], sr=self.sr)
        x = pad_or_crop(x, max_len=self.max_len, train=self.train)

        sounds_mh = torch.tensor(row["sounds_mh"], dtype=torch.float32)
        sounds_mask = sounds_mh.clone()
        y_bad = torch.tensor(float(row["y_bad"]), dtype=torch.float32)
        is_good = torch.tensor(float(row["is_good"]), dtype=torch.float32)

        sample = {
            "audio": torch.tensor(x, dtype=torch.float32),
            "sounds_mask": sounds_mask,
            "y_bad": y_bad,
            "is_good": is_good,
        }

        if self.return_metadata:
            sample["path"] = row["path"]
            sample["sounds"] = row["sounds"]

        return sample