from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data.labeling import build_dataframe, build_label_vocab
from src.data.dataset import SoundDataset


def split_dataframe(df, test_size=0.2, random_state=42, stratify_by_is_good=True):
    stratify = df["is_good"] if stratify_by_is_good else None

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def create_dataloaders(
    good_dir: str,
    bad_dir: str,
    batch_size: int = 8,
    sr: int = 16000,
    max_len_sec: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
    num_workers: int = 0,
    stratify_by_is_good: bool = True,
):
    df = build_dataframe(good_dir, bad_dir)
    _, sound_to_idx, idx_to_sound = build_label_vocab(df)

    train_df, val_df = split_dataframe(
        df=df,
        test_size=test_size,
        random_state=random_state,
        stratify_by_is_good=stratify_by_is_good,
    )

    train_dataset = SoundDataset(
        df=train_df,
        sound_to_idx=sound_to_idx,
        sr=sr,
        max_len_sec=max_len_sec,
    )

    val_dataset = SoundDataset(
        df=val_df,
        sound_to_idx=sound_to_idx,
        sr=sr,
        max_len_sec=max_len_sec,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, sound_to_idx, idx_to_sound, train_df, val_df