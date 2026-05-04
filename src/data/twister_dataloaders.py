from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data.labeling import build_dataframe, build_label_vocab
from src.data.training_data import prepare_training_dataframe
from src.data.twister_dataset import TwisterDataset


def split_train_val_test(
    df,
    test_size=0.2,
    val_size_from_train=0.2,
    random_state=42,
    stratify_by_is_good=True,
):
    stratify_full = df["is_good"] if stratify_by_is_good else None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_full,
    )

    stratify_train = train_df["is_good"] if stratify_by_is_good else None

    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size_from_train,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_train,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def create_twister_dataloaders(
    good_dir: str,
    bad_dir: str,
    batch_size: int = 8,
    sr: int = 16000,
    max_len_sec: int = 10,
    test_size: float = 0.2,
    val_size_from_train: float = 0.2,
    random_state: int = 42,
    num_workers: int = 0,
    stratify_by_is_good: bool = True,
):
    df = build_dataframe(good_dir, bad_dir)
    _, sound_to_idx, idx_to_sound = build_label_vocab(df)
    df = prepare_training_dataframe(df, sound_to_idx)

    train_df, val_df, test_df = split_train_val_test(
        df=df,
        test_size=test_size,
        val_size_from_train=val_size_from_train,
        random_state=random_state,
        stratify_by_is_good=stratify_by_is_good,
    )

    train_dataset = TwisterDataset(
        df=train_df,
        sr=sr,
        max_len_sec=max_len_sec,
        train=True,
    )

    val_dataset = TwisterDataset(
        df=val_df,
        sr=sr,
        max_len_sec=max_len_sec,
        train=False,
    )

    test_dataset = TwisterDataset(
        df=test_df,
        sr=sr,
        max_len_sec=max_len_sec,
        train=False,
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        sound_to_idx,
        idx_to_sound,
        train_df,
        val_df,
        test_df,
    )