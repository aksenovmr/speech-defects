import pandas as pd

from src.data.labeling import sounds_to_multihot


def prepare_training_dataframe(df: pd.DataFrame, sound_to_idx: dict[str, int]) -> pd.DataFrame:
    df = df.copy()

    df["sounds_mh"] = df["sounds"].apply(lambda sounds: sounds_to_multihot(sounds, sound_to_idx))
    df["y_bad"] = 1.0 - df["is_good"].astype(float)

    return df