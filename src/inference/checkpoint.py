import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
from src.models.factory import create_sound_defect_model

def resolve_checkpoint_path(cfg: dict) -> str:
    local_path = Path(cfg["paths"]["checkpoint_path"])

    if local_path.exists():
        return str(local_path)

    hf_repo_id = cfg["paths"].get("hf_repo_id")
    hf_filename = cfg["paths"].get("hf_filename")

    if not hf_repo_id or not hf_filename:
        raise FileNotFoundError()

    local_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded = hf_hub_download(
        repo_id=hf_repo_id,
        filename=hf_filename,
        local_dir=str(local_path.parent),
    )

    return downloaded


def load_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return ckpt


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = load_checkpoint(checkpoint_path, device)

    sound_to_idx = ckpt["sound2id"]
    idx_to_sound = {idx: sound for sound, idx in sound_to_idx.items()}

    model = create_sound_defect_model(
        num_sounds=len(sound_to_idx),
        encoder_name=ckpt["encoder_name"],
        freeze_encoder=True,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt, sound_to_idx, idx_to_sound