from transformers import AutoModel

from src.models.model import SoundDefectModel


def create_sound_defect_model(
    num_sounds: int,
    encoder_name: str = "microsoft/wavlm-base",
    freeze_encoder: bool = True,
):
    encoder = AutoModel.from_pretrained(encoder_name)

    model = SoundDefectModel(
        encoder=encoder,
        num_sounds=num_sounds,
        freeze_encoder=freeze_encoder,
        encoder_name=encoder_name,
    )

    return model