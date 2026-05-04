from typing import List, Optional
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    device: str
    num_sounds: int
    sounds: List[str]


class SoundsResponse(BaseModel):
    sounds: List[str]


class ConfigResponse(BaseModel):
    encoder_name: str
    sr: int
    max_seconds: float
    thr_bad: Optional[float]
    thr_sound: Optional[float]
    available_sounds: List[str]


class CheckedSoundItem(BaseModel):
    sound: str
    prob: float


class SoundInterpretationItem(BaseModel):
    sound: str
    prob: float
    level: str


class PredictResponse(BaseModel):
    file_name: str
    expected_sounds: List[str]
    p_bad: float
    is_bad: Optional[bool]
    checked_sounds: List[CheckedSoundItem]
    flagged_sounds: Optional[List[str]]
    status: str
    message: str
    sound_items: List[SoundInterpretationItem]
    normal_sounds: List[str]
    possible_issue_sounds: List[str]
    clear_issue_sounds: List[str]


class ErrorResponse(BaseModel):
    detail: str