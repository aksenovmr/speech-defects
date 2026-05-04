import subprocess
from pathlib import Path


def convert_audio_to_wav16k_mono(input_path: str, output_path: str):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        output_path,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Не удалось конвертировать аудио через ffmpeg. "
            f"stderr: {result.stderr}"
        )

    if not Path(output_path).exists():
        raise RuntimeError("ffmpeg не создал выходной wav-файл.")