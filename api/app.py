import os
import tempfile
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool

from api.schemas import (
    HealthResponse,
    SoundsResponse,
    ConfigResponse,
    PredictResponse,
)

from src.inference.checkpoint import load_model_from_checkpoint
from src.inference.checkpoint import resolve_checkpoint_path
from src.inference.predict import predict_file
from src.inference.interpret import interpret_result
from src.inference.audio_io import convert_audio_to_wav16k_mono
from src.utils.config import load_yaml_config

from uuid import uuid4
from src.service.storage import init_db, save_prediction, list_predictions


CONFIG_PATH = "configs/inference.yaml"
cfg = load_yaml_config(CONFIG_PATH)

DB_PATH = cfg.get("storage", {}).get("db_path", "artifacts/predictions.db")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt_path = resolve_checkpoint_path(cfg)
    model, ckpt, sound_to_idx, idx_to_sound = load_model_from_checkpoint(
        checkpoint_path=ckpt_path,
        device=device,
    )

    thr_bad = ckpt.get("thr_bad", cfg["fallbacks"]["thr_bad"])
    thr_sound = ckpt.get("thr_sound", None)
    thr_sound_val = thr_sound if thr_sound is not None else cfg["fallbacks"]["thr_sound"]

    model_name = ckpt.get("encoder_name", cfg.get("model", {}).get("encoder_name", "unknown"))
    model_version = ckpt.get("model_version", Path(ckpt_path).stem)

    app.state.model = model
    app.state.ckpt = ckpt
    app.state.sound_to_idx = sound_to_idx
    app.state.idx_to_sound = idx_to_sound
    app.state.thr_bad = thr_bad
    app.state.thr_sound_val = thr_sound_val
    app.state.model_name = model_name
    app.state.model_version = model_version

    init_db(DB_PATH)

    yield


app = FastAPI(
    title="Speech Defects API",
    description="API для детекции дефектов речи по скороговоркам",
    version="1.0.0",
    lifespan=lifespan,
)


def get_state():
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return app.state


def validate_expected_sounds(expected_sounds: List[str], sound_to_idx: dict):
    if not expected_sounds:
        raise HTTPException(status_code=400, detail="expected_sounds must not be empty")

    unknown = [s for s in expected_sounds if s not in sound_to_idx]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown sounds: {unknown}. "
                f"Allowed sounds: {sorted(sound_to_idx.keys())}"
            ),
        )


def _run_inference(
    *,
    file_content: bytes,
    suffix: str,
    model,
    ckpt: dict,
    sound_to_idx: dict,
    expected_sounds: List[str],
    thr_bad: float,
    thr_sound_val: float,
) -> dict:
    tmp_input_path = None
    tmp_wav_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_input:
            tmp_input.write(file_content)
            tmp_input_path = tmp_input.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            tmp_wav_path = tmp_wav.name

        try:
            convert_audio_to_wav16k_mono(tmp_input_path, tmp_wav_path)
        except Exception as e:
            raise RuntimeError(f"Audio conversion failed: {e}")

        return predict_file(
            model=model,
            path=tmp_wav_path,
            sound_to_idx=sound_to_idx,
            expected_sounds=expected_sounds,
            device=device,
            sr=ckpt["sr"],
            max_seconds=ckpt["max_seconds"],
            thr_bad=thr_bad,
            thr_sound=thr_sound_val,
        )

    finally:
        for path in [tmp_input_path, tmp_wav_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

@app.get("/health", response_model=HealthResponse)
def health():
    state = get_state()
    return HealthResponse(
        status="ok",
        device=str(device),
        num_sounds=len(state.sound_to_idx),
        sounds=sorted(state.sound_to_idx.keys()),
    )


@app.get("/ready")
def ready():
    return {"ready": hasattr(app.state, "model")}


@app.get("/sounds", response_model=SoundsResponse)
def get_sounds():
    state = get_state()
    return SoundsResponse(sounds=sorted(state.sound_to_idx.keys()))


@app.get("/config", response_model=ConfigResponse)
def get_config():
    state = get_state()
    return ConfigResponse(
        encoder_name=state.model_name,
        sr=int(state.ckpt.get("sr", 16000)),
        max_seconds=float(state.ckpt.get("max_seconds", 10.0)),
        thr_bad=float(state.thr_bad) if state.thr_bad is not None else None,
        thr_sound=float(state.thr_sound_val) if state.thr_sound_val is not None else None,
        available_sounds=sorted(state.sound_to_idx.keys()),
    )


@app.get("/history")
def history(limit: int = 100, session_id: Optional[str] = None):
    limit = max(1, min(limit, 1000))
    items = list_predictions(db_path=DB_PATH, limit=limit, session_id=session_id)
    return {
        "items": items,
        "count": len(items),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    expected_sounds: List[str] = Form(...),
    session_id: Optional[str] = Form(None),
):
    state = get_state()
    validate_expected_sounds(expected_sounds, state.sound_to_idx)

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    suffix = os.path.splitext(file.filename)[1].lower() if file.filename else ".wav"
    suffix = suffix or ".wav"

    try:
        res = await run_in_threadpool(
            partial(
                _run_inference,
                file_content=content,
                suffix=suffix,
                model=state.model,
                ckpt=state.ckpt,
                sound_to_idx=state.sound_to_idx,
                expected_sounds=expected_sounds,
                thr_bad=state.thr_bad,
                thr_sound_val=state.thr_sound_val,
            )
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    interp = interpret_result(
        p_bad=res["p_bad"],
        checked_sounds=res["checked_sounds"],
        thr_bad_good=state.thr_bad,
        thr_bad_bad=max(cfg["ui"]["thr_bad_bad_min"], state.thr_bad),
        thr_sound_possible=state.thr_sound_val,
        thr_sound_clear=max(cfg["ui"]["thr_sound_clear_min"], state.thr_sound_val),
    )

    checked_sound_scores = {sound: float(prob) for sound, prob in res["checked_sounds"]}

    save_prediction(
        db_path=DB_PATH,
        request_id=str(uuid4()),
        session_id=session_id,
        filename=file.filename or "uploaded_file",
        expected_sounds=res["expected_sounds"],
        p_bad=float(res["p_bad"]),
        is_bad=bool(res["is_bad"]),
        status=interp["status"],
        flagged_sounds=res["flagged_sounds"],
        all_sound_scores=checked_sound_scores,
        model_name=state.model_name,
        model_version=state.model_version,
        thr_bad=float(state.thr_bad) if state.thr_bad is not None else None,
        thr_sound=float(state.thr_sound_val) if state.thr_sound_val is not None else None,
    )

    return PredictResponse(
        file_name=file.filename or "uploaded_file",
        expected_sounds=res["expected_sounds"],
        p_bad=res["p_bad"],
        is_bad=res["is_bad"],
        checked_sounds=[{"sound": s, "prob": p} for s, p in res["checked_sounds"]],
        flagged_sounds=res["flagged_sounds"],
        status=interp["status"],
        message=interp["message"],
        sound_items=interp["sound_items"],
        normal_sounds=interp["normal_sounds"],
        possible_issue_sounds=interp["possible_issue_sounds"],
        clear_issue_sounds=interp["clear_issue_sounds"],
    )


@app.post("/predict_batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    expected_sounds: List[str] = Form(...),
    session_id: Optional[str] = Form(None),
):
    state = get_state()
    validate_expected_sounds(expected_sounds, state.sound_to_idx)

    results = []

    for file in files:
        suffix = os.path.splitext(file.filename)[1].lower() if file.filename else ".wav"
        suffix = suffix or ".wav"

        try:
            content = await file.read()
            if not content:
                raise RuntimeError("Empty file")

            res = await run_in_threadpool(
                partial(
                    _run_inference,
                    file_content=content,
                    suffix=suffix,
                    model=state.model,
                    ckpt=state.ckpt,
                    sound_to_idx=state.sound_to_idx,
                    expected_sounds=expected_sounds,
                    thr_bad=state.thr_bad,
                    thr_sound_val=state.thr_sound_val,
                )
            )

            interp = interpret_result(
                p_bad=res["p_bad"],
                checked_sounds=res["checked_sounds"],
                thr_bad_good=state.thr_bad,
                thr_bad_bad=max(cfg["ui"]["thr_bad_bad_min"], state.thr_bad),
                thr_sound_possible=state.thr_sound_val,
                thr_sound_clear=max(cfg["ui"]["thr_sound_clear_min"], state.thr_sound_val),
            )

            checked_sound_scores = {sound: float(prob) for sound, prob in res["checked_sounds"]}

            save_prediction(
                db_path=DB_PATH,
                request_id=str(uuid4()),
                session_id=session_id,
                filename=file.filename or "uploaded_file",
                expected_sounds=res["expected_sounds"],
                p_bad=float(res["p_bad"]),
                is_bad=bool(res["is_bad"]),
                status=interp["status"],
                flagged_sounds=res["flagged_sounds"],
                all_sound_scores=checked_sound_scores,
                model_name=state.model_name,
                model_version=state.model_version,
                thr_bad=float(state.thr_bad) if state.thr_bad is not None else None,
                thr_sound=float(state.thr_sound_val) if state.thr_sound_val is not None else None,
            )

            results.append({
                "file_name": file.filename,
                "p_bad": res["p_bad"],
                "status": interp["status"],
                "flagged_sounds": res["flagged_sounds"],
            })

        except Exception as e:
            results.append({
                "file_name": file.filename,
                "error": str(e),
            })

    return {"results": results, "count": len(results)}