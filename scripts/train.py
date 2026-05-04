import os
import argparse
import torch

from src.data.twister_dataloaders import create_twister_dataloaders
from src.models.factory import create_sound_defect_model
from src.training.trainer import run_epoch
from src.training.metrics import predict_bad_probs, compute_metrics, best_f1_threshold
from src.inference.thresholding import tune_thr_sound
from src.training.wandb_utils import (
    init_wandb_if_needed,
    log_wandb_if_needed,
    finish_wandb_if_needed,
)
from src.utils.config import load_yaml_config


DEFAULT_CONFIG_PATH = "configs/train.yaml"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to training config yaml",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)

    EXPERIMENT_NAME = cfg.get("experiment", {}).get("name")

    GOOD_DIR = cfg["paths"]["good_dir"]
    BAD_DIR = cfg["paths"]["bad_dir"]
    CKPT_PATH = cfg["paths"]["checkpoint_path"]

    SR = cfg["data"]["sr"]
    MAX_SECONDS = cfg["data"]["max_seconds"]
    TEST_SIZE = cfg["data"]["test_size"]
    VAL_SIZE_FROM_TRAIN = cfg["data"]["val_size_from_train"]
    RANDOM_STATE = cfg["data"]["random_state"]
    NUM_WORKERS = cfg["data"]["num_workers"]

    ENCODER_NAME = cfg["model"]["encoder_name"]
    FREEZE_MODE = cfg["model"]["freeze_mode"]
    UNFREEZE_LAST_N = cfg["model"]["unfreeze_last_n"]

    BATCH_SIZE = cfg["training"]["batch_size"]
    LR_HEAD = cfg["training"]["lr_head"]
    LR_ENCODER = cfg["training"]["lr_encoder"]
    WEIGHT_DECAY = cfg["training"]["weight_decay"]
    NUM_EPOCHS = cfg["training"]["num_epochs"]
    LAMBDA_SPARSE = cfg["training"]["lambda_sparse"]
    GRAD_CLIP = cfg["training"]["grad_clip"]

    THR_DEFAULT = cfg["thresholds"]["default_bad"]
    THR_SCREENING = cfg["thresholds"]["screening_bad"]
    THR_SOUND_FALLBACK = cfg["thresholds"]["fallback_sound"]
    THR_SOUND_QUANTILE = cfg["thresholds"]["sound_quantile"]

    USE_WANDB = cfg["wandb"]["use"]
    WANDB_PROJECT = cfg["wandb"]["project"]

    os.makedirs("checkpoints", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(f"\nUsing encoder: {ENCODER_NAME}")
    print(f"Using config: {args.config}")

    run_name = (
        EXPERIMENT_NAME
        or f"{ENCODER_NAME.split('/')[-1]}_{FREEZE_MODE}_u{UNFREEZE_LAST_N}_seed{RANDOM_STATE}"
    )
    wandb_group = "encoder_comparison"
    wandb_tags = [
        ENCODER_NAME.split("/")[-1],
        FREEZE_MODE,
        f"u{UNFREEZE_LAST_N}",
        f"seed{RANDOM_STATE}",
    ]

    config_for_wandb = {
        "config_path": args.config,
        "experiment_name": EXPERIMENT_NAME,
        "good_dir": GOOD_DIR,
        "bad_dir": BAD_DIR,
        "checkpoint_path": CKPT_PATH,
        "batch_size": BATCH_SIZE,
        "sr": SR,
        "max_seconds": MAX_SECONDS,
        "test_size": TEST_SIZE,
        "val_size_from_train": VAL_SIZE_FROM_TRAIN,
        "random_state": RANDOM_STATE,
        "num_workers": NUM_WORKERS,
        "encoder_name": ENCODER_NAME,
        "freeze_mode": FREEZE_MODE,
        "unfreeze_last_n": UNFREEZE_LAST_N,
        "lr_head": LR_HEAD,
        "lr_encoder": LR_ENCODER,
        "weight_decay": WEIGHT_DECAY,
        "num_epochs": NUM_EPOCHS,
        "lambda_sparse": LAMBDA_SPARSE,
        "grad_clip": GRAD_CLIP,
        "thr_default": THR_DEFAULT,
        "thr_screening": THR_SCREENING,
        "thr_sound_fallback": THR_SOUND_FALLBACK,
        "thr_sound_quantile": THR_SOUND_QUANTILE,
        "device": str(device),
    }

    wandb_run = init_wandb_if_needed(
        use_wandb=USE_WANDB,
        project=WANDB_PROJECT,
        config=config_for_wandb,
        run_name=run_name,
        group=wandb_group,
        tags=wandb_tags,
    )

    (
        train_loader,
        val_loader,
        test_loader,
        sound_to_idx,
        idx_to_sound,
        train_df,
        val_df,
        test_df,
    ) = create_twister_dataloaders(
        good_dir=GOOD_DIR,
        bad_dir=BAD_DIR,
        batch_size=BATCH_SIZE,
        sr=SR,
        max_len_sec=MAX_SECONDS,
        test_size=TEST_SIZE,
        val_size_from_train=VAL_SIZE_FROM_TRAIN,
        random_state=RANDOM_STATE,
        num_workers=NUM_WORKERS,
    )

    model = create_sound_defect_model(
        num_sounds=len(sound_to_idx),
        encoder_name=ENCODER_NAME,
        freeze_encoder=True,
    ).to(device)

    if FREEZE_MODE == "freeze_all":
        pass
    elif FREEZE_MODE == "unfreeze_last_n":
        if UNFREEZE_LAST_N > 0:
            if hasattr(model.encoder, "encoder") and hasattr(model.encoder.encoder, "layers"):
                layers = model.encoder.encoder.layers
            elif hasattr(model.encoder, "layers"):
                layers = model.encoder.layers
            else:
                raise ValueError("Cannot find encoder layers for unfreezing")

            for layer in layers[-UNFREEZE_LAST_N:]:
                for p in layer.parameters():
                    p.requires_grad = True
    else:
        raise ValueError(
            f"Unknown freeze_mode: {FREEZE_MODE}. "
            f"Supported: freeze_all, unfreeze_last_n"
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M")

    log_wandb_if_needed(
        wandb_run,
        {
            "model/trainable_params_m": trainable / 1e6,
            "model/total_params_m": total / 1e6,
            "model/num_sounds": len(sound_to_idx),
            "model/encoder_name": ENCODER_NAME,
        },
    )

    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())

    param_groups = [
        {"params": head_params, "lr": LR_HEAD},
    ]

    if len(enc_params) > 0:
        param_groups.append({"params": enc_params, "lr": LR_ENCODER})

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
    )

    best_val = float("inf")
    best_epoch = -1

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            train=True,
            optimizer=optimizer,
            lambda_sparse=LAMBDA_SPARSE,
            grad_clip=GRAD_CLIP,
        )

        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            train=False,
            optimizer=None,
            lambda_sparse=LAMBDA_SPARSE,
            grad_clip=GRAD_CLIP,
        )

        scheduler.step(val_loss)

        current_head_lr = optimizer.param_groups[0]["lr"]
        current_encoder_lr = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        log_wandb_if_needed(
            wandb_run,
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "lr/head": current_head_lr,
                "lr/encoder": current_encoder_lr,
            },
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch

            ckpt = {
                "epoch": epoch,
                "val_loss": float(best_val),
                "model_state_dict": model.state_dict(),
                "encoder_name": ENCODER_NAME,
                "sound2id": sound_to_idx,
                "sr": SR,
                "max_seconds": MAX_SECONDS,
                "thr_bad": None,
                "thr_sound": None,
                "unfreeze_last_n": UNFREEZE_LAST_N,
                "freeze_mode": FREEZE_MODE,
                "experiment_name": EXPERIMENT_NAME,
                "config_path": args.config,
            }

            torch.save(ckpt, CKPT_PATH)
            print(f"  Saved new best checkpoint: {CKPT_PATH}")

    print(f"\nBest checkpoint: epoch {best_epoch} with val_loss={best_val:.4f}")

    log_wandb_if_needed(
        wandb_run,
        {
            "best/epoch": best_epoch,
            "best/val_loss": best_val,
        },
    )

    best_ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    y_val, p_val = predict_bad_probs(model, val_loader, device)
    y_test, p_test = predict_bad_probs(model, test_loader, device)

    thr_bad, best_f1_val = best_f1_threshold(y_val, p_val)

    val_metrics = {
        "default": compute_metrics(y_val, p_val, THR_DEFAULT),
        "best_f1": compute_metrics(y_val, p_val, thr_bad),
    }

    test_metrics = {
        "default": compute_metrics(y_test, p_test, THR_DEFAULT),
        "screening": compute_metrics(y_test, p_test, THR_SCREENING),
        "best_f1_from_val": compute_metrics(y_test, p_test, thr_bad),
    }

    print("\nVAL metrics:")
    print(val_metrics)

    print("\nTEST metrics:")
    print(test_metrics)

    print("\nBest thr_bad on VAL:", thr_bad)
    print("Best F1 on VAL:", best_f1_val)

    df_val_good = val_df[val_df["is_good"] == 1].reset_index(drop=True)

    thr_sound, thr_sound_info = tune_thr_sound(
        model=model,
        df_good=df_val_good,
        sound_to_idx=sound_to_idx,
        device=device,
        sr=SR,
        max_seconds=MAX_SECONDS,
        quantile=THR_SOUND_QUANTILE,
        fallback=THR_SOUND_FALLBACK,
    )

    print("\nTuned thr_sound on VAL good:")
    print("thr_sound:", thr_sound)
    print("info:", thr_sound_info)

    best_ckpt["thr_bad"] = float(thr_bad)
    best_ckpt["thr_sound"] = float(thr_sound)

    torch.save(best_ckpt, CKPT_PATH)
    print(f"\nUpdated checkpoint with thr_bad={thr_bad}, thr_sound={thr_sound}: {CKPT_PATH}")

    log_wandb_if_needed(
        wandb_run,
        {
            "thresholds/thr_bad": float(thr_bad),
            "thresholds/thr_sound": float(thr_sound),
            "thresholds/thr_sound_num_probs": thr_sound_info.get("num_probs", 0),

            "val/default_auc": val_metrics["default"]["auc"],
            "val/default_ap": val_metrics["default"]["ap"],
            "val/default_acc": val_metrics["default"]["acc"],
            "val/default_f1": val_metrics["default"]["f1"],
            "val/default_f1_macro": val_metrics["default"]["f1_macro"],
            "val/default_precision": val_metrics["default"]["precision"],
            "val/default_recall": val_metrics["default"]["recall"],

            "val/best_f1_auc": val_metrics["best_f1"]["auc"],
            "val/best_f1_ap": val_metrics["best_f1"]["ap"],
            "val/best_f1_acc": val_metrics["best_f1"]["acc"],
            "val/best_f1_f1": val_metrics["best_f1"]["f1"],
            "val/best_f1_f1_macro": val_metrics["best_f1"]["f1_macro"],
            "val/best_f1_precision": val_metrics["best_f1"]["precision"],
            "val/best_f1_recall": val_metrics["best_f1"]["recall"],

            "test/default_auc": test_metrics["default"]["auc"],
            "test/default_ap": test_metrics["default"]["ap"],
            "test/default_acc": test_metrics["default"]["acc"],
            "test/default_f1": test_metrics["default"]["f1"],
            "test/default_f1_macro": test_metrics["default"]["f1_macro"],
            "test/default_precision": test_metrics["default"]["precision"],
            "test/default_recall": test_metrics["default"]["recall"],

            "test/screening_auc": test_metrics["screening"]["auc"],
            "test/screening_ap": test_metrics["screening"]["ap"],
            "test/screening_acc": test_metrics["screening"]["acc"],
            "test/screening_f1": test_metrics["screening"]["f1"],
            "test/screening_f1_macro": test_metrics["screening"]["f1_macro"],
            "test/screening_precision": test_metrics["screening"]["precision"],
            "test/screening_recall": test_metrics["screening"]["recall"],

            "test/best_f1_from_val_auc": test_metrics["best_f1_from_val"]["auc"],
            "test/best_f1_from_val_ap": test_metrics["best_f1_from_val"]["ap"],
            "test/best_f1_from_val_acc": test_metrics["best_f1_from_val"]["acc"],
            "test/best_f1_from_val_f1": test_metrics["best_f1_from_val"]["f1"],
            "test/best_f1_from_val_f1_macro": test_metrics["best_f1_from_val"]["f1_macro"],
            "test/best_f1_from_val_precision": test_metrics["best_f1_from_val"]["precision"],
            "test/best_f1_from_val_recall": test_metrics["best_f1_from_val"]["recall"],
        },
    )

    if wandb_run is not None:
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.summary["best_val_loss"] = best_val
        wandb_run.summary["thr_bad"] = float(thr_bad)
        wandb_run.summary["thr_sound"] = float(thr_sound)
        wandb_run.summary["encoder_name"] = ENCODER_NAME
        wandb_run.summary["checkpoint_path"] = CKPT_PATH
        wandb_run.summary["config_path"] = args.config

    finish_wandb_if_needed(wandb_run)


if __name__ == "__main__":
    main()