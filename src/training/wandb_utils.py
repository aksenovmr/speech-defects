def init_wandb_if_needed(
    use_wandb: bool,
    project: str,
    config: dict,
    run_name: str | None = None,
    group: str | None = None,
    tags: list[str] | None = None,
):
    if not use_wandb:
        return None

    try:
        import wandb
    except ImportError:
        print("wandb not installed, skipping logging")
        return None

    init_kwargs = {
        "project": project,
        "config": config,
    }

    if run_name is not None:
        init_kwargs["name"] = run_name

    if group is not None:
        init_kwargs["group"] = group

    if tags is not None:
        init_kwargs["tags"] = tags

    run = wandb.init(**init_kwargs)
    return run


def log_wandb_if_needed(run, metrics: dict):
    if run is None:
        return
    try:
        import wandb
        wandb.log(metrics)
    except Exception:
        pass


def finish_wandb_if_needed(run):
    if run is None:
        return
    try:
        import wandb
        wandb.finish()
    except Exception:
        pass