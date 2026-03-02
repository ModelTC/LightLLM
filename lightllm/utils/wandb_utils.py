import wandb

global _wandb_run
_wandb_run = None


def get_wandb_run():
    global _wandb_run
    if _wandb_run is None:
        _wandb_run = wandb.init(
            entity="wzjhelloworld-personal",
            project="llm_inner_allalysis",
            name="qwen3-8b",
            id="qwen3-8b11",
            dir=str("./wandb/"),
            job_type="eval",
            reinit=True,
            force=True,
        )
    return _wandb_run
