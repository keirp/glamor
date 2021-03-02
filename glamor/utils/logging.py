_use_wandb = True
_run_path = None


def disable_wandb(run_path):
    global _use_wandb
    global _run_path
    _use_wandb = False
    _run_path = run_path


def wandb_is_enabled():
    global _use_wandb
    return _use_wandb


def run_path():
    global _run_path
    return _run_path


def set_run_path(run_path):
    global _run_path
    _run_path = run_path
