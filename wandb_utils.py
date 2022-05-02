import logging
from datetime import datetime

import wandb

from utils import retry


def init_wandb(cfg):
    """
    Must call initialization of Wandb before summary writer is initialized, otherwise
    sync_tensorboard does not work.
    """

    if not cfg.with_wandb:
        logging.info('Weights and Biases integration disabled')
        return

    if 'wandb_unique_id' not in cfg:
        # if we're going to restart the experiment, this will be saved to a json file
        cfg.wandb_unique_id = f'{cfg.domain_name}_{cfg.task_name}_{cfg.replay_episodes}eps_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'

    logging.info(f'Weights and Biases integration enabled. Project: {cfg.wandb_project}, user: {cfg.wandb_user}, group: {cfg.wandb_group}, unique_id: {cfg.wandb_unique_id}')

    # this can fail occasionally, so we try a couple more times
    @retry(3, exceptions=(Exception,))
    def init_wandb_func():
        wandb.init(
            project=cfg.wandb_project, entity=cfg.wandb_user, sync_tensorboard=True,
            id=cfg.wandb_unique_id,
            name=cfg.wandb_unique_id,
            group=cfg.wandb_group, job_type=cfg.wandb_job_type, tags=cfg.wandb_tags,
            # resume=True,
            resume=False,
            settings=wandb.Settings(start_method='fork'),
        )

    logging.info('Initializing WandB...')
    try:
        if cfg.wandb_key:
            wandb.login(key=cfg.wandb_key)
        init_wandb_func()
    except Exception as exc:
        logging.error(f'Could not initialize WandB! {exc}')

    wandb.config.update(cfg, allow_val_change=True)


def finish_wandb(cfg):
    if cfg.with_wandb:
        import wandb
        wandb.run.finish()
