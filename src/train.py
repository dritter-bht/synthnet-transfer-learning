"""The main run script."""

from typing import List, Optional, Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger

import utils
from datamodules.baseline_finetune_dm import BaselineFinetuneDM
from datamodules.generic_finetune_dm import GenericFinetuneDM

# pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig):
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, num_classes=datamodule.num_classes)
    # TODO: Add model = torch.compile(model) when supported for python 3.11+

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }
    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        # log.info("Validating loaded model before training!")
        # trainer.validate(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


# python src/train.py --config-dir configs --config-name train.yaml
@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
