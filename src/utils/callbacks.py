"""Callbacks for pytorch lightning trainer."""

import numpy as np
import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from torch.utils.data import DataLoader

import utils
from utils.transforms import UnNormalize

log = utils.get_pylogger(__name__)


class FreezeAllButLast(BaseFinetuning):
    def __init__(self):
        super().__init__()

    def freeze_before_training(self, pl_module):
        # freeze any module you want
        for name, param in pl_module.net.named_parameters():
            param.requires_grad = False

        # TODO: Make generic or add model name to know how to select last layers
        # TODO: Make generic or add model name to know how to select last layers
        # For most models. Ohne classification layer named classifier
        if hasattr(pl_module.net, "classifier"):
            pl_module.net.classifier.weight.requires_grad = True
            pl_module.net.classifier.bias.requires_grad = True
        # Deit: Distillation and CLS classifier
        if hasattr(pl_module.net, "cls_classifier"):
            pl_module.net.cls_classifier.weight.requires_grad = True
            pl_module.net.cls_classifier.bias.requires_grad = True
        # Deit: Distillation and CLS classifier
        if hasattr(pl_module.net, "cls_classifier"):
            pl_module.net.distillation_classifier.weight.requires_grad = True
            pl_module.net.distillation_classifier.bias.requires_grad = True

        for name, param in pl_module.net.named_parameters():
            log.debug(f"{name}: requires_grad={param.requires_grad}")

    def finetune_function(self, pl_module, epoch, optimizer) -> None:
        pass


class LogPredictionSamplesCallback(Callback):
    def __init__(self, n: int = 4):
        super().__init__()
        self.n = n

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from the first batch
        if batch_idx == 0:
            x, y = batch
            images = [img for img in x[: self.n]]
            idx2label = trainer.datamodule.idx2label
            captions = [
                f"gt: {idx2label[y_i.item()]} | pred: {idx2label[pred_i.item()]}"
                for y_i, pred_i in zip(y[: self.n], outputs["preds"][: self.n])
            ]
            trainer.logger.experiment.log(
                {"prediction_samples": [wandb.Image(img, caption=cap) for img, cap in zip(images, captions)]},
                commit=False,
            )
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)


class LogTrainingSamplesCallback(Callback):
    def __init__(self, n: int = 4):
        super().__init__()
        self.n = n

    def on_train_start(self, trainer, pl_module) -> None:
        dm = trainer.datamodule
        # original_images = [dm.train[i][0] for i in range(0, self.n)]
        # labels = [dm.idx2label[dm.train[i][1]] for i in range(0, self.n)]
        loader = DataLoader(dataset=dm.train, batch_size=self.n, num_workers=0, shuffle=True)
        samples = next(iter(loader))
        labels = [dm.idx2label[label_i.item()] for label_i in samples[1]]

        trainer.logger.experiment.log(
            {
                "transformed_training_samples": [
                    wandb.Image(img, caption=cap) for img, cap in zip(list(samples[0]), labels)
                ]
            },
            commit=False,
        )
        return super().on_train_start(trainer, pl_module)


class LogTrainingSamplesMultiDataParallelLoaderCallback(Callback):
    def __init__(self, n: int = 4):
        super().__init__()
        self.n = n

    def on_train_start(self, trainer, pl_module) -> None:
        dm = trainer.datamodule
        # TODO: Probably needs fix after MultiConcatDataLoader is implemented
        loader = DataLoader(dataset=dm.train_src[0], batch_size=self.n, num_workers=0, shuffle=True)
        samples = next(iter(loader))
        labels = [dm.idx2label[label_i.item()] for label_i in samples[1]]
        trainer.logger.experiment.log(
            {
                "transformed_training_samples_source": [
                    wandb.Image(img, caption=cap) for img, cap in zip(list(samples[0]), labels)
                ]
            },
            commit=False,
        )

        loader = DataLoader(dataset=dm.train_target[0], batch_size=self.n, num_workers=0, shuffle=True)
        samples = next(iter(loader))
        labels = [dm.idx2label[label_i.item()] for label_i in samples[1]]
        trainer.logger.experiment.log(
            {
                "transformed_training_samples_target": [
                    wandb.Image(img, caption=cap) for img, cap in zip(list(samples[0]), labels)
                ]
            },
            commit=False,
        )
        return super().on_train_start(trainer, pl_module)


class LogLayersRequiresGrad(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module) -> None:
        for name, param in pl_module.net.named_parameters():
            log.info(f"{name}: requires_grad={param.requires_grad}")
        return super().on_train_start(trainer, pl_module)
