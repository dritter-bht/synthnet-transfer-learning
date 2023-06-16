"""A generic DataModule for fine-tuning."""

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from datamodules.dataloaders import MultiDataParallelLoader


class GenericAdaptationDM(pl.LightningDataModule):
    def __init__(
        self,
        train_src_dirs: list = [],
        train_target_dirs: list = [],
        val_dirs: list = [],
        test_dirs: list = [],
        batch_size: int = 64,
        num_workers: int = 4,
        toy: bool = False,
        image_size_w: int = 224,
        image_size_h: int = 224,
        image_mean: list = [0.5, 0.5, 0.5],
        image_std: list = [0.5, 0.5, 0.5],
        resize: bool = False,
        random_resized_crop: bool = True,
        center_crop: bool = False,
        random_horizontal_flip: bool = True,
        random_vertical_flip: bool = False,
        random_color_jitter: bool = False,
        random_grayscale: bool = False,
        augmix: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_src_dirs = train_src_dirs
        self.train_target_dirs = train_target_dirs
        self.val_dirs = val_dirs
        self.test_dirs = test_dirs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.toy = toy
        self.image_size_w = image_size_w
        self.image_size_h = image_size_h
        self.image_size = (image_size_h, image_size_w)
        self.image_mean = image_mean
        self.image_std = image_std
        self.resize = resize
        self.random_resized_crop = random_resized_crop
        self.center_crop = center_crop
        self.random_horizontal_flip = random_horizontal_flip
        self.random_vertical_flip = random_vertical_flip
        self.random_color_jitter = random_color_jitter
        self.random_grayscale = random_grayscale
        self.augmix = augmix

        self.train_transform = transforms.Compose(
            [
                transforms.RandomApply([transforms.Resize(self.image_size)], p=int(self.resize)),
                transforms.RandomApply(
                    [transforms.RandomResizedCrop(self.image_size, scale=(0.7, 1.0))], p=int(self.random_resized_crop)
                ),
                transforms.RandomApply([transforms.CenterCrop(self.image_size)], p=int(self.center_crop)),
                transforms.RandomApply([transforms.RandomHorizontalFlip()], p=int(self.random_horizontal_flip)),
                transforms.RandomApply([transforms.RandomVerticalFlip()], p=int(self.random_vertical_flip)),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)],
                    p=int(self.random_color_jitter),
                ),
                transforms.RandomApply([transforms.RandomGrayscale()], p=int(self.random_grayscale)),
                transforms.RandomApply([transforms.AugMix()], p=int(self.augmix)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.image_mean, std=self.image_std),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.image_mean, std=self.image_std),
            ]
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_src = [ImageFolder(train_src_dir) for train_src_dir in self.train_src_dirs]
        self.train_target = [ImageFolder(train_target_dir) for train_target_dir in self.train_target_dirs]
        self.val = [ImageFolder(val_dir) for val_dir in self.val_dirs]
        self.test = [ImageFolder(test_dir) for test_dir in self.test_dirs]

        for train_src_ds in self.train_src:
            train_src_ds.transform = self.train_transform
        for train_target_ds in self.train_target:
            train_target_ds.transform = self.val_transform
        for val_ds in self.val:
            val_ds.transform = self.val_transform
        for test_ds in self.test:
            test_ds.transform = self.val_transform

        # TODO: FIX and use merged class from all train_src datasets
        self.num_classes = len(self.train_src[0].classes)
        self.label2idx = self.train_src[0].class_to_idx
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

        # If toy is set true, use a very small subset of the data just for testing
        # Use 80 samples for training and 20 for testing
        if self.toy:
            self.train_src = [
                Subset(train_src_ds, np.random.choice(np.arange(len(train_src_ds)), size=80, replace=False))
                for train_src_ds in self.train_src
            ]
            self.train_target = [
                Subset(train_target_ds, np.random.choice(np.arange(len(train_target_ds)), size=80, replace=False))
                for train_target_ds in self.train_target
            ]
            self.val = [
                Subset(val_ds, np.random.choice(np.arange(len(val_ds)), size=80, replace=False)) for val_ds in self.val
            ]
            self.test = [
                Subset(test_ds, np.random.choice(np.arange(len(test_ds)), size=80, replace=False))
                for test_ds in self.test
            ]

    def train_dataloader(self):
        # TODO: Fix when MultiDataConcatLoader is implemented
        train_src_loader = DataLoader(
            dataset=self.train_src[0], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True
        )
        train_target_loader = DataLoader(
            dataset=self.train_target[0], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True
        )
        dataloaders = [train_src_loader, train_target_loader]
        n_batches = len(train_src_loader)
        return MultiDataParallelLoader(dataloaders=dataloaders, n_batches=n_batches)

    def val_dataloader(self):
        return DataLoader(dataset=self.val[0], batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test[0], batch_size=self.batch_size, num_workers=self.num_workers)
