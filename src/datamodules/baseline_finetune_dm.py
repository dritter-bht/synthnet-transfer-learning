"""DataModule interpreted from A Broad Study of Pre-training for Domain Generalization and Adaptation
DOI:10.48550/arXiv.2203.11819 https://www.semanticscholar.org/paper/A-Broad-Study-of-Pre-training-for-Domain-and-Kim-
Wang/e0bffb70cd8b5b5ecdc74e1f730dd7298ecc787b https://github.com/VisionLearningGroup/Benchmark_Domain_Transfer.

@InProceedings{kim2022unified,
    title={A Broad Study of Pre-training for Domain Generalization and Adaptation},
    author={Kim, Donghyun and Wang, Kaihong and Sclaroff, Stan and Saenko, Kate},
    booktitle = {The European Conference on Computer Vision (ECCV)},
    year = {2022}
}

We took info about augmentations, dataset split, and other hardcoded parameters either from the paper
or (if not specified) from their code (e.g. batch_size, seed, transforms params)
"""
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


class BaselineFinetuneDM(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        test_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        toy: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.toy = toy

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train, self.val = random_split(ImageFolder(self.train_dir), [0.8, 0.2])
        self.train = self.train.dataset
        self.val = self.val.dataset
        self.test = ImageFolder(self.test_dir)

        self.train.transform = self.train_transform
        self.val.transform = self.val_transform
        self.test.transform = self.val_transform
        self.num_classes = len(self.train.classes)
        self.label2idx = self.train.class_to_idx
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

        # If toy is set true, use a very small subset of the data just for testing
        # Use 80 samples for training and 20 for testing
        if self.toy:
            self.train = Subset(self.train, np.random.choice(np.arange(len(self.train)), size=80, replace=False))
            self.val = Subset(self.val, np.random.choice(np.arange(len(self.val)), size=80, replace=False))
            self.test = Subset(self.test, np.random.choice(np.arange(len(self.test)), size=80, replace=False))

    def train_dataloader(self):
        return DataLoader(dataset=self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
