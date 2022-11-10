from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import datasets
from torchvision.transforms import transforms
import random
import numpy as np
from PIL import Image

# anisio -> from RepDistiller library
# changes from https://github.com/HobbitLong/RepDistiller/issues/38

class CIFAR100InstanceSample(datasets.CIFAR100):
    """
    CIFAR100Instance+Sample Dataset
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', 
                 is_sample=True, percent=1.0, idxs=None):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 100
        if self.train:
            num_samples = len(self.data)
            label = self.targets
        else:
            num_samples = len(self.data)
            label = self.targets

        self.cls_positive = [[] for i in range(num_classes)]
        # for i in range(num_samples):
        for i in idxs:
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

        # print(f'\n\n\n==>>{len(self.cls_positive[0])}\n{len(self.cls_negative[0])}')


    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
            
class CIFAR100DataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str = './data/',
                 train_val_split: Tuple[int, int, int] = (40000, 10000),
                 batch_size: int = 64,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 distill: str = None,
                 mode: str = None,
                 nce_k: int = None,
                 is_sample: bool = None,
                 percent: float = None,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.distill = distill

    def prepare_data(self):
        datasets.CIFAR100(self.hparams.data_dir, train=True, download=True)
        datasets.CIFAR100(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            if self.distill != 'crd':
                trainset = datasets.CIFAR100(self.hparams.data_dir,
                                             train=True, 
                                             transform=self.train_transform)
                self.data_train, self.data_val = random_split(
                    dataset=trainset,
                    lengths=self.hparams.train_val_split,
                    generator=torch.Generator().manual_seed(42),
                )
                self.data_test = datasets.CIFAR100(self.hparams.data_dir, train=False, transform=self.test_transform)
            else:
                train_size = sum(self.hparams.train_val_split)
                all_data_idxs = [i for i in range(train_size)]
                random.shuffle(all_data_idxs)
                train_idxs = all_data_idxs[:self.hparams.train_val_split[0]]
                val_idxs = all_data_idxs[self.hparams.train_val_split[0]:]
                # print(f'\n\n\n{train_size}\n{len(train_idxs)}\n{len(val_idxs)}')
                # print(f'\n\n==>{set(train_idxs).intersection(set(val_idxs))}')

                self.data_train = CIFAR100InstanceSample(root=self.hparams.data_dir,
                                                  download=True,
                                                  train=True,
                                                  transform=self.train_transform,
                                                  k=self.hparams.nce_k,
                                                  mode=self.hparams.mode,
                                                  is_sample=self.hparams.is_sample,
                                                  percent=self.hparams.percent,
                                                  idxs=train_idxs)
            
                self.data_val = CIFAR100InstanceSample(root=self.hparams.data_dir,
                                                        download=True,
                                                        train=True,
                                                        transform=self.train_transform,
                                                        k=self.hparams.nce_k,
                                                        mode=self.hparams.mode,
                                                        is_sample=self.hparams.is_sample,
                                                        percent=self.hparams.percent,
                                                        idxs=val_idxs)
            
                test_idxs = [i for i in range(10000)]
                self.data_test = CIFAR100InstanceSample(root=self.hparams.data_dir,
                                                        download=True,
                                                        train=False,
                                                        transform=self.train_transform,
                                                        k=self.hparams.nce_k,
                                                        mode=self.hparams.mode,
                                                        is_sample=self.hparams.is_sample,
                                                        percent=self.hparams.percent,
                                                        idxs=test_idxs)
                
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "cifar100.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)

