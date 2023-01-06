from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import datasets
from torchvision.transforms import transforms
import random
import numpy as np
from PIL import Image
import os
import pandas as pd

def get_WB_data(dataset_dir):
    # getting all data
    metadata_df = pd.read_csv(os.path.join(dataset_dir, 'metadata.csv'))
    y_array = metadata_df['y'].values
    n_classes = 2
    confounder_array = metadata_df['place'].values

    filename_array = metadata_df['img_filename'].values
    split_array = metadata_df['split'].values
    split_dict = {
        'train': 0,
        'val': 1,
        'test': 2
    }
    return split_array, split_dict, n_classes, y_array, confounder_array, filename_array

class WB_MultiDomainDatasetTripleFD(Dataset):
    def __init__(self, dataset_dir, train_split, subsample=1, bd_aug=False):
        self.dataset_dir = dataset_dir
        self.train_split = train_split
        self.subsample = subsample
        self.bd_aug = bd_aug

        r = get_WB_data(dataset_dir)
        self.split_array = r[0] 
        self.split_dict = r[1] 
        self.n_classes = r[2]
        self.y_array = r[3]
        self.confounder_array = r[4]
        self.filename_array = r[5]

        self.augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.all_data, self.all_cate = self.make_dataset()

    def make_dataset(self):
        all_data = []
        for cnt in range(self.y_array.shape[0]):
            if self.split_array[cnt] == self.train_split:
                all_data.append([os.path.join(self.dataset_dir, 
                                              self.filename_array[cnt]), 
                                              self.y_array[cnt]])
        all_cate = [[], []]
        for d in all_data:
            each, id = d
            all_cate[id].append(each)

        return all_data, all_cate

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        img_path, label = self.all_data[index]
        img_x_orig = Image.open(img_path).convert('RGB')
        img_x = self.augment_transform(img_x_orig)

        img_xp_path = random.sample(self.all_cate[label], 1)[0]

        img_xp = Image.open(img_xp_path).convert('RGB')
        img_xp = self.transform(img_xp)

        return img_x, img_xp, label

class WaterBirdsDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str = './data/',
                 train_val_split: Tuple[int, int, int] = (40000, 10000),
                 batch_size: int = 64,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 distill: str = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir


    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = WB_MultiDomainDatasetTripleFD(self.data_dir)
            # self.data_val =
            # self.data_test = 

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
