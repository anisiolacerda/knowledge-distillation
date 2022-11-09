import os
import torch
from torchvision import transforms
import pandas as pd
from PIL import Image
import random

def get_data_folder():
    """
    returns path to store the data
    """
    #TODO(anisio): receive data folder from flags
    data_folder = './data/waterbird_complete95_forest2water2'
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    return data_folder

def get_waterbird_dataloaders():
    return None

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

class WB_MultiDomainDatasetTripleFD(torch.utils.data.Dataset):
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


def get_train_loader(dataset_dir, batch_size, num_workers):
    train_dataset = WB_MultiDomainDatasetTripleFD(dataset_dir)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader

class WB_DomainTest(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, split, subsample=1, group=[0,0]) -> None:
        # super().__init__()
        self.dataset_dir = dataset_dir
        self.split = split
        self.subsample = subsample
        self.group = group

        r = get_WB_data(dataset_dir)
        self.split_array = r[0] 
        self.split_dict = r[1] 
        self.n_classes = r[2]
        self.y_array = r[3]
        self.confounder_array = r[4]
        self.filename_array = r[5]

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.all_data = self.make_dataset()

    def make_dataset(self):
        all_data = []
        cnt = 0
        if self.split == 'val':
            flag = 1
        elif self.split == 'test':
            flag = 2
        else:
            raise NotImplementedError(self.split)
        
        for cnt in range(self.y_array.shape[0]):
            if self.split_array[cnt] == flag:
                if self.y_array[cnt] == self.group[0] and self.confounder_array[cnt] == self.group[1]:
                    all_data.append([os.path.join(self.dataset_root_dir, self.filename_array[cnt]), self.y_array[cnt]])

        return all_data

    def __getitem__(self, index):
        img_path, label = self.all_data[index]
        img_x = Image.open(img_path).convert("RGB")
        img_x = self.transform(img_x)
        return img_x, label

    def __len__(self):
        return len(self.all_data)

def get_waterbird_dataloaders(batch_size=128, num_workers=8, is_instance=False):
    """"
    waterbird dataset
    """
    data_folder = get_data_folder()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])


    train_loader = get_train_loader(data_folder, batch_size, num_workers)
    val_loader = get_val_loader()

    return train_loader, val_loader

    




