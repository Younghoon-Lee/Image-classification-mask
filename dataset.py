import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, train_path,transform=None):
        self.train_path = os.path.join(train_path, 'images')
        self.transform = transform
        self.csv = pd.read_csv(os.path.join(train_path, 'train_labeled.csv'))
    def __getitem__(self, index):
        folder_path = os.path.join(self.train_path,self.csv.iloc[index]['path'])
        file_path = os.path.join(folder_path, self.csv.iloc[index]['file_name'])
        image = Image.open(file_path)
        label = self.csv.iloc[index]['label']
        if self.transform:
            image = self.transform(image)

        return image , label

    def __len__(self):
        return len(self.csv)

class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
