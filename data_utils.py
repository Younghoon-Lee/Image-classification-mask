import os

import torch
from torch.utils.data import Dataset

import numpy as np
from PIL import Image

class MaskClassifierDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
    
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[index]).unsqueeze(dim=0)

    def __len__(self):
        return len(self.labels)

def get_paths_and_labels(root_dir, meta_data):
    img_paths = []
    labels = []
    label_dict = {'male': 0, 'female': 3}
    for gender, age, img_dir_name in meta_data:
        _label = label_dict[gender] + (age // 30)
        for file_name in os.listdir(img_dir := os.path.join(root_dir, 'images', img_dir_name)):
            if file_name.startswith('.'):
                continue
            label = _label
            if file_name.startswith('incorrect'):
                label += 6
            elif file_name.startswith('normal'):
                label += 12
            img_paths.append(os.path.join(img_dir, file_name))
            labels.append(label)
    return img_paths, labels

