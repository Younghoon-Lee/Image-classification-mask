import os

import torch
from torch import nn, optim

import torchvision
import torchvision.transforms as T

from sklearn import model_selection
import pandas as pd
from tqdm import tqdm

import config_parser
import data_utils
import models
import logger

config = config_parser.ConfigParser(description='mask classification learner')
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

train_root_dir = os.path.join(config.data_dir, config.train_dir_name)
train_meta_data = pd.read_csv(os.path.join(train_root_dir, 'train.csv')).drop(columns=['id', 'race']).to_numpy()

img_paths, labels = data_utils.get_paths_and_labels(
    root_dir=train_root_dir,
    meta_data=train_meta_data
)
train_img_paths, val_img_paths, train_labels, val_labels = model_selection.train_test_split(
    img_paths, labels, 
    test_size=0.2,
    shuffle=True, 
    stratify=labels
)

transform = T.Compose([
    T.Resize((512, 384), T.InterpolationMode.BICUBIC),
    T.ToTensor(),
])
train_dataset = data_utils.MaskClassifierDataset(
    img_paths=train_img_paths,
    labels=train_labels,
    transform=transform
)
val_dataset = data_utils.MaskClassifierDataset(
    img_paths=val_img_paths,
    labels=val_labels
)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, drop_last=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, drop_last=False)

model = models.MaskClassifierModel(num_classes=18).to(device)
criterion = nn.MultiLabelSoftMarginLoss().to(device)
optimizer = optim.Adam(params=model.parameters(), lr=config.learning_rate)

for epoch in range(1, config.n_epochs + 1):
    running_loss = 0
    n_corrects = 0
    for imgs, labels in tqdm(train_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        predictions = model(imgs)
        labels_one_hot = torch.zeros_like(predictions).scatter_(1, labels, 1)
        loss = criterion(predictions, labels_one_hot)

        running_loss += loss.item()
        n_correct = (predictions.argmax(dim=1).unsqueeze(dim=1) == labels).float().sum(dim=0).item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'epoch: {epoch:02d}/{config.n_epochs}\tcorrect: {(n_correct / config.batch_size) * 100:0.2f}%\tloss: {running_loss / config.batch_size:0.3f}')