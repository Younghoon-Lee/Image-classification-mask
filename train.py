import os
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
from torchvision.transforms.transforms import ColorJitter, RandomHorizontalFlip, RandomRotation
from tqdm import tqdm

from dataset import MyDataset
from models import MyModel

def train(model, num_epoch, optimizer, loss_fn, train_loader, device=torch.device('cuda')):
    model.train()
    for epoch in range(num_epoch):

        running_loss = 0.
        running_acc = 0.

        for idx , (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            _, preds = torch.max(logits, 1)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_acc += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_acc / len(train_loader.dataset)

        print(f"현재 epoch-{epoch}의 train-데이터 셋에서 평균 Loss : {epoch_loss:.3f}, 평균 Accuracy : {epoch_acc:.3f}")
    
    return model