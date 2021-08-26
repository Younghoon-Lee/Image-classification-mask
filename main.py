import os
import pandas as pd
import torch
from torch.nn.modules import loss
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

from dataset import MyDataset, TestDataset
from models import MyModel
from train import train
from inference import inference

def main():

    TRAIN_DATA_PATH = 'train'
    TEST_DATA_PATH = 'eval'
    LEARNING_RATE = 0.0001
    NUM_EPOCH = 5
    device = torch.device('cuda')

    transform = transforms.Compose([
        Resize((512,384), Image.BILINEAR),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    dataset = MyDataset(TRAIN_DATA_PATH, transform)

    train_loader = DataLoader(
        dataset,
        shuffle=True,
        batch_size= 64,
        num_workers=2
    )

    resnet18 = torchvision.models.resnet18(pretrained=True)
    resnet18.fc = torch.nn.Linear(in_features=512, out_features= 18, bias =True)
    nn.init.xavier_uniform_(resnet18.fc.weight)
    stdv = 1/np.sqrt(512)
    resnet18.fc.bias.data.uniform_(-stdv,stdv)
    resnet18.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet18.parameters(), lr=LEARNING_RATE)

    trained_model = train(
        resnet18, NUM_EPOCH, optimizer, loss_fn, train_loader, device
    )

    testDataset = TestDataset(os.path.join(TEST_DATA_PATH, 'images'), transform)

    test_loader = DataLoader(
        testDataset,
        shuffle=False
    )


    inference(trained_model, test_loader, TEST_DATA_PATH, device)