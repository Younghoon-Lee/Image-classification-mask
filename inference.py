import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from dataset import TestDataset

def inference(model, test_loader, test_dir,submission, device):
    model.eval()
    
    all_predictions =[]
    for images in test_loader:
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())
    submission['ans'] = all_predictions

    submission.to_csv(os.path.join(test_dir, 'submission.csv'),index=False)
    print('test inference is done!')