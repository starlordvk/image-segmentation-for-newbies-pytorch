import os
import os.path as osp
import numpy as np
import math

from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms


# !!!!!!!! NO NEED TO WRITE CUSTOM DATASET CLASSES FOR CITY SCAPES. ALL HAIL PYTORCH !!!!!
# Torch vision now supports city scapes and allows to easily load the cityscapes dataset
# by just specifying the directory which has the leftImg8bit and gtFine folders.

IMG_HEIGHT = 110  
IMG_WIDTH = 220

tfms = transforms.Compose([transforms.Resize((IMG_HEIGHT, IMG_WIDTH),interpolation=Image.NEAREST), transforms.PILToTensor()])

trainDataset = torchvision.datasets.Cityscapes('./', split='train', mode='fine',
                     target_type='semantic', transform=tfms, target_transform=tfms)

valDataset = torchvision.datasets.Cityscapes('./', split='val', mode='fine',
                     target_type='semantic', transform=tfms, target_transform=tfms)

testDataset = torchvision.datasets.Cityscapes('./', split='test', mode='fine',
                     target_type='semantic', transform=tfms, target_transform=tfms)
