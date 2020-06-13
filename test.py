import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import os
import numpy as np
import json
import itertools

gen = (x for x in range(10))
index = 5
for i in range(3, 5):
    print(i)
'''
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD)
])

traindata = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform_train)

trainloader = DataLoader(traindata, batch_size=128, shuffle=True, num_workers=2)

print(len(trainloader))
print(len(trainloader.dataset))
'''