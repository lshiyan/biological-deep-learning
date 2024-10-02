# from new_test.ipynb

import models.CNN.model as CNN
import importlib
import numpy as np
import argparse
from pathlib import Path
import torch
import json
from models.hyperparams import ImageType, LearningRule, WeightScale, Inhibition, oneHotEncode
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm
import itertools
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import models.learning as L

transform = transforms.Compose(
    [transforms.ToTensor()])

train_dataset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=1)
test_dataloader = DataLoader(test_dataset, batch_size=1)


topdown = False

import importlib
importlib.reload(L)
importlib.reload(CNN)


with open("COnfigs/config0.json", "r") as file:
        config = json.load(file)


#mymodelCNN = CNN.CNN_Model_from_config((3,32,32), config, 'cpu', 10)
mymodelCNN = CNN.CNN_Model_SoftHeb_from_config((3,32,32), config, 'cpu', 10)
mymodelCNN, mymodel = CNN.CNN_Experiment(epoch=1, mymodel=mymodelCNN, dataloader=train_dataloader, testloader= test_dataloader,
                                         dataset='CIFAR10',
                                      nclasses=10, imgtype=CNN.ImageType.RGB, traintopdown=topdown, testtopdown=topdown, device=torch.device('cpu'), greedytrain=config['greedytrain'])
# print(CNN.CNN_Baseline_test(mymodel=mymodel, data_loader=test_dataloader, imgtype=CNN.ImageType.RGB,
#                                 topdown=topdown))


