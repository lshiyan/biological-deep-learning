import models.MLP.model as MLP
import models.CNN.model as CNN
import importlib
import numpy as np
import argparse
from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm
import itertools
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F

transform_MNIST = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))
        ])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_MNIST)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_MNIST)

train_dataloader = DataLoader(train_dataset, batch_size=1)
test_dataloader = DataLoader(test_dataset, batch_size=1)

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser()
    # ---------------------------------------
    parser.add_argument("--dataset", type=str, default="EMNIST")
    parser.add_argument("--epochs", type=int, default=1)
    # The number of times to loop over the whole dataset
    # ---------------------------------------
    parser.add_argument("--test-epochs", type=int, default=1)
    # Testing model performance on a test every "test-epochs" epochs
    # ---------------------------------------
    parser.add_argument("--topdown", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--hsize", type=int, default=64)
    # ---------------------------------------
    parser.add_argument("--lambd", type=float, default=15)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--rho", type=int, default=10)
    # For distributed training it is important to distinguish between the per-GPU or "local" batch size (which this
    # hyper-parameter sets) and the "effective" batch size which is the product of the local batch size and the number
    # of GPUs in the cluster. With a local batch size of 16, and 10 nodes with 6 GPUs per node, the effective batch size
    # is 960. Effective batch size can also be increased using gradient accumulation which is not demonstrated here.
    # ---------------------------------------
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--tboard-path", type=Path, required=True)
    # While the "output_path" is set in the .isc file and receives reports from each node in the cluster (i.e.
    # "rank_0.txt"), the "save-dir" will be where this script saves and retrieves checkpoints. Referring to the .isc
    # file, you will see that in this case, the save-dir is the same as the output_path. Tensoboard event logs will be
    # saved in the "tboard-path" directory.
    return parser




if __name__ == "__main__":
    #args = get_args_parser().parse_args()

    model = CNN.CNNBaseline_Model(inputshape=(1, 28, 28), kernels=[5, 3, 3], channels=[16, 64, 256], strides=[2, 2, 2],
                                  padding=[0, 1, 0], lambd=4, lr=0.005, gamma=0.99, epsilon=0.01,
                                  rho=0.001, nbclasses=10, topdown=False, device="cpu",
                                  learningrule=CNN.Learning.FullyOrthogonal,
                                  weightscaling=CNN.WeightScale.WeightNormalization,
                                  outputlayerrule=CNN.ClassifierLearning.Contrastive)
    mymodelCNN = CNN.CNNBaseline_Experiment(epoch=1, mymodel=model, dataloader=train_dataloader, dataset='FashionMNIST',
                                      nclasses=10, imgtype=CNN.ImageType.Gray)
    print(CNN.CNN_Baseline_test(mymodel=mymodelCNN, data_loader=test_dataloader, imgtype=CNN.ImageType.Gray,
                                topdown=True))