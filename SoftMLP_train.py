import models.MLP.model as MLP
import models.MLP.experiments as experiments
import models.MLP.layers as layers

#####################################
# Progress Timing and Logging
# ----------------------------
# Important milestones in progress through this script are logged using a timing utility which also includes details of
# when milestones were reached and the elapsedtime between milestones. This can assist with debugging and optimisation.

from cycling_utils import TimestampedTimer

import models.hyperparams

timer = TimestampedTimer("Imported TimestampedTimer")

import argparse
import os
from operator import itemgetter
from pathlib import Path
import json


import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import Compose, Lambda, PILToTensor, RandAugment
from torch.utils.data import Subset

from cycling_utils import (
    MetricsTracker,
    atomic_torch_save,
)

import torch
import numpy as np
import torch.nn as nn 
import math
import matplotlib.pyplot as plt
import itertools
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from enum import Enum
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset
import pandas as pd
from skimage.util import random_noise
import imgaug as ia
import imgaug.augmenters as iaa
from torch.nn import functional as f
from matplotlib import pyplot

ia.seed(2024)

timer.report("Completed imports")

class InterruptableSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset
    ) -> None:
        super().__init__(dataset)
        self.epoch = 0
        self.progress = 0
        self.dataset = dataset

    def state_dict(self):
        return {"progress": self.progress, "epoch": self.epoch}

    def load_state_dict(self, state_dict):
        self.progress = state_dict["progress"]
        self.epoch = state_dict["epoch"]
    
    def advance(self, n):
        self.progress += n
    
    def set_epoch(self, e):
        self.epoch = e
    
    def __iter__(self):
        # Return indices in sequential order
        indices = list(range(len(self.dataset)))
        for idx in indices[self.progress :]:
            yield idx
        #return iter(range(len(self.dataset)))
    
    def __len__(self):
        return len(self.dataset)
    
    def reset_progress(self):
        self.progress = 0

#######################################
# Hyperparameters
# -----------------
# Hyperparameters are adjustable parameters that let you control the model optimization process. Different
# hyperparameter values can impact model training and convergence rates (`read more <https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html>`
# about hyperparameter tuning). We define the following hyperparameters for training:
#  - **Number of Epochs** - the number times to iterate over the dataset
#  - **Batch Size** - the number of data samples propagated through the network before the parameters are updated
#  - **Learning Rate** - how much to update models parameters at each batch/epoch. Smaller values yield slow learning
#       speed, while large values may result in unpredictable behavior during training.

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser()
    # ---------------------------------------
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--finetuneepochs", type=int, default=1)
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


def main(args, timer):
    ##############################################
    # Distributed Training Configuration
    # -----------------
    # The following steps demonstrate configuration of the local GPU for distributed training. This includes
    # initialising the process group, obtaining and setting the rank of the GPU within the cluster and on the local
    # node.

    dist.init_process_group("nccl")  # Expects RANK set in environment variable
    rank = int(os.environ["RANK"])  # Rank of this GPU in cluster
    # world_size = int(os.environ["WORLD_SIZE"]) # Total number of GPUs in the cluster
    args.device_id = int(os.environ["LOCAL_RANK"])  # Rank on local node
    args.is_master = rank == 0  # Master node for saving / reporting
    torch.cuda.set_device(args.device_id)  # Enables calling 'cuda'
    device = torch.cuda.current_device()
    timer.report("Setup for distributed training")

    args.checkpoint_path = args.save_dir / "checkpoint.pt"
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    timer.report("Validated checkpoint path")

    ##############################################
    # Data Transformation and Augmentation
    # ----------------------
    # Training data often requires pre-processing to ensure it is suitable for training, for example converting from a
    # PIL image to a Pytorch Tensor. This is also an opporuntiy to apply random perturbations to the training data each
    # time it is obtained from the dataset which has the effect of augmenting the size of the dataset and reducing
    # over-fitting.
    
    transform_MNIST = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))
        ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_MNIST)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_MNIST)

    timer.report("Initialized datasets")

    train_dataloader = DataLoader(train_dataset, batch_size=64)
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    timer.report("Initialized dataloaders")

    model = MLP.NewMLPBaseline_Model(hsize=64, lamb=5, w_lr=0.01, b_lr=0.01, l_lr=0.03, nclasses=10, device=device)
    # Set greedytrain = true for layer wise training 
    model = experiments.SoftMLPBaseline_Experiment(1, model, train_dataloader, 'MNIST', 10, device, greedytrain=False)


    print("\nTesting the model...")
    model.eval()  
    correct = 0
    total = 0
    with torch.no_grad(): 
        for data in test_dataloader:  
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model.forward(inputs)  
            _, predicted = torch.max(outputs.data, 1) 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on the test set: {accuracy:.2f}%")

    # Save the accuracy to a file in the output directory
    accuracy_file_path = args.save_dir / "accuracy.txt"
    with open(accuracy_file_path, "w") as f:
        f.write(f"Accuracy on the test set: {accuracy:.2f}%\n")

    print("Done!")

if __name__ == "__main__":
    args = get_args_parser().parse_args()

    main(args, timer)
