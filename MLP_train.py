import models.MLP.model as MLP

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
import csv

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Sampler, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import Compose, Lambda, PILToTensor, RandAugment
from torch.utils.data import Subset
import matplotlib.pyplot as plt
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
        #self._has_reset_progress = True
        self.dataset = dataset

    # def _reset_progress(self):
    #     self.progress = 0
    #     self._has_reset_progress = True

    def state_dict(self):
        return {"progress": self.progress, "epoch": self.epoch}

    def load_state_dict(self, state_dict):
        self.progress = state_dict["progress"]
        self.epoch = state_dict["epoch"]
    
    def advance(self, n):
        self.progress += n
    
    def set_epoch(self, e):
        self.epoch = e

    # def advance(self, n: int):
    #     """
    #     Record that n samples have been consumed.
    #     """
    #     self.progress += n
    #     if self.progress > self.num_samples:
    #         raise AdvancedTooFarError(
    #             "You have advanced too far. You can only advance up to the total size of the dataset."
    #         )
    
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

    # def __iter__(self):
    #     if self.shuffle:
    #         # deterministically shuffle based on epoch and seed
    #         g = torch.Generator()
    #         g.manual_seed(self.seed + self.epoch)
    #         indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
    #     else:
    #         indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

    #     if not self.drop_last:
    #         # add extra samples to make it evenly divisible
    #         padding_size = self.total_size - len(indices)
    #         if padding_size <= len(indices):
    #             indices += indices[:padding_size]
    #         else:
    #             indices += (indices * math.ceil(padding_size / len(indices)))[
    #                 :padding_size
    #             ]
    #     else:
    #         # remove tail of data to make it evenly divisible.
    #         indices = indices[: self.total_size]
    #     assert len(indices) == self.total_size

    #     # subsample
    #     indices = indices[self.rank : self.total_size : self.num_replicas]
    #     assert len(indices) == self.num_samples

    #     # slice from progress to pick up where we left off

    #     for idx in indices[self.progress :]:
    #         yield idx

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
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--finetuneepochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=512)
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

#####################################
# We define ``train_loop`` that loops over our optimization code, and ``test_loop`` that evaluates the model's
# performance against our test data. Inside the training loop, optimization happens in three steps:
#  * Call ``optimizer.zero_grad()`` to reset the gradients of model parameters. Gradients by default add up; to prevent
#       double-counting, we explicitly zero them at each iteration.
#  * Backpropagate the prediction loss with a call to ``loss.backward()``. PyTorch deposits the gradients of the loss
#       w.r.t. each parameter.
#  * Once we have our gradients, we call ``optimizer.step()`` to adjust the parameters by the gradients collected in the
#       backward pass.

def log_weights(model):
    print("\nWeight Statistics:")
    for name, layer in model.layers.items():
        # Access weights within each layer
        if hasattr(layer, 'weight'):
            weight = layer.weight
        elif hasattr(layer, 'feedforward'):
            weight = layer.feedforward.weight
        else:
            continue  # Skip layers without weights

        print(f"Layer: {name}")
        print(f"  Mean: {weight.mean().item():.4f}")
        print(f"  Std: {weight.std().item():.4f}")
        print(f"  Min: {weight.min().item():.4f}")
        print(f"  Max: {weight.max().item():.4f}")



def visualize_weight_distribution(model, epoch, save_dir="/root/HebbianTopDown/weight_distributions"):
    os.makedirs(save_dir, exist_ok=True)
    for name, layer in model.layers.items():
        if hasattr(layer, 'weight'):
            weight = layer.weight.detach().cpu().numpy()
        elif hasattr(layer, 'feedforward'):
            weight = layer.feedforward.weight.detach().cpu().numpy()
        else:
            continue

        plt.figure()
        plt.hist(weight.flatten(), bins=50, alpha=0.75, color='blue', label='Weights')
        plt.title(f'Weight Distribution - Layer: {name} (Epoch: {epoch})')
        plt.xlabel('Weight Values')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)

        # Save the plot
        filename = os.path.join(save_dir, f"{name}_epoch_{epoch}.png")
        plt.savefig(filename)
        plt.close()




def train_loop(model, train_dataloader, test_dataloader, metrics, args, checkpoint):
    train_batches_per_epoch = len(train_dataloader)
    # Set the model to training mode - important for layers with different training / inference behaviour
    model.train()

    for epoch in range(args.epoch):
        train_dataloader.sampler.reset_progress()
        for inputs, targets in train_dataloader:

            # Determine the current batch
            batch = train_dataloader.sampler.progress // train_dataloader.batch_size
            is_last_batch = (batch + 1) == train_batches_per_epoch

            # Move input and targets to device
            inputs, targets = inputs.to(args.device_id), targets.to(args.device_id)
            timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - data to device")

            # Forward pass
            if args.dataset == "EMNIST":
                labels = models.hyperparams.oneHotEncode(targets, 47, args.device_id)
            elif args.dataset == "FASHION" or args.dataset == "FashionMNIST" or args.dataset == "MNIST" :
                labels = models.hyperparams.oneHotEncode(targets, 10, args.device_id)
            elif args.dataset == "CIFAR10":
                labels = models.hyperparams.oneHotEncode(targets, 10, args.device_id)
            elif args.dataset == "CIFAR100":
                labels = models.hyperparams.oneHotEncode(targets, 100, args.device_id)

            model.forward(inputs, labels)

            timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - forward pass")

            metrics["train"].update({"examples_seen": len(inputs)})
            metrics["train"].reduce()  # Gather results from all nodes - sums metrics from all nodes into local aggregate

            # Advance sampler - essential for interruptibility
            train_dataloader.sampler.advance(len(inputs))
            timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - advance sampler")

            metrics["train"].reset_local()

            if is_last_batch:
                metrics["train"].end_epoch()  # Store epoch aggregates and reset local aggregate for next epoch


            # total_progress = train_dataloader.sampler.progress + epoch * train_batches_per_epoch
            # Save checkpoint
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "train_sampler": train_dataloader.sampler.state_dict(),
                    "test_sampler": test_dataloader.sampler.state_dict(),
                    "metrics": metrics,
                },
                checkpoint,
            )
            timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - save checkpoint")

        print(f"Epoch {epoch} completed.")
        log_weights(model)
        visualize_weight_distribution(model, epoch)


def test_loop(model, train_dataloader, test_dataloader, metrics, args, checkpoint, hsize, lamb, w_lr, b_lr, l_lr, w_norm, anti_hebb_factor):
    test_batches_per_epoch = len(test_dataloader)
    epoch = 0
    # Set the model to evaluation mode - important for layers with different training / inference behaviour
    test_dataloader.sampler.reset_progress()
    model.eval()

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode also serves to
    # reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            # Determine the current batch
            batch = test_dataloader.sampler.progress // test_dataloader.batch_size
            is_last_batch = (batch + 1) == test_batches_per_epoch
            # Move input and targets to device
            inputs, targets = inputs.to(args.device_id), targets.to(args.device_id)
            timer.report(f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] - data to device")

            # Inference
            predictions = model.forward(inputs)

            timer.report(f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] - inference")

            # Performance metrics logging
            correct = (predictions.argmax(1) == targets).type(torch.float).sum()

            metrics["test"].update({"examples_seen": len(inputs), "correct": correct.item()})
            timer.report(f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] - metrics logging")

            # Advance sampler
            test_dataloader.sampler.advance(len(inputs))

            # Performance summary at the end of the epoch
            if is_last_batch:
                correct, examples_seen = itemgetter("correct", "examples_seen")(metrics["test"].local)
                pct_test_correct = correct / examples_seen
                metrics["test"].end_epoch()
                timer.report(
                    f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] :: TEST ACC: {pct_test_correct}"
                )
                print("Model " + str(int(os.environ["RANK"])) + " has testing accuracy of " + str(pct_test_correct))

                csv_file_path = "/root/HebbianTopDown/AntiHebb_MLP_hyper_search_Softmax/anti_hebb_results.csv"
                file_exists = os.path.isfile(csv_file_path)

                with open(csv_file_path, "a", newline="") as csvfile:
                    fieldnames = ["test_accuracy", "hsize", "lambda", "w_lr", "b_lr", "l_lr", "triangle", "white", "func", "w_norm", "anti_hebb_factor"]
                    
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    if not file_exists:
                        writer.writeheader()

                    writer.writerow({
                        "test_accuracy": pct_test_correct,
                        "hsize": hsize,
                        "lambda": lamb,
                        "w_lr": w_lr,
                        "b_lr": b_lr,
                        "l_lr": l_lr,
                        "triangle":"true",
                        "white":"true",
                        "func": "softmax",
                        "w_norm": w_norm,
                        "anti_hebb_factor": anti_hebb_factor
                    })

            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "train_sampler": train_dataloader.sampler.state_dict(),
                    "test_sampler": test_dataloader.sampler.state_dict(),
                    "metrics": metrics,
                },
                checkpoint,
            )
            
# timer.report("Defined helper function/s, loops, and model")


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

    dataset = args.dataset

    with open("ConfigsMLP/config" + str(rank) + ".json", "r") as file:
        config = json.load(file)
    
    if dataset == "MNIST" :
        transform_MNIST = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_MNIST)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_MNIST)

        timer.report("Initialized datasets")
        model = MLP.NewMLPBaseline_Model(hsize=config['hsize'], lamb=config['lambd'], w_lr=config['w_lr'], b_lr=config['b_lr'], l_lr=config['l_lr'], initial_weight_norm=config['w_norm'], nclasses=10, anti_hebb_factor=config['anti_hebb_factor'] , device=args.device_id)


    elif dataset == "FASHION":
        transform_MNIST = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_MNIST)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_MNIST)

        # Force data loading to ensure dataset is fully initialized
        _ = train_dataset.data, train_dataset.targets
        _ = test_dataset.data, test_dataset.targets

        timer.report("Initialized datasets")   
        model = MLP.NewMLPBaseline_Model(
            hsize=config['hsize'], lamb=config['lambd'], w_lr=config['w_lr'], 
            b_lr=config['b_lr'], l_lr=config['l_lr'], initial_weight_norm=config['w_norm'], 
            nclasses=10, device=args.device_id
        )

    ##############################################
    # Data Samplers and Loaders
    # ----------------------
    # Samplers for distributed training will typically assign a unique subset of the dataset to each GPU, shuffling
    # after each epoch, so that each GPU trains on a distinct subset of the dataset each epoch. The
    # InterruptibleDistributedSampler from cycling_utils by Strong Compute does this while also tracking progress of the
    # sampler through the dataset.

    train_sampler = InterruptableSampler(train_dataset)
    test_sampler = InterruptableSampler(test_dataset)
    timer.report("Initialized samplers")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, sampler=train_sampler, num_workers=3)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, sampler=test_sampler, shuffle=False)
    timer.report("Initialized dataloaders")



    #####################################
    # Metrics and logging
    # -----------------
    # Metrics are commonly tracked and plotted during training to report on progress and model performance.

    metrics = {"train": MetricsTracker(), "test": MetricsTracker()}
    # writer = SummaryWriter(log_dir=args.tboard_path)

    #####################################
    # Retrieve the checkpoint if the experiment is resuming from pause

    lastsavedcheckpoint = "Checkpoints/model" + str(rank) + ".pth.temp"
    savedcheckpoint = "Checkpoints/model" + str(rank) + ".pth"

    if os.path.isfile(lastsavedcheckpoint):
        print(f"Loading checkpoint from {lastsavedcheckpoint}")
        checkpoint = torch.load(lastsavedcheckpoint, map_location=f"cuda:{args.device_id}")

        model.load_state_dict(checkpoint["model"])
        train_dataloader.sampler.load_state_dict(checkpoint["train_sampler"])
        test_dataloader.sampler.load_state_dict(checkpoint["test_sampler"])
        metrics = checkpoint["metrics"]
        timer.report("Retrieved savedcheckpoint")

    #####################################
    # Main training loop
    # --------------------
    # Each epoch the training loop is called within a context set from the training InterruptibleDistributedSampler

    train_loop(
        model, train_dataloader, test_dataloader, metrics, args, savedcheckpoint
    )

    test_loop(
        model, train_dataloader, test_dataloader, metrics, args, savedcheckpoint,
        hsize=config['hsize'], lamb=config['lambd'], w_lr=config['w_lr'], b_lr=config['b_lr'], l_lr=config['l_lr'], w_norm=config['w_norm'], anti_hebb_factor=config['anti_hebb_factor']
    )

    print("Done!")

if __name__ == "__main__":
    args = get_args_parser().parse_args()

    main(args, timer)

#################################################################
# Further Reading
# -----------------------
# - `Loss Functions <https://pytorch.org/docs/stable/nn.html#loss-functions>`_
# - `torch.optim <https://pytorch.org/docs/stable/optim.html>`_
# - `Warmstart Training a Model <https://pytorch.org/tutorials/recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html>`_