import models.MLP.model as MLP
import models.CNN.model as CNN

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
from torch import nn
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
    parser.add_argument("--dataset", type=str, default="EMNIST")
    parser.add_argument("--finetuneepochs", type=int, default=10)
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


# def greedytrain(model, train_dataloader, test_dataloader, metrics, args):
#     epoch = 0
#     train_batches_per_epoch = len(train_dataloader)
#     # Set the model to training mode - important for layers with different training / inference behaviour
#     model.eval()
#     layers = list(mymodel.basemodel.layers.values())

#     for idx in range(len(mymodel.basemodel.layers)):
#         idx += 1
#         for inputs, targets in train_dataloader:

#             # Determine the current batch
#             batch = train_dataloader.sampler.progress // train_dataloader.batch_size
#             is_last_batch = (batch + 1) == train_batches_per_epoch

#             # Move input and targets to device
#             inputs, targets = inputs.to(args.device_id), targets.to(args.device_id)
#             timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - data to device")

#             # Forward pass
#             if args.dataset == "EMNIST":
#                 labels = models.hyperparams.oneHotEncode(targets, 47, args.device_id)
#                 inputs = inputs.reshape(1,1,28,28)
#             elif args.dataset == "FashionMNIST" :
#                 labels = models.hyperparams.oneHotEncode(targets, 10, args.device_id)
#                 inputs = inputs.reshape(1,1,28,28)
#             elif args.dataset == "CIFAR10":
#                 labels = models.hyperparams.oneHotEncode(targets, 10, args.device_id)
#             elif args.dataset == "CIFAR100":
#                 labels = models.hyperparams.oneHotEncode(targets, 100, args.device_id)

#             x = inputs.to(device)
#                                 for r_l in range(idx):
#                         if (r_l + 1) == idx:
#                             _, x = layers[r_l].forward(x, oneHotEncode(labels, nclasses, mymodel.device))
#                         else :
#                             _, x = layers[r_l].forward(x, update_weights=False)

#             model(inputs, labels)

#             timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - forward pass")
        
#             metrics["train"].update({"examples_seen": len(inputs)})
#             metrics["train"].reduce()  # Gather results from all nodes - sums metrics from all nodes into local aggregate

#             # Advance sampler - essential for interruptibility
#             train_dataloader.sampler.advance(len(inputs))
#             timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - advance sampler")

#             metrics["train"].reset_local()

#             if is_last_batch:
#                 metrics["train"].end_epoch()  # Store epoch aggregates and reset local aggregate for next epoch

#             # Saving and reporting
#             if args.is_master:
#                 # total_progress = train_dataloader.sampler.progress + epoch * train_batches_per_epoch
#                 # Save checkpoint
#                 atomic_torch_save(
#                     {
#                         "model": model.state_dict(),
#                         "train_sampler": train_dataloader.sampler.state_dict(),
#                         "test_sampler": test_dataloader.sampler.state_dict(),
#                         "metrics": metrics,
#                     },
#                     args.checkpoint_path,
#                 )
#                 timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - save checkpoint")

def greedytrain_loop(model, train_dataloader, test_dataloader, metrics, args):
    epoch = 0
    train_batches_per_epoch = len(train_dataloader)
    # Set the model to training mode - important for layers with different training / inference behaviour
    model.eval()

    layers = list(model.basemodel.layers.values())

    for idx in range(len(layers)):
        train_dataloader.sampler.reset_progress()
        idx += 1

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
                inputs = inputs.reshape(1,1,28,28)
            elif args.dataset == "FashionMNIST" :
                labels = models.hyperparams.oneHotEncode(targets, 10, args.device_id)
                inputs = inputs.reshape(1,1,28,28)
            elif args.dataset == "CIFAR10":
                labels = models.hyperparams.oneHotEncode(targets, 10, args.device_id)
            elif args.dataset == "CIFAR100":
                labels = models.hyperparams.oneHotEncode(targets, 100, args.device_id)

            x = inputs
            for r_l in range(idx):
                if (r_l + 1) == idx:
                    _, x = layers[r_l].forward(x, labels)
                else :
                    _, x = layers[r_l].forward(x, update_weights=False)

            timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - forward pass")
        
            metrics["train"].update({"examples_seen": len(inputs)})
            metrics["train"].reduce()  # Gather results from all nodes - sums metrics from all nodes into local aggregate

            # Advance sampler - essential for interruptibility
            train_dataloader.sampler.advance(len(inputs))
            timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - advance sampler")

            metrics["train"].reset_local()

            if is_last_batch:
                metrics["train"].end_epoch()  # Store epoch aggregates and reset local aggregate for next epoch

            # Saving and reporting
            if args.is_master:
                # total_progress = train_dataloader.sampler.progress + epoch * train_batches_per_epoch
                # Save checkpoint
                atomic_torch_save(
                    {
                        "model": model.state_dict(),
                        "train_sampler": train_dataloader.sampler.state_dict(),
                        "test_sampler": test_dataloader.sampler.state_dict(),
                        "metrics": metrics,
                    },
                    args.checkpoint_path,
                )
                timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - save checkpoint")

def train_loop(model, train_dataloader, test_dataloader, metrics, args):
    epoch = 0
    train_batches_per_epoch = len(train_dataloader)
    # Set the model to training mode - important for layers with different training / inference behaviour
    model.eval()

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
            inputs = inputs.reshape(1,1,28,28)
        elif args.dataset == "FashionMNIST" :
            labels = models.hyperparams.oneHotEncode(targets, 10, args.device_id)
            inputs = inputs.reshape(1,1,28,28)
        elif args.dataset == "CIFAR10":
            labels = models.hyperparams.oneHotEncode(targets, 10, args.device_id)
        elif args.dataset == "CIFAR100":
            labels = models.hyperparams.oneHotEncode(targets, 100, args.device_id)

        model.train_conv(inputs, labels)

        timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - forward pass")
    
        metrics["train"].update({"examples_seen": len(inputs)})
        metrics["train"].reduce()  # Gather results from all nodes - sums metrics from all nodes into local aggregate

        # Advance sampler - essential for interruptibility
        train_dataloader.sampler.advance(len(inputs))
        timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - advance sampler")

        metrics["train"].reset_local()

        if is_last_batch:
            metrics["train"].end_epoch()  # Store epoch aggregates and reset local aggregate for next epoch

        # Saving and reporting
        if args.is_master:
            # total_progress = train_dataloader.sampler.progress + epoch * train_batches_per_epoch
            # Save checkpoint
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "train_sampler": train_dataloader.sampler.state_dict(),
                    "test_sampler": test_dataloader.sampler.state_dict(),
                    "metrics": metrics,
                },
                args.checkpoint_path,
            )
            timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - save checkpoint")
    




def finetune(model, train_dataloader, test_dataloader, metrics, args):
    epoch = 0
    train_batches_per_epoch = len(train_dataloader)

    train_dataloader.sampler.reset_progress()
    model.train()
    for inputs, targets in train_dataloader:

        # Determine the current batch
        batch = train_dataloader.sampler.progress // train_dataloader.batch_size
        is_last_batch = (batch + 1) == train_batches_per_epoch

        inputs = inputs.to(args.device_id)
        targets = targets.to(args.device_id)

        timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - Fine tuning : data to device")

        if args.dataset == "EMNIST":
            labels = models.hyperparams.oneHotEncode(targets, 47, args.device_id)
            inputs = inputs.reshape(1,1,28,28)
        elif args.dataset == "FashionMNIST" :
            labels = models.hyperparams.oneHotEncode(targets, 10, args.device_id)
            inputs = inputs.reshape(1,1,28,28)
        elif args.dataset == "CIFAR10":
            labels = models.hyperparams.oneHotEncode(targets, 10, args.device_id)
        elif args.dataset == "CIFAR100":
            labels = models.hyperparams.oneHotEncode(targets, 100, args.device_id)

        model.train_classifier(inputs, labels)


        timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - Fine tuning : forward pass")
    
        metrics["train"].update({"examples_seen": len(inputs)})
        metrics["train"].reduce()  # Gather results from all nodes - sums metrics from all nodes into local aggregate

        # Advance sampler - essential for interruptibility
        train_dataloader.sampler.advance(len(inputs))
        timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - Fine tuning : advance sampler")

        metrics["train"].reset_local()

        if is_last_batch:
            metrics["train"].end_epoch()  # Store epoch aggregates and reset local aggregate for next epoch

        # Saving and reporting
        if args.is_master:
            # total_progress = train_dataloader.sampler.progress + epoch * train_batches_per_epoch
            # Save checkpoint
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "train_sampler": train_dataloader.sampler.state_dict(),
                    "test_sampler": test_dataloader.sampler.state_dict(),
                    "metrics": metrics,
                },
                args.checkpoint_path,
            )
            timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - Fine tuning : save checkpoint")

# def test_loop(model, train_dataloader, test_dataloader, metrics, args):

#     test_batches_per_epoch = len(test_dataloader)
#     epoch = 0
#     # Set the model to evaluation mode - important for layers with different training / inference behaviour
#     model.eval()

#     # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode also serves to
#     # reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
#     with torch.no_grad():
#         for inputs, targets in test_dataloader:
#             # Determine the current batch
#             batch = test_dataloader.sampler.progress // test_dataloader.batch_size
#             is_last_batch = (batch + 1) == test_batches_per_epoch
#             # Move input and targets to device
#             inputs, targets = inputs.to(args.device_id), targets.to(args.device_id)
#             timer.report(f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] - data to device")

#             # Inference
#             if args.topdown:
#                 inputs = inputs.reshape(1,1,28,28)
#                 predictions = model.TD_forward_test(inputs)
#             else:
#                 inputs = inputs.reshape(1,1,28,28)
#                 predictions = model.forward_test(inputs)

#             timer.report(f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] - inference")

#             # Performance metrics logging
#             correct = (predictions.argmax(1) == targets).type(torch.float).sum()
#             model.save_acc(correct.item(), len(inputs))

#             metrics["test"].update({"examples_seen": len(inputs), "correct": correct.item()})
#             metrics["test"].reduce()  # Gather results from all nodes - sums metrics from all nodes into local aggregate
#             metrics["test"].reset_local()  # Reset local cache
#             timer.report(f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] - metrics logging")

#             # Advance sampler
#             test_dataloader.sampler.advance(len(inputs))

#             # Performance summary at the end of the epoch
#             if is_last_batch:
#                 correct, examples_seen = itemgetter("correct", "examples_seen")(metrics["test"].agg)
#                 pct_test_correct = correct / examples_seen
#                 metrics["test"].end_epoch()
#                 timer.report(
#                     f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] :: TEST ACC: {pct_test_correct}"
#                 )

#             # Save checkpoint
#             if args.is_master and (is_last_batch or (batch + 1) % 5 == 0):
#                 # Save checkpoint
#                 atomic_torch_save(
#                     {
#                         "model": model.state_dict(),
#                         "train_sampler": train_dataloader.sampler.state_dict(),
#                         "test_sampler": test_dataloader.sampler.state_dict(),
#                         "metrics": metrics,
#                     },
#                     args.checkpoint_path,
#                 )
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

    with open("Configs/config" + str(rank) + ".json", "r") as file:
        config = json.load(file)

    if dataset == "EMNIST":
        transform_EMNIST = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.T),
            transforms.Lambda(lambda x : x.reshape(-1))
        ])
        train_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, download=False, transform=transform_EMNIST)
        test_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, download=False, transform=transform_EMNIST)

        model = MLP.MLPBaseline_Model(args.hsize, args.lambd, args.lr, args.epsilon, args.rho, args.gamma, 47, args.device_id)
        model = model.to(args.device_id)
        #model = DDP(model, device_ids=[args.device_id])
        #timer.report("Prepared model for distributed training")

    elif dataset == "FashionMNIST":
        transform_MNIST = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))
        ])

        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform_MNIST)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_MNIST)

        train_dataset = ConcatDataset([train_dataset]*args.epochs)

        hyperp = hyperps[rank]

        model = CNN.CNNBaseline_Model(inputsize=(1, 28, 28), kernels=[5,3,3], channels=[16,64,256], strides=[2,2,2], padding=[0, 1, 0], lambd=hyperp[0], lr=hyperp[1], gamma=0.99, epsilon=0.01,
            rho=hyperp[2], nbclasses=10, topdown=True, device=args.device_id, wl=hyperp[4], ws=hyperp[5], o=hyperp[3])

        model = model.to(args.device_id)
        #model = DDP(model, device_ids=[args.device_id])
        #timer.report("Prepared model for distributed training")
        v_input = test_dataset[0]

    elif dataset == "CIFAR10":
        transform = transforms.Compose(
            [transforms.ToTensor()])

        train_dataset = datasets.CIFAR10(root='./data', train=True,
                                                download=False, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False,
                                            download=False, transform=transform)
        
        # train_dataset = Subset(train_dataset, list(range(10)))

        model = CNN.CNN_Model_from_config((3,32,32), config, args.device_id, 10)


    timer.report("Initialized datasets")

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

    train_dataloader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler, num_workers=3)
    test_dataloader = DataLoader(test_dataset, batch_size=1, sampler=test_sampler, shuffle=False)
    timer.report("Initialized dataloaders")




    #####################################
    # Metrics and logging
    # -----------------
    # Metrics are commonly tracked and plotted during training to report on progress and model performance.

    metrics = {"train": MetricsTracker(), "test": MetricsTracker()}
    # writer = SummaryWriter(log_dir=args.tboard_path)

    #####################################
    # Retrieve the checkpoint if the experiment is resuming from pause

    if os.path.isfile(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=f"cuda:{args.device_id}")

        model.load_state_dict(checkpoint["model"])
        train_dataloader.sampler.load_state_dict(checkpoint["train_sampler"])
        test_dataloader.sampler.load_state_dict(checkpoint["test_sampler"])
        metrics = checkpoint["metrics"]
        timer.report("Retrieved savedcheckpoint")

    #####################################
    # Main training loop
    # --------------------
    # Each epoch the training loop is called within a context set from the training InterruptibleDistributedSampler
    if config['greedytrain']:
        greedytrain_loop(
            model, train_dataloader, test_dataloader, metrics, args
        )
        finetune(
            model, train_dataloader, test_dataloader, metrics, args
        )
    else:
        train_loop(
            model, train_dataloader, test_dataloader, metrics, args
        )
        finetune(
            model, train_dataloader, test_dataloader, metrics, args
        )

    # test_loop(
    #     model, train_dataloader, test_dataloader, metrics, args
    # )

    #CNN.Save_Model(model, args.dataset, rank, args.topdown, v_input, args.device_id, (model.correct/model.tot)*100)
    torch.save(model, 'SavedModels/model' + str(rank) + ".pth")
    print("Done!")

if __name__ == "__main__":
    args = get_args_parser().parse_args()

    # 0 : hsize
    # 1 : lambda
    # 2 : lr
    # 3 : eps
    # 4 : rho
    # 5 : gamma
    # 6 : classifier_learning
    # 7 : weight_learning
    # 8 : weight_modifier

    # lambds = [3]
    # lr = [5e-5]
    # rho = np.logspace(-7, 0, num=72)
    # classifier_learnings = [models.hyperparams.LearningRule.OutputContrastiveSupervised]
    # weight_learnings = [models.hyperparams.LearningRule.OrthogonalExclusive]
    # weight_mods = [models.hyperparams.WeightScale.WeightNormalization]

    # combinations = list(itertools.product(lambds, lr, rho, classifier_learnings, weight_learnings, weight_mods))
    # hyperps = [list(comb) for comb in combinations]

    main(args, timer)

#################################################################
# Further Reading
# -----------------------
# - `Loss Functions <https://pytorch.org/docs/stable/nn.html#loss-functions>`_
# - `torch.optim <https://pytorch.org/docs/stable/optim.html>`_
# - `Warmstart Training a Model <https://pytorch.org/tutorials/recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html>`_