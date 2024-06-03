#####################################
# Progress Timing and Logging
# ----------------------------
# Important milestones in progress through this script are logged using a timing utility which also includes details of
# when milestones were reached and the elapsedtime between milestones. This can assist with debugging and optimisation.

from cycling_utils import TimestampedTimer

timer = TimestampedTimer("Imported TimestampedTimer")

import argparse
import os
from operator import itemgetter
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import Compose, Lambda, PILToTensor, RandAugment

from cycling_utils import (
    InterruptableDistributedSampler,
    MetricsTracker,
    atomic_torch_save,
)

timer.report("Completed imports")

##############################################################################

#from models.hebbian_network import HebbianNetwork
from layers.scheduler import Scheduler
#from data.data_loader import MNIST_set, fashion_MNIST_set
#from experiments.mlp import MLPExperiment

##############################################################################




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from models.hebbian_network import HebbianNetwork
from models.network import Network 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





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

    # Basic configurations.
    parser.add_argument('--is_training', type=bool, default=True, help='status')
    parser.add_argument('--data_name', type=str, default="MNIST")
    
    # Data Factory
    parser.add_argument('--train_data', type=str, default="data/mnist/train-images.idx3-ubyte")
    parser.add_argument('--train_label', type=str, default="data/mnist/train-labels.idx1-ubyte")
    parser.add_argument('--test_data', type=str, default="data/mnist/t10k-images.idx3-ubyte")
    parser.add_argument('--test_label', type=str, default="data/mnist/t10k-labels.idx1-ubyte")

    # CSV files generated
    parser.add_argument('--train_filename', type=str, default="data/mnist/mnist_train.csv")
    parser.add_argument('--test_filename', type=str, default="data/mnist/mnist_test.csv")

    # Dimension of each layer
    parser.add_argument('--input_dim', type=int, default=784)
    parser.add_argument('--heb_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=10)

    # Hebbian layer hyperparameters
    parser.add_argument('--heb_lr', type=float, default=0.005)
    parser.add_argument('--heb_lamb', type=float, default=15)
    parser.add_argument('--heb_gam', type=float, default=0)

    # Classification layer hyperparameters
    parser.add_argument('--cla_lr', type=float, default=0.001)
    parser.add_argument('--cla_lamb', type=float, default=1)
    parser.add_argument('--cla_gam', type=float, default=0)

    # Shared hyperparameters
    parser.add_argument('--eps', type=float, default=0.01)

    # Experiment parameters
    parser.add_argument('--num_epochs', type=int, default=3)

# ---------------------------------------

    # The number of times to loop over the whole dataset
    parser.add_argument("--epochs", type=int, default=100)

    # Testing model performance on a test every "test-epochs" epochs
    parser.add_argument("--test-epochs", type=int, default=5)
    
    # A model training regularisation technique to reduce over-fitting
    parser.add_argument("--dropout", type=float, default=0.2)
    
    # This example demonstrates a StepLR learning rate scheduler. Different schedulers will require different hyper-parameters.
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr-step-size", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=1)
    
    # ---------------------------------------
    parser.add_argument("--batch-size", type=int, default=16)
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

def oneHotEncode(labels, num_classes):
    one_hot_encoded = torch.zeros(len(labels), num_classes)
    one_hot_encoded.scatter_(1, labels.unsqueeze(1), 1)

    return one_hot_encoded.squeeze()

def set_hebbian_scheduler(heb_lr, step_size, gamma, model):
    scheduler=Scheduler(heb_lr, step_size, gamma)
    #model.set_scheduler(scheduler, 0)
    model.set_scheduler()


def train_loop(model, lr_scheduler, train_dataloader, test_dataloader, metrics, writer, args):
    epoch = train_dataloader.sampler.epoch
    train_batches_per_epoch = len(train_dataloader)
   
    # Set the model to training mode - important for layers with different training / inference behaviour
    model.train()

    if args.gamma != 0 : set_hebbian_scheduler(args.lr, args.lr_step_size, args.gamma, model)

    for inputs, targets in train_dataloader:
        # Reset model parameter gradients
        # optimizer.zero_grad()

        # Determine the current batch
        batch = train_dataloader.sampler.progress // train_dataloader.batch_size
        is_last_batch = (batch + 1) == train_batches_per_epoch

        # Move input and targets to device
        inputs, targets = inputs.to(args.device_id), oneHotEncode(targets, 10).to(args.device_id)
        timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - data to device")
        
        # Forward pass
        predictions = model(inputs, clamped_output=targets)
        timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - forward pass")
        
        # Compute loss and log to metrics
        # loss = loss_fn(predictions, targets)
        metrics["train"].update({"examples_seen": len(inputs)})
        metrics["train"].reduce()  # Gather results from all nodes - sums metrics from all nodes into local aggregate
        timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}]")
        
        # Backpropagation
        # loss.backward()
        # timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - backward pass")
        
        # Update model weights
        # optimizer.step()
        # timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - model parameter update")
        
        # Advance sampler - essential for interruptibility
        train_dataloader.sampler.advance(len(inputs))
        timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - advance sampler")
        
        # Report training metrics
        examples_seen = itemgetter("examples_seen")(metrics["train"].local)
        # batch_avg_loss = total_batch_loss / examples_seen
        metrics["train"].reset_local()

        if is_last_batch:
            lr_scheduler.step()  # Step learning rate scheduler at the end of the epoch
            metrics["train"].end_epoch()  # Store epoch aggregates and reset local aggregate for next epoch

        # Saving and reporting
        if args.is_master:
            total_progress = train_dataloader.sampler.progress + epoch * train_batches_per_epoch
            # writer.add_scalar("Train/avg_loss", batch_avg_loss, total_progress)
            writer.add_scalar("Train/learn_rate", lr_scheduler.get_last_lr(), total_progress)
            # Save checkpoint
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    # "optimizer": optimizer.state_dict(),
                    "train_sampler": train_dataloader.sampler.state_dict(),
                    "test_sampler": test_dataloader.sampler.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "metrics": metrics,
                },
                args.checkpoint_path,
            )
            timer.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - save checkpoint")
    
    model.visualizeWeights()
    

def test_loop(model, lr_scheduler, train_dataloader, test_dataloader, metrics, writer, args):
    epoch = test_dataloader.sampler.epoch
    test_batches_per_epoch = len(test_dataloader)
    
    # Set the model to evaluation mode - important for layers with different training / inference behaviour
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
            predictions = model(inputs)
            timer.report(f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] - inference")
            
            # Test loss
            # test_loss = loss_fn(predictions, targets)
            timer.report(f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] - loss calculation")
            
            # Performance metrics logging
            correct = (predictions.argmax(1) == targets).type(torch.float).sum()
            metrics["test"].update({"examples_seen": len(inputs), "correct": correct.item()})
            metrics["test"].reduce()  # Gather results from all nodes - sums metrics from all nodes into local aggregate
            metrics["test"].reset_local()  # Reset local cache
            timer.report(f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] - metrics logging")
            
            # Advance sampler
            test_dataloader.sampler.advance(len(inputs))

            # Performance summary at the end of the epoch
            if args.is_master and is_last_batch:
                total_loss, correct, examples_seen = itemgetter("loss", "correct", "examples_seen")(metrics["test"].agg)
                avg_test_loss = total_loss / examples_seen
                pct_test_correct = correct / examples_seen
                writer.add_scalar("Test/avg_test_loss", avg_test_loss, epoch)
                writer.add_scalar("Test/pct_test_correct", pct_test_correct, epoch)
                metrics["test"].end_epoch()
                timer.report(
                    f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] :: AVG TEST LOSS: \
                             {avg_test_loss}, TEST ACC: {pct_test_correct}"
                )

            # Save checkpoint
            if args.is_master and (is_last_batch or (batch + 1) % 5 == 0):
                # Save checkpoint
                atomic_torch_save(
                    {
                        "model": model.state_dict(),
                        # "optimizer": optimizer.state_dict(),
                        "train_sampler": train_dataloader.sampler.state_dict(),
                        "test_sampler": test_dataloader.sampler.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "metrics": metrics,
                    },
                    args.checkpoint_path,
                )


timer.report("Defined helper function/s, loops, and model")

# Helper function - optimizer
def optimizer(model):
    optimizer = optim.Adam(model.get_layer("Hebbian Layer").parameters(), self.args.cla_lr)
    return optimizer

def loss_function(model):
    loss_function = nn.CrossEntropyLoss()
    return loss_function


def main(args, timer):
    ##############################################
    # Distributed Training Configuration
    # -----------------
    # The following steps demonstrate configuration of the local GPU for distributed training. This includes
    # initialising the process group, obtaining and setting the rank of the GPU within the cluster and on the local
    # node.

    dist.init_process_group("nccl")  # Expects RANK set in environment variable
    rank = int(os.environ["RANK"])  # Rank of this GPU in cluster
    world_size = int(os.environ["WORLD_SIZE"]) # Total number of GPUs in the cluster
    args.device_id = int(os.environ["LOCAL_RANK"])  # Rank on local node
    args.is_master = rank == 0  # Master node for saving / reporting
    torch.cuda.set_device(args.device_id)  # Enables calling 'cuda'

    timer.report("Setup for distributed training")

    args.checkpoint_path = args.save_dir / "checkpoint.pt"
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    timer.report("Validated checkpoint path")


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Set up model
    model = HebbianNetwork(args)
    model = model.to(args.device_id)
    timer.report("Model set up and moved to device")


    # Set up Data Sampler and Loaders
    # First, training data
    train_data_set = model.get_layer("Input Layer").setup_train_data()
    train_sampler = InterruptableDistributedSampler(train_data_set)
    train_data_loader = DataLoader(train_data_set, batch_size=1, shuffle=False, sampler=train_sampler)  # Added sampler, set shuffle to False
    timer.report("training data(sampler and dataloader) processing set up")

    # Second, testing data
    test_data_set = model.get_layer("Input Layer").setup_test_data()
    test_sampler = InterruptableDistributedSampler(test_data_set)  
    test_data_loader = DataLoader(test_data_set, batch_size=1, shuffle=False, sampler=test_sampler)  # Added sampler, set shuffle to False
    timer.report("testing data(sampler and dataloader) processing set up")












    ##############################################
    # Data Samplers and Loaders
    # ----------------------
    # Samplers for distributed training will typically assign a unique subset of the dataset to each GPU, shuffling
    # after each epoch, so that each GPU trains on a distinct subset of the dataset each epoch. The
    # InterruptibleDistributedSampler from cycling_utils by Strong Compute does this while also tracking progress of the
    # sampler through the dataset.

#    training_data = MNIST_set()
#    test_data = MNIST_set(0)

#    train_sampler = InterruptableDistributedSampler(training_data)
#    test_sampler = InterruptableDistributedSampler(test_data)
#    timer.report("Initialized samplers")

#    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, sampler=train_sampler, num_workers=3)
#    test_dataloader = DataLoader(test_data, batch_size=1, sampler=test_sampler)
    
#    timer.report("Initialized dataloaders")

    ##############################################
    # Model Preparation
    # ----------------------
    # This example demonstrates a Convolutional Neural Network (ConvNet, refer to the accompanying model.py for
    # details). After instantiating the model, we move it to the local device and prepare it for distributed training
    # with DistributedDataParallel (DDP).

    # model = ConvNet(args.dropout)
    # model = model.to(args.device_id)
    # model = DDP(model, device_ids=[args.device_id])
    # timer.report("Prepared model for distributed training")

#    input_dimension = 784
#    hidden_layer_dimension = 64
#    output_dimension = 10
#    heb_lr = 1
#    lamb = 1
#    eps = 0.1
#    model = HebbianNetwork(args, input_dimension, hidden_layer_dimension, 
#                                  output_dimension, heb_lr=heb_lr, lamb=lamb, eps=eps)
#    model = model.to(args.device_id)
#    timer.report("Prepared model for distributed training")
    

    ########################################
    # We initialize the loss function, optimizer, and learning rate scheduler and pass them to ``train_loop`` and
    # ``test_loop``. By setting the loss function reduction strategy to "sum" we are able to confidently summarise the
    # loss accross the whole cluster by summing the loss computed by each node. In general, it is important to consider
    # the validity of the metric summarisation strategy when using distributed training.
    
    lr_scheduler = Scheduler(args.lr, args.lr_step_size, args.gamma)
    timer.report(
        f"Ready for training with hyper-parameters: \ninitial learning_rate: {args.lr}, \nbatch_size: \
                 {args.batch_size}, \nepochs: {args.epochs}"
    )

    #####################################
    # Metrics and logging
    # -----------------
    # Metrics are commonly tracked and plotted during training to report on progress and model performance.

    metrics = {"train": MetricsTracker(), "test": MetricsTracker()}
    writer = SummaryWriter(log_dir=args.tboard_path)

    #####################################
    # Retrieve the checkpoint if the experiment is resuming from pause

    if os.path.isfile(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=f"cuda:{args.device_id}")

        model.load_state_dict(checkpoint["model"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        train_data_loader.sampler.load_state_dict(checkpoint["train_sampler"])
        test_data_loader.sampler.load_state_dict(checkpoint["test_sampler"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        metrics = checkpoint["metrics"]
        timer.report("Retrieved savedcheckpoint")

    #####################################
    # Main training loop
    # --------------------
    # Each epoch the training loop is called within a context set from the training InterruptibleDistributedSampler

    for epoch in range(train_data_loader.sampler.epoch, args.epochs):
        with train_data_loader.sampler.in_epoch(epoch):
            train_loop(
                model, 
                # optimizer, 
                lr_scheduler, 
                # loss_fn, 
                train_data_loader, 
                test_data_loader, 
                metrics, 
                writer, 
                args
            )

            # An inner context is also set from the testing sampler. This ensures that both training and testing can be
            # interrupted and resumed from checkpoint.
            
            if epoch % args.test_epochs == 0:
                with test_data_loader.sampler.in_epoch(epoch):
                    test_loop(
                        model,
                        # optimizer,
                        lr_scheduler,
                        # loss_fn,
                        train_dataloader,
                        test_data_loader,
                        metrics,
                        writer,
                        args,
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