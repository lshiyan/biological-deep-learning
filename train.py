
##############################################################################
# PART 1: Imports and Timer Initialization
##############################################################################

# Here:
    # Import necessary libraries and modules
    # TimestampedTimer is used to log progress and timing of various milestones in the script


from cycling_utils import TimestampedTimer

TIMER = TimestampedTimer("Imported TimestampedTimer")

import logging
import argparse
import os
from operator import itemgetter
from pathlib import Path

import torch
import torch.optim as optim
import torch.distributed as dist
from torch import nn
from torch.nn.functional import one_hot
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

from models.hebbian_network import HebbianNetwork # Model import

TIMER.report("Completed imports")

##############################################################################
# PART 2: Hyperparameters argument parsing
##############################################################################
"""
Method to setup all the arguments and passe them to an argument parser
@param
@return
    parser (argparse.ArgumentParser) = an argument parser containing all arguments for the experience
"""
def get_args_parser():
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
    parser.add_argument('--heb_gam', type=float, default=0.99)

    # Classification layer hyperparameters
    parser.add_argument('--cla_lr', type=float, default=0.005)
    parser.add_argument('--cla_lamb', type=float, default=1)

    # Shared hyperparameters
    parser.add_argument('--eps', type=float, default=10e-5)

# ---------------------------------------

    # The number of times to loop over the whole dataset
    parser.add_argument("--epochs", type=int, default=1000)

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

    # ---------------------------------------
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--tboard-path", type=Path, required=True)

    return parser


##############################################################################
# PART 3: Training Loop
##############################################################################
"""
Method defining how a single training epoch works
@param
    model (models.Network) = the network that is being trained
    train_data_loader (torch.DataLoader) = dataloader with the training data
    test_data_loader (torch.DataLoader) = dataloader with testing data
    metrics (dict{str:MetricsTracker}) = a tracker to keep track of the metrics
    writer (SummaryWriter) = a custom logger NOTE: not being used?
    args (argparse.ArgumentParser) = arguments that were passed to the function
@return
    ___ (void) = no returns
"""
def train_loop(model, train_data_loader, test_data_loader, metrics, writer, args):

    # Epoch and batch set up
    epoch = train_data_loader.sampler.epoch # Gets the current epoch from the trian_data_loader
    train_batches_per_epoch = len(train_data_loader) # This is the number of batches in one epoch 
   
    # Set the model to training mode - important for layers with different training / inference behaviour
    model.train()

    # Loop through training batches
    for inputs, targets in train_data_loader:

        # Batch information
        batch = train_data_loader.sampler.progress // train_data_loader.batch_size # Calculates the current batch
        is_last_batch = (batch + 1) == train_batches_per_epoch # Checks if current batch is last batch in current epoch

        # Move input and targets to device
        inputs, targets = inputs.to(args.device_id).float(), one_hot(targets, 10).squeeze().to(args.device_id).float()
        TIMER.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - data to device")
        
        # Forward pass
        predictions = model(inputs, clamped_output=targets)
        TIMER.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - forward pass")
        
        # Update metrics
        metrics["train"].update({"examples_seen": len(inputs)})
        metrics["train"].reduce()  # Gather results from all nodes - sums metrics from all nodes into local aggregate
        TIMER.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}]")
        
        # Advance sampler - essential for interruptibility 
        train_data_loader.sampler.advance(len(inputs))  # moves the sampler forward by the number of examples in current batch
        TIMER.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - advance sampler")
        
        # Report training metrics
        examples_seen = itemgetter("examples_seen")(metrics["train"].local)
        metrics["train"].reset_local()

        # Checks if current batch is last in epoch
        if is_last_batch:
            metrics["train"].end_epoch()  # Store epoch aggregates and reset local aggregate for next epoch

        # Saving and reporting
        if args.is_master:
            total_progress = train_data_loader.sampler.progress + epoch * train_batches_per_epoch

            # Save current state of model to checkpoint
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "train_sampler": train_data_loader.sampler.state_dict(),
                    "test_sampler": test_data_loader.sampler.state_dict(),
                    "metrics": metrics,
                },
                args.checkpoint_path,
            )
            TIMER.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - save checkpoint")
    
    

##############################################################################
# PART 4: Testing Loop
##############################################################################

# Here:
    # This function manages the testing process for each epoch
        # 1. Sets the model to evaluation mode
        # 2. Loops through the test data, performing forward passes and updating metrics
        # 3. Saves checkpoints periodically


def test_loop(model, train_data_loader, test_data_loader, metrics, writer, args):

    # Epoch and batch set up
    epoch = test_data_loader.sampler.epoch
    test_batches_per_epoch = len(test_data_loader)
    
    # Set the model to evaluation mode - important for layers with different training / inference behaviour
    model.eval()

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode 
    #  also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        
        # Loop thorugh testing batches
        for inputs, targets in test_data_loader:
            
            # Determine the current batch
            batch = test_data_loader.sampler.progress // test_data_loader.batch_size
            is_last_batch = (batch + 1) == test_batches_per_epoch
            
            # Move input and targets to device
            inputs, targets = inputs.to(args.device_id), targets.to(args.device_id)
            TIMER.report(f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] - data to device")
            
            # Inference
            predictions = model(inputs)
            TIMER.report(f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] - inference")
            
            # Performance metrics logging
            correct = (predictions.argmax(1) == targets).type(torch.float).sum() 
                # For each prediction, 'predictions.argmax(1)' finds the index of the maximum value -> corresponds to the predicted class
                # Then, this compares the predicted classes to the true target classes
                # Then, I convert the boolean array of correct to float (where True becomes 1.0 and False becomes 0.0)
                # Lastly, I sum them to get the total number of correct predictions.

            # print(f"Predictions: {predictions}.")
            # print(f"True Labels: {targets}.")
            metrics["test"].update({"examples_seen": len(inputs), "correct": correct.item()})
            metrics["test"].reduce()  # Gather results from all nodes - sums metrics from all nodes into local aggregate
            metrics["test"].reset_local()  # Reset local cache
            TIMER.report(f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] - metrics logging")
            
            # Degubbing purposes
            debug = logging.getLogger("Debug Log")
            debug.info(f"Prediciton/Actual: {predictions.argmax(1).item()}/{targets.item()}.")

            # Advance sampler -> Advances the sampler by the number of examples in the current batch.
            test_data_loader.sampler.advance(len(inputs))

            # Performance summary at the end of the epoch
            if args.is_master and is_last_batch:
                total_loss, correct, examples_seen = itemgetter("loss", "correct", "examples_seen")(metrics["test"].agg)
                avg_test_loss = total_loss / examples_seen
                pct_test_correct = correct / examples_seen
                writer.add_scalar("Test/avg_test_loss", avg_test_loss, epoch)
                writer.add_scalar("Test/pct_test_correct", pct_test_correct, epoch)
                metrics["test"].end_epoch()
                TIMER.report(
                    f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] :: AVG TEST LOSS: \
                             {avg_test_loss}, TEST ACC: {pct_test_correct}"
                )
                print(f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] :: AVG TEST LOSS: \
                             {avg_test_loss}, TEST ACC: {pct_test_correct}")
                
                
                test = logging.getLogger("Test Log")
                test.info(f'Epoch Number: {epoch} || Test Accuracy: {pct_test_correct} || Average Test Loss: {avg_test_loss} || PCT Test Loss: {pct_test_correct}')
                

            # Save checkpoint
            if args.is_master and (is_last_batch or (batch + 1) % 5 == 0):
                # Save checkpoint
                atomic_torch_save(
                    {
                        "model": model.state_dict(),
                        "train_sampler": train_data_loader.sampler.state_dict(),
                        "test_sampler": test_data_loader.sampler.state_dict(),
                        "metrics": metrics,
                    },
                    args.checkpoint_path,
                )


def model_test(model, test_data):
    model.eval()

    data_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(args.device_id), targets.to(args.device_id)
            predictions = model(inputs)
            correct = (predictions.argmax(1) == targets).type(torch.float).sum()

    return correct/len(data_loader)


##############################################################################
# PART 5: Main Function
##############################################################################
"""
Method describing the main part of the code -> how experiment will be ran
@param
    args (argparse.ArgumentParser) = arguments passed to the main function
@return
    ___ (void) = no returns
"""
def main(args):
    # ===========================================
    # Distributed Training Configuration
    # ===========================================
    dist.init_process_group("nccl")  # Expects RANK set in environment variable
    rank = int(os.environ["RANK"])  # Rank of this GPU in cluster
    world_size = int(os.environ["WORLD_SIZE"]) # Total number of GPUs in the cluster
    args.device_id = int(os.environ["LOCAL_RANK"])  # Rank on local node
    args.is_master = rank == 0  # Master node for saving / reporting
    torch.cuda.set_device(args.device_id)  # Enables calling 'cuda'

    TIMER.report("Setup for distributed training")

    args.checkpoint_path = args.save_dir / "checkpoint.pt"
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    TIMER.report("Validated checkpoint path")



    # ===========================================
    # Set up model
    # ===========================================
    model = HebbianNetwork(args).float()
    model = model.to(args.device_id)
    TIMER.report("Model set up and moved to device")



    # ===========================================
    # Set up datasets for training and testing purposes
    # ===========================================
    
    # Training dataset
    train_data_set = model.get_module("Input Layer").setup_train_data()
    train_sampler = InterruptableDistributedSampler(train_data_set)
    train_data_loader = DataLoader(train_data_set, batch_size=1, shuffle=False, sampler=train_sampler)  # Added sampler, set shuffle to False
    TIMER.report("training data(sampler and dataloader) processing set up")

    # Testing dataset
    test_data_set = model.get_module("Input Layer").setup_test_data()
    test_sampler = InterruptableDistributedSampler(test_data_set)  
    test_data_loader = DataLoader(test_data_set, batch_size=1, shuffle=False, sampler=test_sampler)  # Added sampler, set shuffle to False
    TIMER.report("testing data(sampler and dataloader) processing set up")

    # Logging process
    TIMER.report(
        f"Ready for training with hyper-parameters: \ninitial learning_rate: {args.lr}, \nbatch_size: \{args.batch_size}, \nepochs: {args.epochs}"
    )



    # ===========================================
    # Metrics and Logging
    # ===========================================

    # Setup metrics and logger
    metrics = {"train": MetricsTracker(), "test": MetricsTracker()} # Initialize Metric Tracker for both training and testing
    writer = SummaryWriter(log_dir=args.tboard_path)



    # ===========================================
    # Retrieve the checkpoint if the experiment is resuming from pause
    # ===========================================
    if os.path.isfile(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=f"cuda:{args.device_id}")

        model.load_state_dict(checkpoint["model"])
        train_data_loader.sampler.load_state_dict(checkpoint["train_sampler"])
        test_data_loader.sampler.load_state_dict(checkpoint["test_sampler"])
        metrics = checkpoint["metrics"]
        TIMER.report("Retrieved savedcheckpoint")



    # ===========================================
    # Training and testing process
    # ===========================================
   
    # Loops through each epoch from current epoch to total number of epochs
    for epoch in range(train_data_loader.sampler.epoch, args.epochs): 
        with train_data_loader.sampler.in_epoch(epoch):
            train_loop(
                model, 
                train_data_loader, 
                test_data_loader, 
                metrics, 
                writer, 
                args
            )


            # TESTING AT INTERVALS
            # Determines how frequently the model is tested thorughout training
            if epoch % args.test_epochs == 0:
                with test_data_loader.sampler.in_epoch(epoch):
                    test_loop(
                        model,
                        train_data_loader,
                        test_data_loader,
                        metrics,
                        writer,
                        args,
                    )

    model.visualize_weights(folder_path)
    accuracy = model_test(model, test_data_set) # Final test after entire training
    param = logging.getLogger("Parameter Log")
    param.info(f"Accuracy of model after training for {args.epochs} epochs: {accuracy}")

    print("Done!")



##############################################################################
# PART 6: What code will be ran when file is ran
##############################################################################
# Helper function
"""
Method to create a logger to log information
@param
    name (str) = name of logger
    file (str) - path to file
    level (logging.Level) = level of the log
    format (logging.Formatter) = format of the log
@return
    logger (logging.Logger) = a logger
"""
def configure_logger(name, file, level=logging.INFO, format=logging.Formatter('%(asctime)s || %(message)s')):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.FileHandler(file)
    handler.setLevel(level)
    handler.setFormatter(format)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# Actual code that will be ran
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    
    # Create folder in results to store training and testing results for this experiment
    output_path = os.environ["OUTPUT_PATH"]
    exp_num = output_path.split("/")[-1]
    folder_path = f"results/experiment-{exp_num}"
    log_result_path = folder_path + "/testing.log"
    log_param_path = folder_path + "/parameters.log"
    log_debug_path = folder_path + "/debug.log"
    log_print_path = folder_path + "/prints.log"
    log_basic_path = folder_path + "/basic.log"
    log_format = logging.Formatter('%(asctime)s || %(message)s')
    log_level = logging.INFO

    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"Experiment {exp_num} result folder created successfully.")
    else:
        print(f"Experiment {exp_num} result folder already exists.")

    # Create logs
    print_log = configure_logger("Print Log", log_print_path, log_level, log_format) # Replace print statements (for debugging purposes)
    test_log = configure_logger("Test Log", log_result_path, log_level, log_format) # Test accuracy
    param_log = configure_logger("Parameter Log", log_param_path, log_level, log_format) # Experiment parameters
    debug_log = configure_logger("Debug Log", log_debug_path, log_level, log_format) # Debugging stuff

    # Logging training parameters
    if os.path.getsize(log_param_path) == 0:
        param_log.info(f"Input Dimension: {args.input_dim}")
        param_log.info(f"Hebbian Layer Dimension: {args.heb_dim}")
        param_log.info(f"Outout Dimension: {args.output_dim}")
        param_log.info(f"Hebbian Layer Learning Rate: {args.heb_lr}")
        param_log.info(f"Hebbian Layer Lambda: {args.heb_lamb}")
        param_log.info(f"Hebbian Layer Gamma: {args.heb_gam}")
        param_log.info(f"Classification Layer Learning Rate: {args.cla_lr}")
        param_log.info(f"Classification Layer Lambda: {args.cla_lamb}")
        param_log.info(f"Epsilon: {args.eps}")
        param_log.info(f"Number of Epochs: {args.epochs}")

    # Run experiment
    main(args)