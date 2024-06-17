
##############################################################################
# PART 1: Imports and Timer Initialization
##############################################################################

# Built-in imports
import logging
import argparse
import os
import shutil
import time
from operator import itemgetter
from pathlib import Path

# Pytorch imports
import torch
import torch.optim as optim
import torch.distributed as dist
from torch import nn
from torch.nn.functional import one_hot
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# Custom defined model imports
from models.hebbian_network import HebbianNetwork # Model import

# Utils imports
from utils.experiment_logger import *
from utils.experiment_parser import *



##############################################################################
# PART 2: Create logs for experiment and parse arguments
##############################################################################
# Start timer
START_TIME = time.time()

# Parse Arguments
ARGS = parse_arguments()

# Setup the result folder
EXP_NUM = 'cpu-1' if ARGS.local_machine else os.environ["OUTPUT_PATH"].split("/")[-1]
RESULT_PATH = f"results/experiment-{EXP_NUM}"

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)
    print(f"Experiment {EXP_NUM} result folder created successfully.")

# else:
#     try:
#         shutil.rmtree(RESULT_PATH)
#         print(f"Removed {RESULT_PATH}.")
#         os.makedirs(RESULT_PATH, exist_ok=True)
#         print(f"Experiment {EXP_NUM} result folder re-created successfully.")
#     except OSError as e:
#         print(f"Error: {e.strerror}")

# Create logs
PRINT_LOG = get_print_log("Print Log", RESULT_PATH) # Replace print statements (for debugging purposes)
TEST_LOG = get_test_log("Test Log", RESULT_PATH) # Test accuracy
PARAM_LOG = get_parameter_log("Parameter Log", RESULT_PATH) # Experiment parameters
DEBUG_LOG = get_debug_log("Debug Log", RESULT_PATH) # Debugging stuff
EXP_LOG = get_experiment_log("Experiment Log", RESULT_PATH) # Logs during experiment

EXP_LOG.info("Completed imports.")
EXP_LOG.info("Completed log setups.")
EXP_LOG.info("Completed arguments parsing.")
EXP_LOG.info(f"Experiment {EXP_NUM} result folder created successfully.")

# Strong Compute Logging and Imports
if not ARGS.local_machine:
    from torch.utils.tensorboard import SummaryWriter
    from torchvision import datasets
    from torchvision.transforms import Compose, Lambda, PILToTensor, RandAugment
    from cycling_utils import (
        TimestampedTimer,
        InterruptableDistributedSampler,
        MetricsTracker,
        atomic_torch_save,
    )
    TIMER = TimestampedTimer("Imported TimestampedTimer")
    TIMER.report("Completed imports")



##############################################################################
# PART 3: Training
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
def train_loop_sc(model, train_data_loader, test_data_loader, metrics, writer, args):
    EXP_LOG.info("Started 'train_loop_sc.")

    # Epoch and batch set up
    epoch = train_data_loader.sampler.epoch # Gets the current epoch from the trian_data_loader
    train_batches_per_epoch = len(train_data_loader) # This is the number of batches in one epoch 
    
    EXP_LOG.info(f"This training batch is epoch #{epoch} with {train_batches_per_epoch} training examples per batch.")

    # Set the model to training mode - important for layers with different training / inference behaviour
    model.train()

    EXP_LOG.info("Set the model to training mode.")

    # Loop through training batches
    for inputs, targets in train_data_loader:

        # Batch information
        batch = train_data_loader.sampler.progress // train_data_loader.batch_size # Calculates the current batch
        is_last_batch = (batch + 1) == train_batches_per_epoch # Checks if current batch is last batch in current epoch

        # Move input and targets to device
        inputs, targets = inputs.to(args.device_id).float(), one_hot(targets, 10).squeeze().to(args.device_id).float()
        TIMER.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - data to device")
        EXP_LOG.info(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - data to device")
        
        # Forward pass
        predictions = model(inputs, clamped_output=targets)
        TIMER.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - forward pass")
        EXP_LOG.info(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - forward pass")
        
        # Update metrics
        metrics["train"].update({"examples_seen": len(inputs)})
        metrics["train"].reduce()  # Gather results from all nodes - sums metrics from all nodes into local aggregate
        TIMER.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}]")
        EXP_LOG.info(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}]")
        
        # Advance sampler - essential for interruptibility 
        train_data_loader.sampler.advance(len(inputs))  # moves the sampler forward by the number of examples in current batch
        TIMER.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - advance sampler")
        EXP_LOG.info(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - advance sampler")
        
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
            EXP_LOG.info(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - save checkpoint")
            EXP_LOG.info(f"Examples seen: {examples_seen}")


"""
Method defining how a single training epoch works
@param
    model (models.Network) = the network that is being trained
    train_data_loader (torch.DataLoader) = dataloader with the training data
    device (str) = name of device on which computations will be done
@return
    ___ (void) = no returns
"""
def train_loop_cpu(model, train_data_loader, device):
    EXP_LOG.info("Started 'train_loop_cpu' function.")
    
    model.train()
    
    EXP_LOG.info("Set the model to training mode.")

    for inputs, labels in train_data_loader:
        inputs, labels = inputs.to(device).float(), one_hot(labels, 10).squeeze().to(device).float()
        model(inputs, clamped_output=labels)
    
    EXP_LOG.info("Completed 1 epoch of training of 'train_loop_cpu'.")
    


##############################################################################
# PART 4: Testing
##############################################################################
"""
Method that test the model at certain epochs during the training process
@param
    model (models.Network) = model to be trained
    train_data_loder (torch.DataLoader) = dataloader containing the training dataset
    test_data_loader (torch.DataLoader) = dataloader containing the testing dataset
    metrics (dict{str:MetricsTracker}) = a tracker to keep track of the metrics
    writer (SummaryWriter) = a custom logger
    args (argparse.ArgumentParser) = arguments that are pased from the command shell
@return
    ___ (void) = no returns
"""
def test_loop_sc(model, train_data_loader, test_data_loader, metrics, writer, args):

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


"""
Method that test the model at certain epochs during the training process
@param
    model (models.Network) = model to be trained
    test_data_loader (torch.DataLoader) = dataloader containing the testing dataset
    device (str) = name of device on which computations will be done
    epoch (int) = epoch number training is at
@return
    correct/total (float) = accuracy of the model
"""
def test_loop_cpu(model, test_data_loader, device, epoch):
    EXP_LOG.info("Started 'test_loop' function.")

    model.eval()

    EXP_LOG.info("Set the model to testing mode.")

    with torch.no_grad():
        correct = 0
        total = len(test_data_loader.dataset)

        for inputs, labels in test_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            predictions = model(inputs)

            EXP_LOG.info(f"The inputs were put into the model ({labels.item()}) and {predictions.argmax(1).item()} are the predictions.")

            correct += (predictions.argmax(1) == labels).type(torch.float).sum().item()

            EXP_LOG.info(f"The number of correct predictions until now: {correct} out of {total}.")

        EXP_LOG.info(f"Completed testing with {correct} out of {total}.")
        TEST_LOG.info(f'Epoch Number: {epoch} || Test Accuracy: {correct/total}') 
        EXP_LOG.info("Completed 'test_loop' function.")
        
        return correct / total



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
    if not args.local_machine:
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
    else:
        args.device_id = 'cpu'
        torch.device(args.device_id)


    # ===========================================
    # Set up model
    # ===========================================
    model = HebbianNetwork(args).float()
    model = model.to(args.device_id)
    if not args.local_machine: TIMER.report("Model set up and moved to device")
    EXP_LOG.info("Created model for the experiment.")


    # ===========================================
    # Set up datasets for training and testing purposes
    # ===========================================
    
    # Prepping Variables
    train_data_set = None
    train_data_sampler = None
    train_data_loader = None

    test_data_set = None
    test_data_sampler = None
    test_data_loader = None

    
    if not args.local_machine:
        # Training dataset
        train_data_set = model.get_module("Input Layer").setup_train_data()
        train_sampler = InterruptableDistributedSampler(train_data_set)
        train_data_loader = DataLoader(train_data_set, batch_size=args.batch_size, shuffle=False, sampler=train_sampler)  # Added sampler, set shuffle to False
        TIMER.report("training data(sampler and dataloader) processing set up")
        EXP_LOG.info("Completed setup for training dataset and dataloader.")

        # Testing dataset
        test_data_set = model.get_module("Input Layer").setup_test_data()
        test_sampler = InterruptableDistributedSampler(test_data_set)  
        test_data_loader = DataLoader(test_data_set, batch_size=args.batch_size, shuffle=False, sampler=test_sampler)  # Added sampler, set shuffle to False
        TIMER.report("testing data(sampler and dataloader) processing set up")
        EXP_LOG.info("Completed setup for testing dataset and dataloader.")

        # Logging process
        TIMER.report(
            f"Ready for training with hyper-parameters: \nlearning_rate: {args.lr}, \nbatch_size: \{args.batch_size}, \nepochs: {args.epochs}"
        )
        EXP_LOG.info(f"Ready for training with hyper-parameters: learning_rate ({args.lr}), batch_size ({args.batch_size}), epochs ({args.epochs}).")
    else:
        # Training dataset
        train_data_set = model.get_module("Input Layer").setup_train_data()
        train_data_loader = DataLoader(train_data_set, batch_size=args.batch_size, shuffle=True)
        EXP_LOG.info("Completed setup for training dataset and dataloader.")

        # Testing dataset
        test_data_set = model.get_module("Input Layer").setup_test_data()
        test_data_loader = DataLoader(test_data_set, batch_size=args.batch_size, shuffle=True)
        EXP_LOG.info("Completed setup for testing dataset and dataloader.")
        EXP_LOG.info(f"Ready for training with hyper-parameters: learning_rate ({args.lr}), batch_size ({args.batch_size}), epochs ({args.epochs}).")


    if not args.local_machine:
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
        EXP_LOG.info("Started training and testing loops.")
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

    else:
        EXP_LOG.info("Started training and testing loops.")
        for epoch in range(0, args.epochs): 
            train_loop_cpu(
                model, 
                train_data_loader, 
                args.device_id,
            )

            test_loop_cpu(
                model,
                test_data_loader,
                args.device_id,
                epoch
            )
    
    EXP_LOG.info("Completed training of model.")        
    model.visualize_weights(RESULT_PATH)
    EXP_LOG.info("Visualize weights of model after training.")
    accuracy = test_loop_cpu(model, test_data_loader, args.device_id, args.epochs)
    EXP_LOG.info("Completed all 3 different testing methods.")
    PARAM_LOG.info(f"Accuracy of model after training for {args.epochs} epochs: {accuracy}")
    EXP_LOG.info("Experiment Completed!!!")

    print("Done!")



##############################################################################
# PART 6: What code will be ran when file is ran
##############################################################################

# Actual code that will be ran
if __name__ == "__main__":

    # Logging training parameters
    if os.path.getsize(RESULT_PATH+'/parameters.log') == 0:
        EXP_LOG.info("Started logging of experiment parameters.")
        PARAM_LOG.info(f"Input Dimension: {ARGS.input_dim}")
        PARAM_LOG.info(f"Hebbian Layer Dimension: {ARGS.heb_dim}")
        PARAM_LOG.info(f"Outout Dimension: {ARGS.output_dim}")
        PARAM_LOG.info(f"Hebbian Layer Lambda: {ARGS.heb_lamb}")
        PARAM_LOG.info(f"Hebbian Layer Gamma: {ARGS.heb_gam}")
        PARAM_LOG.info(f"Classification Layer Lambda: {ARGS.cla_lamb}")
        PARAM_LOG.info(f"Network Learning Rate: {ARGS.lr}")
        PARAM_LOG.info(f"Epsilon: {ARGS.eps}")
        PARAM_LOG.info(f"Number of Epochs: {ARGS.epochs}")
        EXP_LOG.info("Completed logging of experiment parameters.")

    # Logging start of experiment
    EXP_LOG.info("Start of experiment.")

    # Run experiment
    main(ARGS)

    # End timer
    END_TIME = time.time()
    DURATION = END_TIME - START_TIME
    minutes = (DURATION // 60)
    seconds = DURATION % 60
    EXP_LOG.info(f"The experiment took {minutes}m:{seconds}s to be completed.")