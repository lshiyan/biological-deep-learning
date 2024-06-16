##############################################################################
# PART 1: Imports
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
# PART 2: Create logs for experiment
##############################################################################

# Start timer
START_TIME = time.time()

# Setup the result folder
EXP_NUM = 'cpu-1'
RESULT_PATH = f"results/experiment-{EXP_NUM}"

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)
    print(f"Experiment {EXP_NUM} result folder created successfully.")
else:
    try:
        shutil.rmtree(RESULT_PATH)
        print(f"Removed {RESULT_PATH}.")
        os.makedirs(RESULT_PATH, exist_ok=True)
        print(f"Experiment {EXP_NUM} result folder re-created successfully.")
    except OSError as e:
        print(f"Error: {e.strerror}")

# Create logs
PRINT_LOG = get_print_log("Print Log", RESULT_PATH) # Replace print statements (for debugging purposes)
TEST_LOG = get_test_log("Test Log", RESULT_PATH) # Test accuracy
PARAM_LOG = get_parameter_log("Parameter Log", RESULT_PATH) # Experiment parameters
DEBUG_LOG = get_debug_log("Debug Log", RESULT_PATH) # Debugging stuff
EXP_LOG = get_experiment_log("Experiment Log", RESULT_PATH) # Logs during experiment

EXP_LOG.info("Completed imports.")
EXP_LOG.info("Completed log setups.")



##############################################################################
# PART 3: Helper functions
##############################################################################
def get_optimizer(model):
    optimizer = optim.Adam(model.get_module("Hebbian Layer").parameters(), 0.001)
    return optimizer

def get_loss_function():
    loss_function = nn.CrossEntropyLoss()
    return loss_function



##############################################################################
# PART 4: Training
##############################################################################
"""
Method defining how a single training epoch works
@param
    model (models.Network) = the network that is being trained
    train_data_loader (torch.DataLoader) = dataloader with the training data
    device (str) = name of device on which computations will be done
    optimizer (torch.Optim) = optimizer for 
@return
    ___ (void) = no returns
"""
def train_loop(model, train_data_loader, device):
    model.train()
    
    EXP_LOG.info("Set the model to training mode.")

    for inputs, labels in train_data_loader:
        inputs, labels = inputs.to(device).float(), one_hot(labels, 10).squeeze().to(device).float()

        #EXP_LOG.info(f"Sent inputs and labels to specified device ({device}).")

        model(inputs, clamped_output=labels)

        #EXP_LOG.info(f"The inputs and labels passed through model for training.")



##############################################################################
# PART 5: Testing
##############################################################################
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
def test_loop(model, test_data_loader, device, epoch):
    EXP_LOG.info("Started 'test_loop' function.")

    model.eval()

    EXP_LOG.info("Set the model to testing mode.")

    with torch.no_grad():
        correct = 0
        seen = 0
        total = len(test_data_loader.dataset)

        for inputs, labels in test_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            #EXP_LOG.info(f"Sent inputs and labels to specified device ({device}).")

            predictions = model(inputs)

            EXP_LOG.info(f"The inputs were put into the model ({labels.item()}) and {predictions.argmax(1).item()} are the predictions.")

            correct += (predictions.argmax(1) == labels).type(torch.float).sum().item()
            seen += len(labels)

            EXP_LOG.info(f"The number of correct predictions until now: {correct} out of {seen}.")

        EXP_LOG.info(f"Completed testing with {correct} out of {seen}.")
        EXP_LOG.warning(f"Check if seen ({seen}) is same as total ({total})")
        TEST_LOG.info(f'Epoch Number: {epoch} || Test Accuracy: {correct/total}') 
        EXP_LOG.info("Completed 'test_loop' function.")
        
        return correct / total



##############################################################################
# PART 6: Main Function
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
    args.device_id = 'cpu'
    torch.device(args.device_id)


    # ===========================================
    # Set up model
    # ===========================================
    model = HebbianNetwork(args).float()
    model = model.to(args.device_id)
    EXP_LOG.info("Created model for the experiment.")


    # ===========================================
    # Set up datasets for training and testing purposes
    # ===========================================
    
    # Training dataset
    train_data_set = model.get_module("Input Layer").setup_train_data()
    train_data_loader = DataLoader(train_data_set, batch_size=args.batch_size, shuffle=True)
    EXP_LOG.info("Completed setup for training dataset and dataloader.")

    # Testing dataset
    test_data_set = model.get_module("Input Layer").setup_test_data()
    test_data_loader = DataLoader(test_data_set, batch_size=args.batch_size, shuffle=True)
    EXP_LOG.info("Completed setup for testing dataset and dataloader.")


    # ===========================================
    # Training and testing process
    # ===========================================
   
    # Loops through each epoch from current epoch to total number of epochs
    EXP_LOG.info("Started training and testing loops.")
    
    for epoch in range(0, args.epochs): 
        train_loop(
            model, 
            train_data_loader, 
            args.device_id,
        )

        test_loop(
            model,
            test_data_loader,
            args.device_id,
            epoch
        )

    EXP_LOG.info("Completed training of model.")

    model.visualize_weights(RESULT_PATH)

    EXP_LOG.info("Visualize weights of model after training.")

    accuracy = test_loop(model, test_data_loader, args.device_id, args.epochs)

    EXP_LOG.info("Completed all 3 different testing methods.")

    PARAM_LOG.info(f"Accuracy of model after training for {args.epochs} epochs: {accuracy}")

    EXP_LOG.info("Experiment Completed!!!")



##############################################################################
# PART 7: What code will be ran when file is ran
##############################################################################

# Actual code that will be ran
args = parse_arguments()
EXP_LOG.info("Completed arguments' parsing.")

# Logging training parameters
if os.path.getsize(RESULT_PATH+'/parameters.log') == 0:
    EXP_LOG.info("Started logging of experiment parameters.")
    PARAM_LOG.info(f"Input Dimension: {args.input_dim}")
    PARAM_LOG.info(f"Hebbian Layer Dimension: {args.heb_dim}")
    PARAM_LOG.info(f"Outout Dimension: {args.output_dim}")
    PARAM_LOG.info(f"Hebbian Layer Lambda: {args.heb_lamb}")
    PARAM_LOG.info(f"Hebbian Layer Gamma: {args.heb_gam}")
    PARAM_LOG.info(f"Classification Layer Lambda: {args.cla_lamb}")
    PARAM_LOG.info(f"Network Learning Rate: {args.lr}")
    PARAM_LOG.info(f"Epsilon: {args.eps}")
    PARAM_LOG.info(f"Number of Epochs: {args.epochs}")
    EXP_LOG.info("Completed logging of experiment parameters.")

# Logging start of experiment
EXP_LOG.info("Start of experiment.")

# Run experiment
main(args)

# End timer
END_TIME = time.time()
DURATION = END_TIME - START_TIME
EXP_LOG.info(f"The experiment took {DURATION} time to be completed.")