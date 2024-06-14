
##############################################################################
# PART 1: Imports and Timer Initialization
##############################################################################

# Built-in imports
import logging
import argparse
import os
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

    return parser



##############################################################################
# PART 3: Training
##############################################################################
"""
Method defining how a single training epoch works
@param
    model (models.Network) = the network that is being trained
    train_data_loader (torch.DataLoader) = dataloader with the training data
    test_data_loader (torch.DataLoader) = dataloader with testing data
    args (argparse.ArgumentParser) = arguments that were passed to the function
@return
    ___ (void) = no returns
"""
def train_loop(model, train_data_loader, test_data_loader, args):

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
        
        # Forward pass
        predictions = model(inputs, clamped_output=targets)
        
        # Advance sampler - essential for interruptibility 
        train_data_loader.sampler.advance(len(inputs))  # moves the sampler forward by the number of examples in current batch
    
    

##############################################################################
# PART 4: Testing
##############################################################################
"""
Method that test the model at certain epochs during the training process
@param
    model (models.Network) = model to be trained
    train_data_loder (torch.DataLoader) = dataloader containing the training dataset
    test_data_loader (torch.DataLoader) = dataloader containing the testing dataset
    args (argparse.ArgumentParser) = arguments that are pased from the command shell
@return
    ___ (void) = no returns
"""
def test_loop(model, train_data_loader, test_data_loader, args):

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
            
            # Inference
            predictions = model(inputs)
            
            # Performance metrics logging
            correct = (predictions.argmax(1) == targets).type(torch.float).sum() 
            
            # Degubbing purposes
            debug = logging.getLogger("Debug Log")
            debug.info(f"Prediciton/Actual: {predictions.argmax(1).item()}/{targets.item()}.")

            # Advance sampler -> Advances the sampler by the number of examples in the current batch.
            test_data_loader.sampler.advance(len(inputs))
            test = logging.getLogger("Test Log")
            test.info(f'Epoch Number: {epoch} || Test Accuracy: {pct_test_correct} || Average Test Loss: {avg_test_loss} || PCT Test Loss: {pct_test_correct}')


"""
Method to test the model on the entire testing dataset
@param
    model (models.Network) = ML model to be tested
    test_dataset (torch.DataSet) = testing dataset that model will be tested on
@return
    correct/len(data_loader) (floattt) = accuracy of the model
"""
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
    args.device_id = 'cpu'
    torch.device(args.device_id)


    # ===========================================
    # Set up model
    # ===========================================
    model = HebbianNetwork(args).float()
    model = model.to(args.device_id)


    # ===========================================
    # Set up datasets for training and testing purposes
    # ===========================================
    
    # Training dataset
    train_data_set = model.get_module("Input Layer").setup_train_data()
    train_data_loader = DataLoader(train_data_set, batch_size=args.batch_size, shuffle=True)

    # Testing dataset
    test_data_set = model.get_module("Input Layer").setup_test_data()
    test_data_loader = DataLoader(test_data_set, batch_size=args.batch_size, shuffle=True)


    # ===========================================
    # Training and testing process
    # ===========================================
   
    # Loops through each epoch from current epoch to total number of epochs
    for epoch in range(0, args.epochs): 
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
print("Hello")
args = get_args_parser().parse_args()

# Create folder in results to store training and testing results for this experiment
exp_num = 'cpu-1'
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