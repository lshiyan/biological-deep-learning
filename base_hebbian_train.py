##############################################################################
# PART 1: Imports and Timer Initialization
##############################################################################

# Built-in imports
import os
import shutil
import time
from operator import itemgetter

# Pytorch imports
import torch
import torch.distributed as dist
from torch.nn.functional import one_hot
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset

# Custom defined model imports
from models.hebbian_network import HebbianNetwork # Model import

# Utils imports
from utils.experiment_logger import *
from utils.experiment_parser import *
from utils.experiment_timer import *



##############################################################################
# PART 2: Create logs for experiment and parse arguments
##############################################################################

# Start timer
START_TIME = time.time()
TRAIN_TIME = 0
TEST_ACC_TIME = 0
TRAIN_ACC_TIME = 0

# Parse Arguments
ARGS = parse_arguments()
# Args.local_machine = False

# Setup the result folder
EXP_NUM = 'cpu-1' if ARGS.local_machine else os.environ["OUTPUT_PATH"].split("/")[-1]
RESULT_PATH = f"results/experiment-{EXP_NUM}"

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)
    os.makedirs(f"{RESULT_PATH}/classification", exist_ok=True)
    os.makedirs(f"{RESULT_PATH}/hebbian", exist_ok=True)
    print(f"Experiment '{EXP_NUM}' result folder created successfully.")
    print(f"Experiment '{EXP_NUM}/classification' result folder created successfully.")
    print(f"Experiment '{EXP_NUM}/hebbian' result folder created successfully.")
    

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
TEST_LOG = get_test_acc_log("Test Log", RESULT_PATH) # Testing accuracy
TRAIN_LOG = get_train_acc_log("Train Log", RESULT_PATH) # Training accuracy
PARAM_LOG = get_parameter_log("Parameter Log", RESULT_PATH) # Experiment parameters
DEBUG_LOG = get_debug_log("Debug Log", RESULT_PATH) # Debugging stuff
EXP_LOG = get_experiment_log("Experiment Log", RESULT_PATH) # Logs during experiment

EXP_LOG.info("Completed imports.")
EXP_LOG.info("Completed log setups.")
EXP_LOG.info("Completed arguments parsing.")
EXP_LOG.info(f"Experiment '{EXP_NUM}' result folder created successfully.")

# Strong Compute Logging and Imports
if not ARGS.local_machine:
    from torch.utils.tensorboard import SummaryWriter
    from torchvision import datasets # type: ignore
    from torchvision.transforms import Compose, Lambda, PILToTensor, RandAugment # type: ignore
    from cycling_utils import ( # type: ignore
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
    epoch_num (int) = training epoch current training loop is at
    metrics (dict{str:MetricsTracker}) = a tracker to keep track of the metrics
    args (argparse.ArgumentParser) = arguments that were passed to the function
@return
    ___ (void) = no returns
"""
def train_loop(model, train_data_loader, test_data_loader, train_test_data_loader, args, epoch_num, metrics=None):
    global TRAIN_TIME
    train_start = time.time()
    EXP_LOG.info("Started 'train_loop'.")

    # Epoch and batch set up
    epoch = epoch_num if args.local_machine else train_data_loader.sampler.epoch # Gets the current epoch from the train_data_loader
    train_batches_per_epoch = len(train_data_loader) # This is the number of batches in one epoch 
    
    EXP_LOG.warning(f"Make sure epoch number ({epoch}) is same as loop epoch number ({epoch_num}).")
    EXP_LOG.info(f"This training batch is epoch #{epoch} with {train_batches_per_epoch} batches of size {args.batch_size} in this epoch.")

    # Set the model to training mode - important for layers with different training / inference behaviour
    model.train()

    EXP_LOG.info("Set the model to training mode.")

    # Loop through training batches
    for inputs, labels in train_data_loader:   
        # Move input and targets to device
        inputs, labels = inputs.to(args.device_id).float(), one_hot(labels, 10).squeeze().to(args.device_id).float()
        # EXP_LOG.info(f"EPOCH [{epoch}] - data to device")
        
        
        # Forward pass
        model(inputs, clamped_output=labels)
        # EXP_LOG.info(f"EPOCH [{epoch}] - forward pass")
        
        
        # PART ONLY USED IN STRONG COMPUTE
        if not args.local_machine:
            # Batch information
            batch = train_data_loader.sampler.progress // train_data_loader.batch_size # Calculates the current batch
            is_last_batch = (batch + 1) == train_batches_per_epoch # Checks if current batch is last batch in current epoch
            
            TIMER.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - data to device")
            TIMER.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - forward pass")
            
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
                        "train_test_sampler": train_test_data_loader.sampler.state_dict(),
                        "metrics": metrics,
                    },
                    args.checkpoint_path,
                )
                TIMER.report(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - save checkpoint")
                EXP_LOG.info(f"EPOCH [{epoch}] TRAIN BATCH [{batch} / {train_batches_per_epoch}] - save checkpoint")
                EXP_LOG.info(f"Examples seen: {examples_seen}")
    
    train_end = time.time()
    training_time = train_end - train_start
    TRAIN_TIME += training_time
    
    model.visualize_weights(RESULT_PATH, epoch_num, 'training')
        
    EXP_LOG.info("Completed 1 epoch of 'train_loop'.")
    EXP_LOG.info(f"Training of epoch #{epoch} took {time_to_str(training_time)}.")



##############################################################################
# PART 4: Testing
##############################################################################
"""
Method that test the model at certain epochs during the training process
@param
    model (models.Network) = model to be trained
    train_data_loder (torch.DataLoader) = dataloader containing the training dataset
    test_data_loader (torch.DataLoader) = dataloader containing the testing dataset
    epoch_num (int) = epoch number of training iteration that is being tested on
    metrics (dict{str:MetricsTracker}) = a tracker to keep track of the metrics
    writer (SummaryWriter) = a custom logger
    args (argparse.ArgumentParser) = arguments that are pased from the command shell
@return
    final_accuracy (float) = accuracy of the test
"""
def testing_accuracy(model, train_data_loader, test_data_loader, train_test_data_loader, args, epoch_num, metrics=None, writer=None):
    global TEST_ACC_TIME
    test_start = time.time()
    EXP_LOG.info("Started 'testing_accuracy' function.")

    # Epoch and batch set up
    epoch = epoch_num if args.local_machine else test_data_loader.sampler.epoch
    test_batches_per_epoch = len(test_data_loader)

    EXP_LOG.warning(f"Make sure epoch number ({epoch}) is same as loop epoch number ({epoch_num}).")
    EXP_LOG.info(f"This testing batch is epoch #{epoch} with {test_batches_per_epoch} batches of size {args.batch_size} in this epoch.")
    
    # Set the model to evaluation mode - important for layers with different training / inference behaviour
    model.eval()

    EXP_LOG.info("Set the model to testing mode.")

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode 
    #  also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    final_accuracy = 0

    with torch.no_grad():
        correct_sum = 0
        total = len(test_data_loader.dataset)

        # Loop thorugh testing batches
        for inputs, labels in test_data_loader:
            # Move input and targets to device
            inputs, labels = inputs.to(args.device_id), labels.to(args.device_id)
            # EXP_LOG.info(f"EPOCH [{epoch}] - data to device")
            
            # Inference
            predictions = model(inputs)
            # EXP_LOG.info(f"EPOCH [{epoch}] - inference")
            
            # Evaluates performance of model on testing dataset
            correct = (predictions.argmax(1) == labels).type(torch.float).sum()
            correct_sum += correct 

            # Degubbing purposes
            # EXP_LOG.info(f"The inputs were put into the model ({labels.item()}) and {predictions.argmax(1).item()} are the predictions.")
            # DEBUG_LOG.info(f"Prediciton/Actual: {predictions.argmax(1).item()}/{labels.item()}.")
            # EXP_LOG.info(f"The number of correct predictions until now: {correct_sum} out of {total}.")

            # PART ONLY USED IN STRONG COMPUTE
            if not args.local_machine:
                # Determine the current batch
                batch = test_data_loader.sampler.progress // test_data_loader.batch_size
                is_last_batch = (batch + 1) == test_batches_per_epoch
                
                TIMER.report(f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] - data to device")
                TIMER.report(f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] - inference")
                
                # Updating metrics
                metrics["test"].update({"examples_seen": len(inputs), "correct": correct.item()})
                metrics["test"].reduce()  # Gather results from all nodes - sums metrics from all nodes into local aggregate
                metrics["test"].reset_local()  # Reset local cache
                TIMER.report(f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] - metrics logging")
                EXP_LOG.info(f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] - metrics logging")

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
                    EXP_LOG.info(
                        f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] :: AVG TEST LOSS: \
                                {avg_test_loss}, TEST ACC: {pct_test_correct}"
                    )
                    print(f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] :: AVG TEST LOSS: \
                                {avg_test_loss}, TEST ACC: {pct_test_correct}")
                    
                    
                    TEST_LOG.info(f'Epoch Number: {epoch} || Test Accuracy: {pct_test_correct} || Average Test Loss: {avg_test_loss} || PCT Test Loss: {pct_test_correct}')
                    final_accuracy = pct_test_correct

                # Save checkpoint
                if args.is_master and (is_last_batch or (batch + 1) % 5 == 0):
                    # Save checkpoint
                    atomic_torch_save(
                        {
                            "model": model.state_dict(),
                            "train_sampler": train_data_loader.sampler.state_dict(),
                            "test_sampler": test_data_loader.sampler.state_dict(),
                            "train_test_sampler": train_test_data_loader.sampler.state_dict(),
                            "metrics": metrics,
                        },
                        args.checkpoint_path,
                    )
            # PART IF NOT ON STRONG COMPUTE
            else:
                final_accuracy = correct_sum/total
            
    test_end = time.time()
    testing_time = test_end - test_start
    TEST_ACC_TIME += testing_time
    
    model.visualize_weights(RESULT_PATH, epoch_num, 'test_acc')
    
    EXP_LOG.info(f"Completed testing with {correct_sum} out of {total}.")
    EXP_LOG.info("Completed 'testing_accuracy' function.")
    EXP_LOG.info(f"Testing (test acc) of epoch #{epoch} took {time_to_str(testing_time)}.")

    TEST_LOG.info(f'Epoch Number: {epoch} || Test Accuracy: {final_accuracy}')
    
    return final_accuracy


"""
Method that test the model on the training data at certain epochs during the training process
@param
    model (models.Network) = model to be trained
    train_data_loder (torch.DataLoader) = dataloader containing the training dataset
    test_data_loader (torch.DataLoader) = dataloader containing the testing dataset
    epoch_num (int) = epoch number of training iteration that is being tested on
    metrics (dict{str:MetricsTracker}) = a tracker to keep track of the metrics
    writer (SummaryWriter) = a custom logger
    args (argparse.ArgumentParser) = arguments that are pased from the command shell
@return
    final_accuracy (float) = accuracy of the test
"""
def training_accuracy(model, train_data_loader, test_data_loader, train_test_data_loader, args, epoch_num, metrics=None, writer=None):
    global TRAIN_ACC_TIME
    test_start = time.time()
    EXP_LOG.info("Started 'training_accuracy' function.")

    # Epoch and batch set up
    epoch = epoch_num if args.local_machine else train_test_data_loader.sampler.epoch
    test_batches_per_epoch = len(train_test_data_loader)

    EXP_LOG.warning(f"Make sure epoch number ({epoch}) is same as loop epoch number ({epoch_num}).")
    EXP_LOG.info(f"This testing batch is epoch #{epoch} with {test_batches_per_epoch} batches of size {args.batch_size} in this epoch.")
    
    # Set the model to evaluation mode - important for layers with different training / inference behaviour
    model.eval()

    EXP_LOG.info("Set the model to testing mode.")

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode 
    #  also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    final_accuracy = 0

    with torch.no_grad():
        correct_sum = 0
        total = len(train_test_data_loader.dataset)

        # Loop thorugh testing batches
        for inputs, labels in train_test_data_loader:
            # Move input and targets to device
            inputs, labels = inputs.to(args.device_id), labels.to(args.device_id)
            # EXP_LOG.info(f"EPOCH [{epoch}] - data to device")
            
            # Inference
            predictions = model(inputs)
            # EXP_LOG.info(f"EPOCH [{epoch}] - inference")
            
            # Evaluates performance of model on testing dataset
            correct = (predictions.argmax(1) == labels).type(torch.float).sum()
            correct_sum += correct 

            # Degubbing purposes
            # EXP_LOG.info(f"The inputs were put into the model ({labels.item()}) and {predictions.argmax(1).item()} are the predictions.")
            # DEBUG_LOG.info(f"Prediciton/Actual: {predictions.argmax(1).item()}/{labels.item()}.")
            # EXP_LOG.info(f"The number of correct predictions until now: {correct_sum} out of {total}.")

            # PART ONLY USED IN STRONG COMPUTE
            if not args.local_machine:
                # Determine the current batch
                batch = train_test_data_loader.sampler.progress // train_test_data_loader.batch_size
                is_last_batch = (batch + 1) == test_batches_per_epoch
                
                TIMER.report(f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] - data to device")
                TIMER.report(f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] - inference")
                
                # Updating metrics
                metrics["train-test"].update({"examples_seen": len(inputs), "correct": correct.item()})
                metrics["train-test"].reduce()  # Gather results from all nodes - sums metrics from all nodes into local aggregate
                metrics["train-test"].reset_local()  # Reset local cache
                TIMER.report(f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] - metrics logging")
                EXP_LOG.info(f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] - metrics logging")

                # Advance sampler -> Advances the sampler by the number of examples in the current batch.
                train_test_data_loader.sampler.advance(len(inputs))

                # Performance summary at the end of the epoch
                if args.is_master and is_last_batch:
                    total_loss, correct, examples_seen = itemgetter("loss", "correct", "examples_seen")(metrics["train-test"].agg)
                    avg_test_loss = total_loss / examples_seen
                    pct_test_correct = correct / examples_seen
                    writer.add_scalar("Test/avg_test_loss", avg_test_loss, epoch)
                    writer.add_scalar("Test/pct_test_correct", pct_test_correct, epoch)
                    metrics["train-test"].end_epoch()
                    TIMER.report(
                        f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] :: AVG TEST LOSS: \
                                {avg_test_loss}, TEST ACC: {pct_test_correct}"
                    )
                    EXP_LOG.info(
                        f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] :: AVG TEST LOSS: \
                                {avg_test_loss}, TEST ACC: {pct_test_correct}"
                    )
                    print(f"EPOCH [{epoch}] TEST BATCH [{batch} / {test_batches_per_epoch}] :: AVG TEST LOSS: \
                                {avg_test_loss}, TEST ACC: {pct_test_correct}")
                    
                    
                    TRAIN_LOG.info(f'Epoch Number: {epoch} || Test Accuracy: {pct_test_correct} || Average Test Loss: {avg_test_loss} || PCT Test Loss: {pct_test_correct}')
                    final_accuracy = pct_test_correct

                # Save checkpoint
                if args.is_master and (is_last_batch or (batch + 1) % 5 == 0):
                    # Save checkpoint
                    atomic_torch_save(
                        {
                            "model": model.state_dict(),
                            "train_sampler": train_data_loader.sampler.state_dict(),
                            "test_sampler": test_data_loader.sampler.state_dict(),
                            "train_test_sampler": train_test_data_loader.sampler.state_dict(),
                            "metrics": metrics,
                        },
                        args.checkpoint_path,
                    )
            # PART IF NOT ON STRONG COMPUTE
            else:
                final_accuracy = correct_sum/total
    
    test_end = time.time()
    testing_time = test_end - test_start
    TRAIN_ACC_TIME += testing_time
    
    model.visualize_weights(RESULT_PATH, epoch_num, 'train_acc')
            
    EXP_LOG.info(f"Completed testing with {correct_sum} out of {total}.")
    EXP_LOG.info("Completed 'training_accuracy' function.")
    EXP_LOG.info(f"Testing (train acc) of epoch #{epoch} took {time_to_str(testing_time)}.")

    TRAIN_LOG.info(f'Epoch Number: {epoch} || Test Accuracy: {final_accuracy}')
    
    return final_accuracy



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

    train_test_data_set = None
    train_test_data_sampler = None
    train_test_data_loader = None

    
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

        # Training dataset for testing
        train_test_data_set = model.get_module("Input Layer").setup_train_data()
        train_test_sampler = InterruptableDistributedSampler(test_data_set)  
        train_test_data_loader = DataLoader(train_test_data_set, batch_size=args.batch_size, shuffle=False, sampler=train_test_sampler)  # Added sampler, set shuffle to False
        TIMER.report("training testing data(sampler and dataloader) processing set up")
        EXP_LOG.info("Completed setup for training dataset and dataloader for testing purposes.")

        # Logging process
        TIMER.report(
            f"Ready for training with hyper-parameters: \nlearning_rate: {args.lr}, \nbatch_size: {args.batch_size}, \nepochs: {args.epochs}"
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

        # Training dataset for testing
        train_test_data_set = model.get_module("Input Layer").setup_train_data()
        train_test_data_loader = DataLoader(train_test_data_set, batch_size=args.batch_size, shuffle=True)
        EXP_LOG.info("Completed setup for training dataset and dataloader for testing purposes.")

        EXP_LOG.info(f"Ready for training with hyper-parameters: learning_rate ({args.lr}), batch_size ({args.batch_size}), epochs ({args.epochs}).")


    if not args.local_machine:
    # ===========================================
    # Metrics and Logging
    # ===========================================
        # Setup metrics and logger
        metrics = {"train": MetricsTracker(), "test": MetricsTracker(), "train-test": MetricsTracker()} # Initialize Metric Tracker for both training and testing
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
            train_test_data_loader.sampler.load_state_dict(checkpoint["train_test_sampler"])
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
                    train_test_data_loader,
                    args,
                    epoch,
                    metrics
                )


                # TESTING AT INTERVALS
                # Determines how frequently the model is tested thorughout training
                if epoch % args.test_epochs == 0:
                    with test_data_loader.sampler.in_epoch(epoch):
                        testing_accuracy(
                            model,
                            train_data_loader,
                            test_data_loader,
                            train_test_data_loader,
                            args,
                            epoch,
                            metrics,
                            writer
                        )
                        training_accuracy(
                            model,
                            train_data_loader,
                            test_data_loader,
                            args,
                            epoch,
                            metrics,
                            writer
                        )

    else:
        # compare_datasets(train_data_set, train_test_data_set)
        # compare_dataloaders(train_data_loader, train_test_data_loader)
        
        EXP_LOG.info("Started training and testing loops.")
        for epoch in range(0, args.epochs): 
            train_loop(
                model, 
                train_data_loader,
                test_data_loader,
                train_test_data_loader,
                args,
                epoch
            )
            if epoch % args.test_epochs == 0:
                testing_accuracy(
                    model,
                    train_data_loader,
                    test_data_loader,
                    train_test_data_loader,
                    args,
                    epoch
                )
                training_accuracy(
                    model,
                    train_data_loader,
                    test_data_loader,
                    train_test_data_loader,
                    args,
                    epoch
                )
    
    EXP_LOG.info("Completed training of model.")        
    model.visualize_weights(RESULT_PATH, args.epochs, 'final')
    EXP_LOG.info("Visualize weights of model after training.")
    test_acc = testing_accuracy(model, train_data_loader, test_data_loader, train_test_data_loader, args, args.epochs)
    train_acc = training_accuracy(model, train_data_loader, test_data_loader, train_test_data_loader, args, args.epochs)
    EXP_LOG.info("Completed testing methods.")
    PARAM_LOG.info(f"Training accuracy of model after training for {args.epochs} epochs: {train_acc}")
    PARAM_LOG.info(f"Testing accuracy of model after training for {args.epochs} epochs: {test_acc}")
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
        PARAM_LOG.info(f"Start time of experiment: {time_to_str(START_TIME)}")
        
        EXP_LOG.info("Completed logging of experiment parameters.")

    # Logging start of experiment
    EXP_LOG.info("Start of experiment.")
    
    # Log arguments
    PRINT_LOG.info(f"local_machine: {ARGS.local_machine}.")

    # Run experiment
    main(ARGS)

    # End timer
    END_TIME = time.time()
    DURATION = END_TIME - START_TIME
    EXP_LOG.info(f"The experiment took {time_to_str(DURATION)} to be completed.")
    PARAM_LOG.info(f"End time of experiment: {time_to_str(END_TIME)}")
    PARAM_LOG.info(f"Runtime of experiment: {time_to_str(DURATION)}")
    PARAM_LOG.info(f"Train time of experiment: {time_to_str(TRAIN_TIME)}")
    PARAM_LOG.info(f"Test time (test acc) of experiment: {time_to_str(TEST_ACC_TIME)}")
    PARAM_LOG.info(f"Test time (train acc) of experiment: {time_to_str(TRAIN_ACC_TIME)}")
