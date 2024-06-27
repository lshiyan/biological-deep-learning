# Built-in imports
import os
import shutil
import time
from abc import ABC
import argparse
from typing import Tuple, List

# Pytorch imports
import torch
from torch.utils.data import DataLoader, TensorDataset

# Custom defined model imports
from interfaces.network import Network

# Utils imports
from utils.experiment_logger import *
from utils.experiment_parser import *
from utils.experiment_timer import *


class Experiment(ABC):
    """
    INTERFACE
    Experiment for training and testing NN models with custom parameters -> all experiments must implement interface
    This will help with creating and running different types of experiments
    
    @instance attr.
        model: model used in experiment
        ARGS: arguments for experiment
        START_TIME: start time of experiment
        END_TIMER: end of experiment
        DURATION: duration of experiment
        TRAIN_TIME: training time
        TEST_ACC_TIME: testing time
        TRAIN_ACC_TIME: testing time
        EXP_NAME: experiment name
        RESULT_PATH: where result files will be created
        PRINT_LOG: print log
        TEST_LOG: log with all test accuracy results
        TRAIN_LOG: log with all trainning accuracy results
        PARAM_LOG: parameter log for experiment
        DEBUG_LOG: debugging
        EXP_LOG: logging of experiment process
    """
    def __init__(self, model: Network, args: argparse.Namespace, name: str) -> None:
        """
        CONTRUCTOR METHOD

        @param
            model: model to be trained and tested in experiment
            args: all arguments passed for experiment
            name: name of experiment
        @return
            None
        """
        # Model and parameters
        self.model: Network = model.to(args.device).float()
        self.ARGS: argparse.Namespace = args
        
        # Timers
        self.START_TIME: float = None
        self.END_TIMER: float = None
        self.DURATION: float = None
        self.TRAIN_TIME: float = 0
        self.TEST_ACC_TIME: float = 0
        self.TRAIN_ACC_TIME: float = 0
        
        # Result outputs
        self.EXP_NAME: str = name
        self.RESULT_PATH: str = f"results/experiment-{self.EXP_NAME}"
        
        if not os.path.exists(self.RESULT_PATH):
            os.makedirs(self.RESULT_PATH, exist_ok=True)
            os.makedirs(f"{self.RESULT_PATH}/Classification", exist_ok=True)
            os.makedirs(f"{self.RESULT_PATH}/Hebbian", exist_ok=True)
            print(f"Experiment '{self.EXP_NAME}' result folder created successfully.")
            print(f"Experiment '{self.EXP_NAME}/Classification' result folder created successfully.")
            print(f"Experiment '{self.EXP_NAME}/Hebbian' result folder created successfully.")
        else:
            try:
                shutil.rmtree(self.RESULT_PATH)
                print(f"Removed {self.RESULT_PATH}.")
                os.makedirs(self.RESULT_PATH, exist_ok=True)
                os.makedirs(f"{self.RESULT_PATH}/Classification", exist_ok=True)
                os.makedirs(f"{self.RESULT_PATH}/Hebbian", exist_ok=True)
                print(f"Experiment {self.EXP_NAME} result folder re-created successfully.")
                print(f"Experiment '{self.EXP_NAME}/Classification' result folder re-created successfully.")
                print(f"Experiment '{self.EXP_NAME}/Hebbian' result folder re-created successfully.")
            except OSError as e:
                print(f"Error: {e.strerror}")
        
        # Loggers for experiment
        self.loggers: List[logging.Logger] = []
        self.PRINT_LOG: logging.Logger = configure_logger("Print Log", f"{self.RESULT_PATH}/prints.log") # Replace print statements (for debugging purposes)
        self.TEST_LOG: logging.Logger = configure_logger("Test Log", f"{self.RESULT_PATH}/testing_accuracy.log") # Testing accuracy
        self.TRAIN_LOG: logging.Logger = configure_logger("Train Log", f"{self.RESULT_PATH}/training_accuracy.log") # Training accuracy
        self.PARAM_LOG: logging.Logger = configure_logger("Parameter Log", f"{self.RESULT_PATH}/parameters.log") # Experiment parameters
        self.DEBUG_LOG: logging.Logger = configure_logger("Debug Log", f"{self.RESULT_PATH}/debug.log", level=logging.DEBUG) # Debugging stuff
        self.EXP_LOG: logging.Logger = configure_logger("Experiment Log", f"{self.RESULT_PATH}/experiment_process.log") # Logs during experiment
        
        self.loggers.append(self.PRINT_LOG)
        self.loggers.append(self.TEST_LOG)
        self.loggers.append(self.TRAIN_LOG)
        self.loggers.append(self.PARAM_LOG)
        self.loggers.append(self.DEBUG_LOG)
        self.loggers.append(self.EXP_LOG)

        # Logging of experiment
        self.EXP_LOG.info("Completed imports.")
        self.EXP_LOG.info("Completed log setups.")
        self.EXP_LOG.info("Completed arguments parsing.")
        self.EXP_LOG.info(f"Experiment '{self.EXP_NAME}' result folder created successfully.")

    
    def training(self, train_data_loader: DataLoader, epoch: int, visualize: bool = True) -> None:
        raise NotImplementedError("This method was not implemented.")
    
    
    def testing(self, test_data_loader: DataLoader, set_name: str, epoch: int, visualize: bool = True) -> float:
        raise NotImplementedError("This method was not implemented.")
    
    
    def run(self) -> Tuple[float, float]:
        """
        METHOD
        Runs the experiment
        @param
            None
        @return
            (test_acc, train_acc): tuple of final testing and training accuracies
        """
        # Start timer
        self.START_TIME = time.time()
        self.EXP_LOG.info("Start of experiment.")
        
        torch.device(self.ARGS.device) # NOTE: Should this line be here or used where we create the experiment itself
        self.PRINT_LOG.info(f"local_machine: {self.ARGS.local_machine}.")
        
        # Logging training parameters
        self.EXP_LOG.info("Started logging of experiment parameters.")
        self.PARAM_LOG.info(f"Input Dimension: {self.ARGS.input_dim}")
        self.PARAM_LOG.info(f"Hebbian Layer Dimension: {self.ARGS.heb_dim}")
        self.PARAM_LOG.info(f"Outout Dimension: {self.ARGS.output_dim}")
        self.PARAM_LOG.info(f"Hebbian Layer Lambda: {self.ARGS.heb_lamb}")
        self.PARAM_LOG.info(f"Hebbian Layer Gamma: {self.ARGS.heb_gam}")
        self.PARAM_LOG.info(f"Hebbian Layer Epsilon: {self.ARGS.heb_eps}")
        self.PARAM_LOG.info(f"Network Learning Rate: {self.ARGS.lr}")
        self.PARAM_LOG.info(f"Number of Epochs: {self.ARGS.epochs}")
        self.PARAM_LOG.info(f"Start time of experiment: {time.strftime('%Y-%m-%d %Hh:%Mm:%Ss', time.localtime(self.START_TIME))}")
        
        self.EXP_LOG.info("Completed logging of experiment parameters.")
        
        # Training dataset
        train_data_set: TensorDataset = self.model.get_module("Input").setup_data('train')
        train_data_loader: DataLoader = DataLoader(train_data_set, batch_size=self.ARGS.batch_size, shuffle=True)
        self.EXP_LOG.info("Completed setup for training dataset and dataloader.")

        # Testing dataset
        test_data_set: TensorDataset = self.model.get_module("Input").setup_data('test')
        test_data_loader: DataLoader = DataLoader(test_data_set, batch_size=self.ARGS.batch_size, shuffle=True)
        self.EXP_LOG.info("Completed setup for testing dataset and dataloader.")
        
        self.EXP_LOG.info("Started training and testing loops.")
        
        for epoch in range(0, self.ARGS.epochs):
            # Testing accuracy
            self.testing(test_data_loader, 'test', epoch, visualize=True)
            
            # Training accuracy
            self.testing(train_data_loader, 'train', epoch, visualize=True)
            
            # Training
            self.training(train_data_loader, epoch, visualize=True)
        
        self.EXP_LOG.info("Completed training of model.")        
        self.model.visualize_weights(self.RESULT_PATH, self.ARGS.epochs, 'final')
        self.EXP_LOG.info("Visualize weights of model after training.")
        test_acc = self.testing(test_data_loader, 'test', self.ARGS.epochs, visualize=True)
        train_acc = self.testing(train_data_loader, 'train', self.ARGS.epochs, visualize=True)
        self.EXP_LOG.info("Completed final testing methods.")
        self.PARAM_LOG.info(f"Training accuracy of model after training for {self.ARGS.epochs} epochs: {train_acc}")
        self.PARAM_LOG.info(f"Testing accuracy of model after training for {self.ARGS.epochs} epochs: {test_acc}")
        
        # End timer
        self.END_TIME = time.time()
        self.DURATION = self.END_TIME - self.START_TIME
        self.EXP_LOG.info(f"The experiment took {time_to_str(self.DURATION)} to be completed.")
        self.PARAM_LOG.info(f"End time of experiment: {time.strftime('%Y-%m-%d %Hh:%Mm:%Ss', time.localtime(self.END_TIME))}")
        self.PARAM_LOG.info(f"Runtime of experiment: {time_to_str(self.DURATION)}")
        self.PARAM_LOG.info(f"Train time of experiment: {time_to_str(self.TRAIN_TIME)}")
        self.PARAM_LOG.info(f"Test time (test acc) of experiment: {time_to_str(self.TEST_ACC_TIME)}")
        self.PARAM_LOG.info(f"Test time (train acc) of experiment: {time_to_str(self.TRAIN_ACC_TIME)}")
        self.EXP_LOG.info("The experiment has been completed.")
        
        return (test_acc, train_acc)
    
    
    def cleanup(self) -> None:
        """
        METHOD
        Cleanup used ressources
        @param
            None
        @return
            None        
        """
        for logger in self.loggers:
            close_logger(logger)