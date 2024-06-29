# Built-in imports
import os
import shutil
from abc import ABC
import argparse
from typing import Tuple, List

# Pytorch imports
from torch.utils.data import DataLoader

# Custom defined model imports
from interfaces.network import Network

# Utils imports
from utils.experiment_constants import DataSetNames, ExperimentPhases, ExperimentTypes
from utils.experiment_logger import *
from utils.experiment_parser import *
from utils.experiment_timer import *


class Experiment(ABC):
    """
    INTERFACE
    Defines a base for different experiments -> Every experiment must implement interface
    @instance attr.
        model (Network): model used in experiment
        batch_size (int): size of each batch of data
        epochs (int): number of epochs to train
        test_epochs (int): interval at which testing will be done
        device (str): device that will be used for CUDA
        local_machine (bool): where code is ran
        experiment_type (ExperimentTypes): what type of experiment to be ran
        
        START_TIME (float): start time of experiment
        END_TIMER (float): end of experiment
        DURATION (float): duration of experiment
        TRAIN_TIME (float): training time
        TEST_ACC_TIME (float): testing time
        TRAIN_ACC_TIME (float): testing time
        EXP_NAME (str): experiment name
        RESULT_PATH (str): where result files will be created
        PRINT_LOG (logging.Logger): print log
        TEST_LOG (logging.Logger): log with all test accuracy results
        TRAIN_LOG (logging.Logger): log with all trainning accuracy results
        PARAM_LOG (logging.Logger): parameter log for experiment
        DEBUG_LOG (logging.Logger): debugging
        EXP_LOG (logging.Logger): logging of experiment process
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
        experiment_mapping = {member.value.upper(): member for member in ExperimentTypes}
        
        # Experiment parameters
        self.model: Network = model.to(args.device).float()
        self.batch_size: int = args.batch_size
        self.epochs: int = args.epochs
        self.test_epochs: int = args.test_epochs
        self.device: str = args.device
        self.local_machine: bool = args.local_machine
        self.experiment_type: ExperimentTypes = experiment_mapping[args.experiment_type.upper()]
        
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
            os.makedirs(f"{self.RESULT_PATH}/Output", exist_ok=True)
            os.makedirs(f"{self.RESULT_PATH}/Hidden", exist_ok=True)
            print(f"Experiment '{self.EXP_NAME}' result folder created successfully.")
            print(f"Experiment '{self.EXP_NAME}/Output' result folder created successfully.")
            print(f"Experiment '{self.EXP_NAME}/Hidden' result folder created successfully.")
        else:
            try:
                shutil.rmtree(self.RESULT_PATH)
                print(f"Removed {self.RESULT_PATH}.")
                os.makedirs(self.RESULT_PATH, exist_ok=True)
                os.makedirs(f"{self.RESULT_PATH}/Output", exist_ok=True)
                os.makedirs(f"{self.RESULT_PATH}/Hidden", exist_ok=True)
                print(f"Experiment {self.EXP_NAME} result folder re-created successfully.")
                print(f"Experiment '{self.EXP_NAME}/Output' result folder re-created successfully.")
                print(f"Experiment '{self.EXP_NAME}/Hidden' result folder re-created successfully.")
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

    
    def _training(self, 
                  train_data_loader: DataLoader, 
                  epoch: int, sname: DataSetNames, 
                  phase: ExperimentPhases, 
                  visualize: bool = True
                  ) -> None:
        raise NotImplementedError("This method was not implemented.")
    
    
    def _testing(self, 
                 test_data_loader: DataLoader, 
                 set_name: str, 
                 epoch: int, 
                 sname: DataSetNames, 
                 phase: ExperimentPhases
                 ) -> Tuple[float, ...]:
        raise NotImplementedError("This method was not implemented.")
    
    
    def run(self) -> Tuple[float, ...]:
        raise NotImplementedError("This method was not implemented.")
        
    
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