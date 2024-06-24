# Built-in imports
import time

# Pytorch imports
import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader

# Custom defined model imports
from experiments.experiment import Experiment
from models.hebbian_network import HebbianNetwork

# Utils imports
from utils.experiment_logger import *
from utils.experiment_parser import *
from utils.experiment_timer import *


class BaseHebCpu(Experiment):
    """
    CLASS
    Experiment for base hebbian model on cpu
    
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
    def __init__(self, model: HebbianNetwork, args: argparse.Namespace, name: str) -> None:
        """
        CONTRUCTOR METHOD

        @param
            model: model to be trained and tested in experiment
            args: all arguments passed for experiment
            name: name of experiment
        @return
            None
        """
        super().__init__(model, args, name)
        
        
    def training(self, train_data_loader: DataLoader, epoch: int, visualize: bool = True) -> None:
        """
        METHOD
        Training model for 1 epoch
        @param
            train_data_loader: dataloader with the training data
            epoch : training epoch current training loop is at
            visualize: if the weights of model are willing to be visualized
        @return
            None
        """
        train_start: float = time.time()
        self.EXP_LOG.info("Started 'train_loop' function.")

        # Epoch and batch set up
        train_batches_per_epoch: int = len(train_data_loader)
        self.EXP_LOG.info(f"This training batch is epoch #{epoch} with {train_batches_per_epoch} batches of size {self.ARGS.batch_size} in this epoch.")

        # Set the model to training mode - important for layers with different training / inference behaviour
        self.model.train()
        self.EXP_LOG.info("Set the model to training mode.")

        # Loop through training batches
        for inputs, labels in train_data_loader:   
            # Move input and targets to device
            inputs, labels = inputs.to(self.ARGS.device_id).float(), one_hot(labels, 10).squeeze().to(self.ARGS.device_id).float()
            # EXP_LOG.info(f"EPOCH [{epoch}] - data to device")
            
            # Forward pass
            self.model(inputs, clamped_output=labels)
            # EXP_LOG.info(f"EPOCH [{epoch}] - forward pass")
        
        train_end: float = time.time()
        training_time: float = train_end - train_start
        self.TRAIN_TIME += training_time
        
        if visualize: self.model.visualize_weights(self.RESULT_PATH, epoch, 'training')
            
        self.EXP_LOG.info(f"Training of epoch #{epoch} took {time_to_str(training_time)}.")
        self.EXP_LOG.info("Completed 'train_loop' function.")
          
        
    def testing(self, test_data_loader: DataLoader, set_name: str, epoch: int, visualize: bool = True) -> float:
        """
        METHOD
        Test model with test dataset and determine its accuracy
        @param
            test_data_loader: dataloader containing the testing dataset
            set_name: name of set for logging purposes (test/train)
            epoch: epoch number of training iteration that is being tested on
            visualize: if the weights of model are willing to be visualized
        @return
            accuracy: float value between [0, 1] to show accuracy model got on test
        """
        test_start: float = time.time()
        self.EXP_LOG.info("Started 'testing_accuracy' function.")

        # Epoch and batch set up
        test_batches_per_epoch = len(test_data_loader)
        self.EXP_LOG.info(f"This testing batch is epoch #{epoch} with {test_batches_per_epoch} batches of size {self.ARGS.batch_size} in this epoch.")
        
        # Set the model to evaluation mode - important for layers with different training / inference behaviour
        self.model.eval()
        self.EXP_LOG.info("Set the model to testing mode.")

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode 
        final_accuracy: float = 0

        with torch.no_grad():
            correct: int = 0
            total: int = len(test_data_loader.dataset)

            # Loop thorugh testing batches
            for inputs, labels in test_data_loader:
                # Move input and targets to device
                inputs, labels = inputs.to(self.ARGS.device_id), labels.to(self.ARGS.device_id)
                # EXP_LOG.info(f"EPOCH [{epoch}] - data to device")
                
                # Inference
                predictions: torch.Tensor = self.model(inputs)
                # EXP_LOG.info(f"EPOCH [{epoch}] - inference")
                
                # Evaluates performance of model on testing dataset
                correct += (predictions.argmax(1) == labels).type(torch.float).sum()

                # Degubbing purposes
                # EXP_LOG.info(f"The inputs were put into the model ({labels.item()}) and {predictions.argmax(1).item()} are the predictions.")
                # DEBUG_LOG.info(f"Prediciton/Actual: {predictions.argmax(1).item()}/{labels.item()}.")
                # EXP_LOG.info(f"The number of correct predictions until now: {correct_sum} out of {total}.")

            final_accuracy = correct/total
                
        test_end = time.time()
        testing_time = test_end - test_start
        self.TEST_ACC_TIME += testing_time
        
        if visualize: self.model.visualize_weights(self.RESULT_PATH, epoch, f'{set_name.lower()}_acc')
        
        self.EXP_LOG.info(f"Completed testing with {correct} out of {total}.")
        self.EXP_LOG.info("Completed 'testing_accuracy' function.")
        self.EXP_LOG.info(f"Testing ({set_name.lower()} acc) of epoch #{epoch} took {time_to_str(testing_time)}.")

        if set_name == 'test': self.TEST_LOG.info(f'Epoch Number: {epoch} || Test Accuracy: {final_accuracy}')
        if set_name == 'train': self.TRAIN_LOG.info(f'Epoch Number: {epoch} || Train Accuracy: {final_accuracy}')
        
        return final_accuracy