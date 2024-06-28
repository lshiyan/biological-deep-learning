# Built-in imports
import time
from typing import Tuple, Type

# Pytorch imports
import torch
from torch import linalg as LA
import torch.nn as nn
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset

# Custom defined model imports
from interfaces.experiment import Experiment
from interfaces.network import Network
from layers.input_layer import InputLayer

from layers.base.data_setup_layer import DataSetupLayer

# Utils imports
from utils.experiment_logger import *
from utils.experiment_parser import *
from utils.experiment_timer import *



class GeneralizationExperiment(Experiment):
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
            inputs, labels = inputs.to(self.ARGS.device).float(), one_hot(labels, 36).squeeze().to(self.ARGS.device).float()
            # EXP_LOG.info(f"EPOCH [{epoch}] - data to device")
            
            # Forward pass
            self.model(inputs, clamped_output=labels, freeze=True)
            # EXP_LOG.info(f"EPOCH [{epoch}] - forward pass")
        
        train_end: float = time.time()
        training_time: float = train_end - train_start
        self.TRAIN_TIME += training_time
        
        if visualize: self.model.visualize_weights(self.RESULT_PATH, epoch, 'training')
            
        self.EXP_LOG.info(f"Training of epoch #{epoch} took {time_to_str(training_time)}.")
        self.EXP_LOG.info("Completed 'train_loop' function.")
    
    
    def reconstruct_train(self, train_data_loader: DataLoader, epoch: int, visualize: bool = True) -> None:
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
        self.EXP_LOG.info("Started 'reconstruct_train' function.")

        # Epoch and batch set up
        train_batches_per_epoch: int = len(train_data_loader)
        self.EXP_LOG.info(f"This training batch is epoch #{epoch} with {train_batches_per_epoch} batches of size {self.ARGS.batch_size} in this epoch.")

        # Set the model to training mode - important for layers with different training / inference behaviour
        self.model.train()
        self.EXP_LOG.info("Set the model to training mode.")

        # Loop through training batches
        for inputs, labels in train_data_loader:   
            # Move input and targets to device
            inputs, labels = inputs.to(self.ARGS.device).float(), one_hot(labels, 36).squeeze().to(self.ARGS.device).float()
            
            # Forward pass
            self.model(inputs, clamped_output=labels, reconstruct=True)
        
        train_end: float = time.time()
        training_time: float = train_end - train_start
        self.TRAIN_TIME += training_time
        
        if visualize: self.model.visualize_weights(self.RESULT_PATH, epoch, 'training')
            
        self.EXP_LOG.info(f"Training of epoch #{epoch} took {time_to_str(training_time)}.")
        self.EXP_LOG.info("Completed 'reconstruct_train' function.")
          
        
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
                inputs, labels = inputs.to(self.ARGS.device), labels.to(self.ARGS.device)
                # EXP_LOG.info(f"EPOCH [{epoch}] - data to device")
                
                # Inference
                predictions: torch.Tensor = self.model(inputs)
                # EXP_LOG.info(f"EPOCH [{epoch}] - inference")
                
                # Evaluates performance of model on testing dataset
                correct += (predictions.argmax(1) == labels).type(torch.float).sum()

            final_accuracy = correct/total
                
        test_end = time.time()
        testing_time = test_end - test_start

        if set_name == 'test': self.TEST_ACC_TIME += testing_time
        if set_name == 'train': self.TRAIN_ACC_TIME += testing_time
        
        if visualize: self.model.visualize_weights(self.RESULT_PATH, epoch, f'{set_name.lower()}_acc')
        
        self.EXP_LOG.info(f"Completed testing with {correct} out of {total}.")
        self.EXP_LOG.info("Completed 'testing_accuracy' function.")
        self.EXP_LOG.info(f"Testing ({set_name.lower()} acc) of epoch #{epoch} took {time_to_str(testing_time)}.")

        if set_name == 'test': self.TEST_LOG.info(f'Epoch Number: {epoch} || Test Accuracy: {final_accuracy}')
        if set_name == 'train': self.TRAIN_LOG.info(f'Epoch Number: {epoch} || Train Accuracy: {final_accuracy}')
        
        return final_accuracy
    
    
    def reconstruct_test(self, test_data_loader: DataLoader, set_name: str, epoch: int, visualize: bool = True):
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
        self.EXP_LOG.info("Started 'reconstruct_test' function.")

        # Epoch and batch set up
        test_batches_per_epoch = len(test_data_loader)
        self.EXP_LOG.info(f"This testing batch is epoch #{epoch} with {test_batches_per_epoch} batches of size {self.ARGS.batch_size} in this epoch.")
        
        # Set the model to evaluation mode - important for layers with different training / inference behaviour
        self.model.eval()
        self.EXP_LOG.info("Set the model to testing mode.")

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode 
        norm_error: float = 0
        
        # Cosine Similarity Score
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_error: float = 0
        
        with torch.no_grad():
            total_norm_error: float = 0
            total_cos_error: float = 0
            total: int = len(test_data_loader.dataset)

            # Loop thorugh testing batches
            for inputs, labels in test_data_loader:
                # Move input and targets to device
                inputs, labels = inputs.to(self.ARGS.device), labels.to(self.ARGS.device)
                
                # Inference
                predictions: torch.Tensor = self.model(inputs, reconstruct=True)
                weights_transpose = self.model.get_module("Hidden").fc.weight
                reconstruct_input = torch.matmul(predictions, weights_transpose)
                
                # Norm difference
                norm_reconstructed = LA.vector_norm(reconstruct_input, ord=2)
                norm_input = LA.vector_norm(inputs, ord=2)
                curr_error = LA.vector_norm((reconstruct_input / norm_reconstructed) - (inputs / norm_input)) 
                curr_error_squared = curr_error ** 2
                total_norm_error += curr_error_squared
                
                # Cosine Similarity
                cur_cos_error = cos(inputs, reconstruct_input)
                scalar_cos_error = cur_cos_error.mean()
                total_cos_error += scalar_cos_error
                
            cos_error = total_cos_error / total
            norm_error = total_norm_error / total
                
        test_end = time.time()
        testing_time = test_end - test_start

        if set_name == 'test': self.TEST_ACC_TIME += testing_time
        if set_name == 'train': self.TRAIN_ACC_TIME += testing_time
        
        if visualize: self.model.visualize_weights(self.RESULT_PATH, epoch, f'{set_name.lower()}_acc')
        
        self.EXP_LOG.info("Completed 'testing_accuracy' function.")
        self.EXP_LOG.info(f"Reconstruction Testing ({set_name.lower()} acc) of epoch #{epoch} took {time_to_str(testing_time)}.")

        if set_name == 'test': self.TEST_LOG.info(f'Epoch Number: {epoch} || Test Accuracy: cos-sim = {cos_error}, norm = {norm_error}')
        if set_name == 'train': self.TRAIN_LOG.info(f'Epoch Number: {epoch} || Train Accuracy: cos-sim = {cos_error}, norm = {norm_error}')
        
        return (cos_error, norm_error)
    
    
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
        
        # Get input layer class of model
        input_layer: InputLayer = self.model.get_module("Input")
        input_class: Type[InputLayer] = globals()[input_layer.__class__.__name__]
        
        # Training dataset
        train_data_set: TensorDataset = input_class.setup_data(self.ARGS.train_data, self.ARGS.train_label, self.ARGS.train_filename, 'train', 60000)
        train_data_loader: DataLoader = DataLoader(train_data_set, batch_size=self.ARGS.batch_size, shuffle=True)
        self.EXP_LOG.info("Completed setup for training dataset and dataloader.")

        # Testing dataset
        test_data_set: TensorDataset = input_class.setup_data(self.ARGS.test_data, self.ARGS.test_label, self.ARGS.test_filename, 'test', 10000)
        test_data_loader: DataLoader = DataLoader(test_data_set, batch_size=self.ARGS.batch_size, shuffle=True)
        self.EXP_LOG.info("Completed setup for testing dataset and dataloader.")
        
        # Extented training dataset
        e_train_data_set: TensorDataset = input_class.setup_data(self.ARGS.e_train_data, self.ARGS.e_train_label, self.ARGS.e_train_filename, 'train', 60000)
        e_train_data_loader: DataLoader = DataLoader(e_train_data_set, batch_size=self.ARGS.batch_size, shuffle=True)
        self.EXP_LOG.info("Completed setup for training dataset and dataloader.")
        
        # Extended testing dataset
        e_test_data_set: TensorDataset = input_class.setup_data(self.ARGS.e_test_data, self.ARGS.e_test_label, self.ARGS.e_test_filename, 'test', 10000)
        e_test_data_loader: DataLoader = DataLoader(e_test_data_set, batch_size=self.ARGS.batch_size, shuffle=True)
        self.EXP_LOG.info("Completed setup for testing dataset and dataloader.")
        
        
        self.EXP_LOG.info("Started training and testing loops.")
        
        # Reconstruction -> Hebbian layer training and testing
        for epoch in range(0, self.ARGS.epochs):
            # Testing accuracy
            self.reconstruct_test(test_data_loader, 'test', epoch, visualize=True)
            self.reconstruct_test(e_test_data_loader, 'e_test', epoch, visualize=True)
            
            # Training accuracy
            self.reconstruct_test(train_data_loader, 'train', epoch, visualize=True)
            self.reconstruct_test(e_train_data_loader, 'e_train', epoch, visualize=True)
            
            # Training
            self.reconstruct_train(train_data_loader, epoch, visualize=True)
        
        

        
        
        # Freezing weights -> training classification    
        for epoch in range(0, self.ARGS.epochs):
            # Testing accuracy
            self.testing(test_data_loader, 'test', epoch, visualize=True)
            self.testing(e_test_data_loader, 'e_test', epoch, visualize=True)
            
            # Training accuracy
            self.testing(train_data_loader, 'train', epoch, visualize=True)
            self.testing(e_train_data_loader, 'e_train', epoch, visualize=True)
            
            # Training
            self.training(e_train_data_loader, epoch, visualize=True)
            
        
        self.EXP_LOG.info("Completed training of model.")        
        self.model.visualize_weights(self.RESULT_PATH, self.ARGS.epochs, 'final')
        self.EXP_LOG.info("Visualize weights of model after training.")
        
        # Final testing of model
        test_acc_mnist = self.testing(test_data_loader, 'test', self.ARGS.epochs, visualize=True)
        train_acc_mnist = self.testing(train_data_loader, 'train', self.ARGS.epochs, visualize=True)
        test_acc_emnist = self.testing(e_test_data_loader, 'test', self.ARGS.epochs, visualize=True)
        train_acc_emnist = self.testing(e_train_data_loader, 'train', self.ARGS.epochs, visualize=True)
        rec_cos_test_mnist, rec_norm_test_mnist = self.reconstruct_test(test_data_loader, 'test', self.ARGS.epochs, visualize=True)
        rec_cos_test_emnist, rec_norm_test_emnist = self.reconstruct_test(e_test_data_loader, 'e_test', self.ARGS.epochs, visualize=True)
        rec_cos_train_mnist, rec_norm_train_mnist = self.reconstruct_test(train_data_loader, 'train', self.ARGS.epochs, visualize=True)
        rec_cos_train_emnist, rec_norm_train_emnist = self.reconstruct_test(e_train_data_loader, 'e_train', self.ARGS.epochs, visualize=True)
        self.EXP_LOG.info("Completed final testing methods.")
        
        # Logging final parameters of experiment 
        self.PARAM_LOG.info(f"Training accuracy of model after training for {self.ARGS.epochs} epochs: {train_acc_mnist} (MNIST)")
        self.PARAM_LOG.info(f"Testing accuracy of model after training for {self.ARGS.epochs} epochs: {test_acc_mnist} (MNIST)")
        self.PARAM_LOG.info(f"Training accuracy of model after training for {self.ARGS.epochs} epochs: {train_acc_emnist} (E-MNIST)")
        self.PARAM_LOG.info(f"Testing accuracy of model after training for {self.ARGS.epochs} epochs: {test_acc_emnist} (E-MNIST)")
        
        self.PARAM_LOG.info(f"Reconstruction training accuracy of model after training for {self.ARGS.epochs} epochs: cos = {rec_cos_train_mnist}, norm = {rec_norm_train_mnist} (MNIST)")
        self.PARAM_LOG.info(f"Reconstruction testing accuracy of model after training for {self.ARGS.epochs} epochs:  cos ={rec_cos_test_mnist}, norm = {rec_norm_test_mnist} (MNIST)")
        self.PARAM_LOG.info(f"Reconstruction training accuracy of model after training for {self.ARGS.epochs} epochs: cos = {rec_cos_train_emnist}, norm = {rec_norm_train_emnist} (E-MNIST)")
        self.PARAM_LOG.info(f"Reconstruction testing accuracy of model after training for {self.ARGS.epochs} epochs: cos = {rec_cos_test_emnist}, norm = {rec_norm_test_emnist} (E-MNIST)")
        
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
        
        return (
            train_acc_mnist, 
            test_acc_mnist, 
            train_acc_emnist, 
            test_acc_emnist, 
            rec_cos_train_mnist, 
            rec_norm_train_mnist, 
            rec_cos_test_mnist, 
            rec_norm_test_mnist, 
            rec_cos_train_emnist, 
            rec_norm_train_emnist, 
            rec_cos_test_emnist, 
            rec_norm_test_emnist
        )