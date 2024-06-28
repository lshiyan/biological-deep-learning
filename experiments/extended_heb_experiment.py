# Built-in imports
import time
from typing import Tuple, Type

# Pytorch imports
import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset
from torch import linalg as LA
import torch.nn as nn

# Custom defined model imports
from interfaces.experiment import Experiment
from interfaces.network import Network
from layers.input_layer import InputLayer

from layers.hopfield_sanger.hsang_input_layer import HSangInputLayer
from layers.hopfield_sigmoid.hsig_input_layer import HSigInputLayer
from layers.hopfield_YZZ.hyzz_input_layer import HYZZInputLayer

from layers.relu_sanger.rsang_input_layer import RSangInputLayer
from layers.relu_sigmoid.rsig_input_layer import RSigInputLayer
from layers.relu_YZZ.ryzz_input_layer import RYZZInputLayer

from layers.softmax_sanger.ssang_input_layer import SSangInputLayer
from layers.softmax_sigmoid.ssig_input_layer import SSigInputLayer
from layers.softmax_YZZ.syzz_input_layer import SYZZInputLayer

from layers.extended_hebbian.extended_input_layer import EHebInputLayer
from layers.extended_hebbian.extended_hebbian_layer import EHebHebbianLayer
from layers.extended_hebbian.extended_classification_layer import EHebClassificationLayer

# Utils imports
from utils.experiment_logger import *
from utils.experiment_parser import *
from utils.experiment_timer import *



class EHebExperiment(Experiment):

    """
    Class used for out-of-distribution testing
    
    """

    def __init__(self, model: Network, args: argparse.Namespace, name: str) -> None:
        """
        CONSTRUCTOR METHOD
        
        @param
            model: model to be trained and tested in experiment
            args: all arguments passed for experiment
            name: name of experiment
        @return
            None
        """
        super().__init__(model, args, name)

    
    def compute_reconstruction_norm_difference(self, data_loader: DataLoader) -> float:

        # First, I set the model to evaluation mode
        self.model.eval()

        # Now, I initialize the total error to be zero
        total_error = 0.0

        # Now, I disable the gradient calculation
        with torch.no_grad():

            # Loop through each batch in data loader

            for batch in data_loader:

                inputs, _ = batch

                inputs = inputs.to(self.ARGS.device).float()    # Move input to device 

                # The next step is to reconstruct the input from the hidden activation using the Hebbian Layer weights
                hidden_activations = self.model.get_module("Hebbian")(inputs)

                # Getting transpose weight
                W_transpose = self.model.get_module("Hebbian").fc.weights.T

                # reconstructed input using hidden activations
                reconstructed_input = torch.matmul(hidden_activations, W_transpose)

                # Calcuate the L2 norm, this is to normalize the data
                norm_reconstructed = LA.vector_norm(reconstructed_input, ord=2)
                norm_input = LA.vector_norm(inputs, ord=2)


                # Calculate the norm difference as specified by prof
                    # This should mimic: (|| (x_tilt / ||x_tilt|| ) - (x / ||x||) || )^2

                curr_error = LA.vector_norm( (reconstructed_input / norm_reconstructed)  -  (inputs / norm_input)  ) 

                curr_error_squared = torch.pow(curr_error, exp=2)
        
                # Increment total error
                total_error += curr_error_squared 
            
            # Now, I return the average reconstruction norm difference error
            return total_error / len(data_loader.dataset)

            
    
    
    def compute_reconstruction_cosine_difference(self, data_loader: DataLoader) -> float:

        # First, I set the model to evaluation mode
        self.model.eval()

        # Now, I initialize the total error to be zero
        total_error = 0.0

        # Define the Cosine Similarity operation
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        # Now, I disable the gradient calculation
        with torch.no_grad():

            # Loop through each batch in data loader

            for batch in data_loader:

                inputs, _ = batch

                inputs = inputs.to(self.ARGS.device).float()    # Move input to device 

                # The next step is to reconstruct the input from the hidden activation using the Hebbian Layer weights
                hidden_activations = self.model.get_module("Hebbian")(inputs)

                # Getting transpose weight
                W_transpose = self.model.get_module("Hebbian").fc.weights.T

                # reconstructed input using hidden activations
                reconstructed_input = torch.matmul(hidden_activations, W_transpose)

                # Compute the cosine similarity
                cos_error = cos(inputs, reconstructed_input)

                # Convert to a scalar by taking the mean
                scalar_cos_error = cos_error.mean()

                # Increment total error
                total_error += scalar_cos_error 
        

            # Now, I return the average reconstruction norm difference error
            return total_error / len(data_loader.dataset)



    def reinitialize_classification_weights(self):

        classification_layer = self.model.get_module("Classification")

        # After retrieving the classification layer, I apply Xavier uniform initialization to the weights
        # This is done to have a brand new classification weight for my classification layer
        classification_layer.fc.weight.data = nn.init.xavier_uniform_(classification_layer.fc.weight.data)
        classification_layer.fc.bias.data = nn.init.zeros_(classification_layer.fc.bias.data)               # Honestly migth not be necessary, but done just in case 



    def training(self, train_data_loader: DataLoader, epoch: int, visualize: bool = True, in_distribution: bool = True) -> None:
        """
        METHOD
        Training model for 1 epoch
        @param
            train_data_loader: dataloader with the training data
            epoch : training epoch current training loop is at
            visualize: if the weights of model are willing to be visualized
            in_distribution: if it is currently training in distribution
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
            inputs, labels = inputs.to(self.ARGS.device).float(), one_hot(labels, 10).squeeze().to(self.ARGS.device).float()
            
            # Forward pass
            self.model(inputs, clamped_output=labels, in_distribution=in_distribution)
        
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
        self.EXP_LOG.info("Started 'testing_accuracy' function, which will test accuracy")

        # Epoch and batch set up
        test_batches_per_epoch = len(test_data_loader)
        self.EXP_LOG.info(f"ACCURACY: This testing batch is epoch #{epoch} with {test_batches_per_epoch} batches of size {self.ARGS.batch_size} in this epoch.")
        
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

                # Degubbing purposes
                # EXP_LOG.info(f"The inputs were put into the model ({labels.item()}) and {predictions.argmax(1).item()} are the predictions.")
                # DEBUG_LOG.info(f"Prediciton/Actual: {predictions.argmax(1).item()}/{labels.item()}.")
                # EXP_LOG.info(f"The number of correct predictions until now: {correct_sum} out of {total}.")

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
    



    def reconstruction_testing_norm_difference(self, test_data_loader: DataLoader, set_name: str, epoch: int, visualize: bool = True) -> float:
        """
        METHOD
        Test model for reconstruction error and uses norm difference as the error measure
        @param
            test_data_loader: dataloader containing the testing dataset
            set_name: name of set for logging purposes (test/train)
            epoch: epoch number of training iteration that is being tested on
            visualize: if the weights of model are willing to be visualized
        @return
            accuracy: float value between [0, 1] to show accuracy model got on test
        """
        #test_start: float = time.time()
        self.EXP_LOG.info("Started 'reconstruction_testing_norm_difference' function.")

        # Epoch and batch set up
        test_batches_per_epoch = len(test_data_loader)
        self.EXP_LOG.info(f"RECONSTRUCTION-NORM-ERROR: This testing batch is epoch #{epoch} with {test_batches_per_epoch} batches of size {self.ARGS.batch_size} in this epoch.")
        
        # Set the model to evaluation mode - important for layers with different training / inference behaviour
        self.model.eval()
        self.EXP_LOG.info("Set the model to testing mode.")

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode 
        total_error: float = 0.0

        with torch.no_grad():
            total: int = len(test_data_loader.dataset)


            # Loop through each batch in data loader

            for batch in test_data_loader:

                inputs, _ = batch

                inputs = inputs.to(self.ARGS.device).float()    # Move input to device 

                # The next step is to reconstruct the input from the hidden activation using the Hebbian Layer weights
                hidden_activations = self.model.get_module("Hebbian")(inputs)

                # Getting transpose weight
                W_transpose = self.model.get_module("Hebbian").fc.weights.T

                # reconstructed input using hidden activations
                reconstructed_input = torch.matmul(hidden_activations, W_transpose)

                # Calcuate the L2 norm, this is to normalize the data
                norm_reconstructed = LA.vector_norm(reconstructed_input, ord=2)
                norm_input = LA.vector_norm(inputs, ord=2)


                # Calculate the norm difference as specified by prof
                    # This should mimic: (|| (x_tilt / ||x_tilt|| ) - (x / ||x||) || )^2

                curr_error = LA.vector_norm( (reconstructed_input / norm_reconstructed)  -  (inputs / norm_input)  ) 

                curr_error_squared = torch.pow(curr_error, exp=2)
        
                # Increment total error
                total_error += curr_error_squared 

            final_average_norm_error = total_error/total
                
        # test_end = time.time()
        # testing_time = test_end - test_start

        #if set_name == 'test': self.TEST_ACC_TIME += testing_time
        #if set_name == 'train': self.TRAIN_ACC_TIME += testing_time
        
        if visualize: self.model.visualize_weights(self.RESULT_PATH, epoch, f'{set_name.lower()}_acc')
        
        self.EXP_LOG.info(f"RECONSTRUCTION-NORM-ERROR: Completed testing with an averaged norm error of {final_average_norm_error}")
        self.EXP_LOG.info("Completed 'reconstruction_testing_norm_difference' function.")
        # self.EXP_LOG.info(f"Testing ({set_name.lower()} acc) of epoch #{epoch} took {time_to_str(testing_time)}.")

        if set_name == 'test': self.TEST_LOG.info(f'Epoch Number: {epoch} || Reconstruction-Norm-Error: {final_average_norm_error}')
        if set_name == 'train': self.TRAIN_LOG.info(f'Epoch Number: {epoch} || Reconstruction-Norm-Error: {final_average_norm_error}')
        
        return final_average_norm_error
    

    def reconstruction_testing_cosine_difference(self, test_data_loader: DataLoader, set_name: str, epoch: int, visualize: bool = True) -> float:
        """
        METHOD
        Test model for reconstruction error and uses norm difference as the error measure
        @param
            test_data_loader: dataloader containing the testing dataset
            set_name: name of set for logging purposes (test/train)
            epoch: epoch number of training iteration that is being tested on
            visualize: if the weights of model are willing to be visualized
        @return
            accuracy: float value between [0, 1] to show accuracy model got on test
        """
        #test_start: float = time.time()
        self.EXP_LOG.info("Started 'reconstruction_testing_cosine_difference' function.")

        # Epoch and batch set up
        test_batches_per_epoch = len(test_data_loader)
        self.EXP_LOG.info(f"RECONSTRUCTION-COSINE-ERROR: This testing batch is epoch #{epoch} with {test_batches_per_epoch} batches of size {self.ARGS.batch_size} in this epoch.")
        
        # Set the model to evaluation mode - important for layers with different training / inference behaviour
        self.model.eval()
        self.EXP_LOG.info("Set the model to testing mode.")

        # Now, I initialize the total error to be zero
        total_error: float = 0.0

        # Define the Cosine Similarity operation
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        # Now, I disable the gradient calculation
        with torch.no_grad():
            total: int = len(test_data_loader.dataset)

            # Loop through each batch in data loader

            for batch in test_data_loader:

                inputs, _ = batch

                inputs = inputs.to(self.ARGS.device).float()    # Move input to device 

                # The next step is to reconstruct the input from the hidden activation using the Hebbian Layer weights
                hidden_activations = self.model.get_module("Hebbian")(inputs)

                # Getting transpose weight
                W_transpose = self.model.get_module("Hebbian").fc.weights.T

                # reconstructed input using hidden activations
                reconstructed_input = torch.matmul(hidden_activations, W_transpose)

                # Compute the cosine similarity
                cos_error = cos(inputs, reconstructed_input)

                # Convert to a scalar by taking the mean
                scalar_cos_error = cos_error.mean()

                # Increment total error
                total_error += scalar_cos_error 

            final_average_norm_error = total_error/total
                
        # test_end = time.time()
        # testing_time = test_end - test_start

        #if set_name == 'test': self.TEST_ACC_TIME += testing_time
        #if set_name == 'train': self.TRAIN_ACC_TIME += testing_time
        
        if visualize: self.model.visualize_weights(self.RESULT_PATH, epoch, f'{set_name.lower()}_acc')
        
        self.EXP_LOG.info(f"RECONSTRUCTION-COSINE-ERROR:: Completed testing with an averaged norm error of {final_average_norm_error}")
        self.EXP_LOG.info("Completed 'reconstruction_testing_cosine_difference' function.")
        # self.EXP_LOG.info(f"Testing ({set_name.lower()} acc) of epoch #{epoch} took {time_to_str(testing_time)}.")

        if set_name == 'test': self.TEST_LOG.info(f'Epoch Number: {epoch} || Reconstruction-Cosine-Error: {final_average_norm_error}')
        if set_name == 'train': self.TRAIN_LOG.info(f'Epoch Number: {epoch} || Reconstruction-Cosine-Error: {final_average_norm_error}')
        
        return final_average_norm_error


















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

        # Training dataset, BOTH in distribution and OUT of distribution
        train_data_set_in_distribution: TensorDataset = input_class.setup_data(self.ARGS.train_data, 
                                                                               self.ARGS.train_label, 
                                                                               self.ARGS.train_filename, 
                                                                               'train', 
                                                                               60000, 
                                                                               in_distribution = True)
        train_data_loader_in_distribution: DataLoader = DataLoader(train_data_set_in_distribution, 
                                                                   batch_size=self.ARGS.batch_size, 
                                                                   shuffle=True)
        self.EXP_LOG.info("Completed setup for training dataset and dataloader - IN DISTRIBUTION")

        train_data_set_out_of_distribution: TensorDataset = input_class.setup_data(self.ARGS.out_distribution_train_data, 
                                                                                   self.ARGS.out_distribution_train_label, 
                                                                                   self.ARGS.out_distribution_train_filename, 
                                                                                   'train', 
                                                                                   60000, 
                                                                                   in_distribution = False)
        train_data_loader_out_of_distribution: DataLoader = DataLoader(train_data_set_out_of_distribution, 
                                                                       batch_size=self.ARGS.batch_size, 
                                                                       shuffle=True)
        self.EXP_LOG.info("Completed setup for training dataset and dataloader - OUT OF DISTRIBUTION")

        # Testing dataset, BOTH in distribution and OUT of distribution
        test_data_set_in_distribution: TensorDataset = input_class.setup_data(self.ARGS.test_data, 
                                                                              self.ARGS.test_label, 
                                                                              self.ARGS.test_filename, 
                                                                              'test', 
                                                                              10000, 
                                                                              in_distribution = True)
        test_data_loader_in_distribution: DataLoader = DataLoader(test_data_set_in_distribution, 
                                                                  batch_size=self.ARGS.batch_size, 
                                                                  shuffle=True)
        self.EXP_LOG.info("Completed setup for testing dataset and dataloader - IN DISTRIBUTION.")

        test_data_set_out_of_distribution: TensorDataset = input_class.setup_data(self.ARGS.out_distribution_test_data, 
                                                                               self.ARGS.out_distribution_test_label, 
                                                                               self.ARGS.out_distribution_test_filename, 
                                                                               'test', 
                                                                               10000, 
                                                                               in_distribution = False)
        test_data_loader_out_of_distribution: DataLoader = DataLoader(test_data_set_out_of_distribution, 
                                                                   batch_size=self.ARGS.batch_size, 
                                                                   shuffle=True)
        self.EXP_LOG.info("Completed setup for testing dataset and dataloader - OUT OF DISTRIBUTION.")


        self.EXP_LOG.info("Started training and testing loops.")
        
        for epoch in range(0, self.ARGS.epochs):

            # EVERYTHING HERE IS FOR IN DISTRIBUTION
            # Testing accuracy
            self.testing(test_data_loader_in_distribution, 'test', epoch, visualize=True)

            # Reconstruction norm difference
            self.reconstruction_testing_norm_difference(test_data_loader_in_distribution, 'train', epoch, visualize=True)

            # Reconstruction cosine difference
            self.reconstruction_testing_cosine_difference(test_data_loader_in_distribution, 'train', epoch, visualize=True)
            
            # Training accuracy
            self.testing(train_data_loader_in_distribution, 'train', epoch, visualize=True)

            # Reconstruction norm difference
            self.reconstruction_testing_norm_difference(train_data_loader_in_distribution, 'train', epoch, visualize=True)

            # Reconstruction cosine difference
            self.reconstruction_testing_cosine_difference(train_data_loader_in_distribution, 'train', epoch, visualize=True)
        
            # Training
            self.training(train_data_loader_in_distribution, epoch, visualize=True, in_distribution=True)





            # EVERYTHING HERE IS FOR OUT OF DISTRIBUTION
            # Testing accuracy
            self.testing(test_data_loader_out_of_distribution, 'test', epoch, visualize=True)

            # Reconstruction norm difference
            self.reconstruction_testing_norm_difference(test_data_loader_out_of_distribution, 'train', epoch, visualize=True)

            # Reconstruction cosine difference
            self.reconstruction_testing_cosine_difference(test_data_loader_out_of_distribution, 'train', epoch, visualize=True)
            
            # Training accuracy
            self.testing(train_data_loader_out_of_distribution, 'train', epoch, visualize=True)

            # Reconstruction norm difference
            self.reconstruction_testing_norm_difference(train_data_loader_out_of_distribution, 'train', epoch, visualize=True)

            # Reconstruction cosine difference
            self.reconstruction_testing_cosine_difference(train_data_loader_out_of_distribution, 'train', epoch, visualize=True)
            
            # Training
            self.training(train_data_loader_out_of_distribution, epoch, visualize=True, in_distribution=False)

            

        self.EXP_LOG.info("Completed training of model.")        
        self.model.visualize_weights(self.RESULT_PATH, self.ARGS.epochs, 'final')
        self.EXP_LOG.info("Visualize weights of model after training.")



        # NOW, WITH A FULLY TRAINED MODEL, I WILL IMPLEMENT THE SECOND TEST -> WHICH IS TO TRAIN WITH A FROZEN input-to-hebbian weight and reinitialized hebbian-to-classification weight
            # The goal is to evaluate how well the network can adapt to new classification tasks after freezing the learned input-to-hidden weights and reinitializing the hidden-to-output weights.

        # First, I reinitialize the classification weights
        self.reinitialize_classification_weights()





        # EVERYTHING HERE IS FOR IN DISTRIBUTION
        test_acc_in_distribution = self.testing(test_data_loader_in_distribution, 'test', self.ARGS.epochs, visualize=True)
        test_reconstruct_norm_difference_in_distribution = self.reconstruction_testing_norm_difference(test_data_loader_in_distribution, 'train', epoch, visualize=True)
        test_reconstruct_cosine_difference_in_distribution = self.reconstruction_testing_cosine_difference(test_data_loader_in_distribution, 'train', epoch, visualize=True)

        train_acc_in_distribution = self.testing(train_data_loader_in_distribution, 'train', self.ARGS.epochs, visualize=True)
        train_reconstruct_norm_difference_in_distribution = self.reconstruction_testing_norm_difference(train_data_loader_in_distribution, 'train', epoch, visualize=True)
        train_reconstruct_cosine_difference_in_distribution = self.reconstruction_testing_cosine_difference(train_data_loader_in_distribution, 'train', epoch, visualize=True)
        self.EXP_LOG.info("Completed final testing methods.")
        self.PARAM_LOG.info(f"IN-DISTRIBUTION: Training accuracy of model after training for {self.ARGS.epochs} epochs: {train_acc_in_distribution}")
        self.PARAM_LOG.info(f"IN-DISTRIBUTION: Training average of reconstruction norm loss of model after training for {self.ARGS.epochs} epochs: {train_reconstruct_norm_difference_in_distribution}")
        self.PARAM_LOG.info(f"IN-DISTRIBUTION: Training average of reconstruction cosine loss of model after training for {self.ARGS.epochs} epochs: {train_reconstruct_cosine_difference_in_distribution}")
        self.PARAM_LOG.info(f"IN-DISTRIBUTION: Testing accuracy of model after training for {self.ARGS.epochs} epochs: {test_acc_in_distribution}")
        self.PARAM_LOG.info(f"IN-DISTRIBUTION: Testing average of reconstruction norm loss of model after training for {self.ARGS.epochs} epochs: {test_reconstruct_norm_difference_in_distribution}")
        self.PARAM_LOG.info(f"IN-DISTRIBUTION: Testing average of reconstruction cosine loss of model after training for {self.ARGS.epochs} epochs: {test_reconstruct_cosine_difference_in_distribution}")


        # EVERYTHING HERE IS FOR OUT OF DISTRIBUTION
        test_acc_out_of_distribution = self.testing(test_data_loader_out_of_distribution, 'test', self.ARGS.epochs, visualize=True)
        test_reconstruct_norm_difference_out_of_distribution = self.reconstruction_testing_norm_difference(test_data_loader_out_of_distribution, 'train', epoch, visualize=True)
        test_reconstruct_cosine_difference_out_of_distribution = self.reconstruction_testing_cosine_difference(test_data_loader_out_of_distribution, 'train', epoch, visualize=True)

        train_acc_out_of_distribution = self.testing(train_data_loader_out_of_distribution, 'train', self.ARGS.epochs, visualize=True)
        train_reconstruct_norm_difference_out_of_distribution = self.reconstruction_testing_norm_difference(train_data_loader_out_of_distribution, 'train', epoch, visualize=True)
        train_reconstruct_cosine_difference_out_of_distribution = self.reconstruction_testing_cosine_difference(train_data_loader_out_of_distribution, 'train', epoch, visualize=True)
        self.EXP_LOG.info("Completed final testing methods.")
        self.PARAM_LOG.info(f"OUT-OF-DISTRIBUTION: Training accuracy of model after training for {self.ARGS.epochs} epochs: {train_acc_out_of_distribution}")
        self.PARAM_LOG.info(f"OUT-OF-DISTRIBUTION: Training average of reconstruction norm loss of model after training for {self.ARGS.epochs} epochs: {train_reconstruct_norm_difference_out_of_distribution}")
        self.PARAM_LOG.info(f"OUT-OF-DISTRIBUTION: Training average of reconstruction cosine loss of model after training for {self.ARGS.epochs} epochs: {train_reconstruct_cosine_difference_out_of_distribution}")
        self.PARAM_LOG.info(f"OUT-OF-DISTRIBUTION: Testing accuracy of model after training for {self.ARGS.epochs} epochs: {test_acc_out_of_distribution}")
        self.PARAM_LOG.info(f"OUT-OF-DISTRIBUTION: Testing average of reconstruction norm loss of model after training for {self.ARGS.epochs} epochs: {test_reconstruct_norm_difference_out_of_distribution}")
        self.PARAM_LOG.info(f"OUT-OF-DISTRIBUTION: Testing average of reconstruction cosine loss of model after training for {self.ARGS.epochs} epochs: {test_reconstruct_cosine_difference_out_of_distribution}")



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


        return (test_acc_in_distribution,
                test_reconstruct_norm_difference_in_distribution,
                test_reconstruct_cosine_difference_in_distribution,
                train_acc_in_distribution,
                train_reconstruct_norm_difference_in_distribution,
                train_reconstruct_cosine_difference_in_distribution,
                test_acc_out_of_distribution,
                test_reconstruct_norm_difference_out_of_distribution,
                test_reconstruct_cosine_difference_out_of_distribution,
                train_acc_out_of_distribution,
                train_reconstruct_norm_difference_out_of_distribution,
                train_reconstruct_cosine_difference_out_of_distribution
                )

































































