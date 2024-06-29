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
from utils.experiment_constants import DataSetNames, ExperimentPhases, LayerNames, Purposes
from utils.experiment_logger import *
from utils.experiment_parser import *
from utils.experiment_timer import *



class GeneralizationExperiment(Experiment):
    ################################################################################################
    # Constructor Method
    ################################################################################################
    """
    CLASS
    Experiment for generalization (reconstruction/weight freezing)
    @instance attr.
        Experiment ATTR.
            model (Network): model used in experiment
            batch_size (int): size of each batch of data
            epochs (int): number of epochs to train
            test_epochs (int): interval at which testing will be done
            device (str): device that will be used for CUDA
            local_machine (bool): where code is ran
            experiment_type (ExperimentTypes): what type of experiment to be ran
            
            START_TIME (int): start time of experiment
            END_TIMER (int): end of experiment
            DURATION (int): duration of experiment
            TRAIN_TIME (int): training time
            TEST_ACC_TIME (int): testing time
            TRAIN_ACC_TIME (int): testing time
            EXP_NAME (str): experiment name
            RESULT_PATH (str): where result files will be created
            PRINT_LOG (logging.Logger): print log
            TEST_LOG (logging.Logger): log with all test accuracy results
            TRAIN_LOG (logging.Logger): log with all trainning accuracy results
            PARAM_LOG (logging.Logger): parameter log for experiment
            DEBUG_LOG (logging.Logger): debugging
            EXP_LOG (logging.Logger): logging of experiment process
        OWN ATTR.
            train_data (str): path to train data
            train_label (str): path to train label
            train_fname (str): path to train filename
            test_data (str): path to test data
            test_label (str): path to test label
            test_fname (str): path to test filename
            
            e_train_data (str): path to e_train data
            e_train_label (str): path to e_train label
            e_train_fname (str): path to e_train filename
            e_test_data (str): path to e_test data
            e_test_label (str): path to e_test label
            e_test_fname (str): path to e_test filename
            
            REC_TRAIN_TIME (float): reconstruction training time
            REC_TRAIN_ACC_TIME (float): reconstruction training accuracy test time
            REC_TEST_ACC_TIME (float): reconstruction testing accuracy test time
            FREEZE_TRAIN_TIME (float): weight freeze train time
            FREEZE_TRAIN_ACC_TIME (float): wegiht freeze training accuracy test time
            FREEZE_TEST_ACC_TIME (float): wegiht freeze testing accuracy test time   
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
        self.train_data: str = args.train_data
        self.train_label: str = args.train_label
        self.train_fname: str = args.train_filename
        self.test_data: str = args.test_data
        self.test_label: str = args.test_label
        self.test_fname: str = args.test_filename
        
        self.e_train_data: str = args.e_train_data
        self.e_train_label: str = args.e_train_label
        self.e_train_fname: str = args.e_train_filename
        self.e_test_data: str = args.e_test_data
        self.e_test_label: str = args.e_test_label
        self.e_test_fname: str = args.e_test_filename
        
        self.REC_TRAIN_TIME: float = 0
        self.REC_TRAIN_ACC_TIME: float = 0
        self.REC_TEST_ACC_TIME: float = 0
        self.FREEZE_TRAIN_TIME: float = 0
        self.FREEZE_TRAIN_ACC_TIME: float = 0
        self.FREEZE_TEST_ACC_TIME: float = 0
        
    
    
    ################################################################################################
    # Phase 1 Training and Testing: Reconstruction (Hebbian Layer)
    ################################################################################################    
    def _reconstruct_train(self, train_data_loader: DataLoader, epoch: int, sname: DataSetNames, visualize: bool = True) -> None:
        """
        METHOD
        Reconstruction training model for 1 epoch
        @param
            train_data_loader: dataloader with the training data
            epoch : training epoch current training loop is at
            visualize: if the weights of model are willing to be visualized
        @return
            None
        """
        train_start: float = time.time()
        self.EXP_LOG.info(f"Started 'reconstruct_train' function with {sname.value.lower().capitalize()}.")

        # Epoch and batch set up
        train_batches_per_epoch: int = len(train_data_loader)
        self.EXP_LOG.info(f"This training batch is epoch #{epoch} with {train_batches_per_epoch} batches of size {self.ARGS.batch_size} in this epoch.")

        # Set the model to training mode - important for layers with different training / inference behaviour
        self.model.train()
        self.EXP_LOG.info("Set the model to training mode.")

        # Loop through training batches
        for inputs, labels in train_data_loader:   
            # Move input and targets to device
            inputs, labels = inputs.to(self.device).float(), one_hot(labels, 36).squeeze().to(self.device).float()
            
            # Forward pass
            self.model(inputs, clamped_output=labels, reconstruct=True)
        
        train_end: float = time.time()
        training_time: float = train_end - train_start
        self.TRAIN_TIME += training_time
        self.REC_TRAIN_TIME += training_time
        
        if visualize: self.model.visualize_weights(self.RESULT_PATH, epoch, 'rec_train')
            
        self.EXP_LOG.info(f"Training of epoch #{epoch} took {time_to_str(training_time)}.")
        self.EXP_LOG.info("Completed 'reconstruct_train' function.")
    
    
    def _reconstruct_test(self, test_data_loader: DataLoader, purpose: Purposes, epoch: int, sname: DataSetNames) -> Tuple[float, float]:
        """
        METHOD
        Reconstruction testing of model with test dataset and determine its accuracy
        @param
            test_data_loader: dataloader containing the testing dataset
            purpose: name of set for logging purposes (test/train)
            epoch: epoch number of training iteration that is being tested on
            visualize: if the weights of model are willing to be visualized
        @return
            accuracy: float value between [0, 1] to show accuracy model got on test
        """
        test_start: float = time.time()
        self.EXP_LOG.info(f"Started 'reconstruct_test' function with {sname.value.lower().capitalize()}.")

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
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Inference
                predictions: torch.Tensor = self.model(inputs, reconstruct=True)
                weights_transpose = self.model.get_module(LayerNames.HIDDEN).fc.weight
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

        if purpose == Purposes.TEST_ACCURACY: 
            self.TEST_ACC_TIME += testing_time
            self.FREEZE_TEST_ACC_TIME += testing_time
        if purpose == Purposes.TRAIN_ACCURACY: 
            self.TRAIN_ACC_TIME += testing_time
            self.FREEZE_TRAIN_ACC_TIME += testing_time
        
        self.EXP_LOG.info("Completed 'reconstruct_test' function.")
        self.EXP_LOG.info(f"Reconstruction Testing ({purpose.value.lower()} acc) of epoch #{epoch} took {time_to_str(testing_time)}.")

        if purpose == Purposes.TEST_ACCURACY: self.TEST_LOG.info(f'Epoch Number: {epoch} || Dataset: {sname.value.lower().capitalize()} || Test Accuracy: cos-sim = {cos_error}, norm = {norm_error}')
        if purpose == Purposes.TRAIN_ACCURACY: self.TRAIN_LOG.info(f'Epoch Number: {epoch} || Dataset: {sname.value.lower().capitalize()} || Train Accuracy: cos-sim = {cos_error}, norm = {norm_error}')
        
        return (cos_error, norm_error)
    
    
    
    ################################################################################################
    # Phase 2 Training and Testing: Freezing Weights (Classification Layer)
    ################################################################################################
    def _freeze_train(self, train_data_loader: DataLoader, epoch: int, sname: DataSetNames, visualize: bool = True) -> None:
        """
        METHOD
        Freezing weights training model for 1 epoch
        @param
            train_data_loader: dataloader with the training data
            epoch : training epoch current training loop is at
            visualize: if the weights of model are willing to be visualized
        @return
            None
        """
        train_start: float = time.time()
        self.EXP_LOG.info(f"Started 'freeze_train' function with {sname.value.lower().capitalize()}.")

        # Epoch and batch set up
        train_batches_per_epoch: int = len(train_data_loader)
        self.EXP_LOG.info(f"This training batch is epoch #{epoch} with {train_batches_per_epoch} batches of size {self.ARGS.batch_size} in this epoch.")

        # Set the model to training mode - important for layers with different training / inference behaviour
        self.model.train()
        self.EXP_LOG.info("Set the model to training mode.")

        # Loop through training batches
        for inputs, labels in train_data_loader:   
            # Move input and targets to device
            inputs, labels = inputs.to(self.device).float(), one_hot(labels, 36).squeeze().to(self.device).float()
            
            # Forward pass
            self.model(inputs, clamped_output=labels, freeze=True)
        
        train_end: float = time.time()
        training_time: float = train_end - train_start
        self.TRAIN_TIME += training_time
        self.FREEZE_TRAIN_TIME += training_time
        
        if visualize: self.model.visualize_weights(self.RESULT_PATH, epoch, 'freeze_train')
            
        self.EXP_LOG.info(f"Training of epoch #{epoch} took {time_to_str(training_time)}.")
        self.EXP_LOG.info("Completed 'freeze_train' function.")
      
        
    def _freeze_test(self, test_data_loader: DataLoader, purpose: Purposes, epoch: int, sname: DataSetNames) -> float:
        """
        METHOD
        Test model with test dataset and determine its accuracy
        @param
            test_data_loader: dataloader containing the testing dataset
            purpose: name of set for logging purposes (test/train)
            epoch: epoch number of training iteration that is being tested on
            visualize: if the weights of model are willing to be visualized
        @return
            accuracy: float value between [0, 1] to show accuracy model got on test
        """
        test_start: float = time.time()
        self.EXP_LOG.info(f"Started 'freeze_test' function with {sname.value.lower().capitalize()}.")

        # Epoch and batch set up
        test_batches_per_epoch = len(test_data_loader)
        self.EXP_LOG.info(f"This testing batch is epoch #{epoch} with {test_batches_per_epoch} batches of size {self.batch_size} in this epoch.")
        
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
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # EXP_LOG.info(f"EPOCH [{epoch}] - data to device")
                
                # Inference
                predictions: torch.Tensor = self.model(inputs)
                # EXP_LOG.info(f"EPOCH [{epoch}] - inference")
                
                # Evaluates performance of model on testing dataset
                correct += (predictions.argmax(-1) == labels).type(torch.float).sum()

            final_accuracy = correct/total
                
        test_end = time.time()
        testing_time = test_end - test_start

        if purpose == Purposes.TEST_ACCURACY: 
            self.TEST_ACC_TIME += testing_time
            self.FREEZE_TEST_ACC_TIME += testing_time
        if purpose == Purposes.TRAIN_ACCURACY: 
            self.TRAIN_ACC_TIME += testing_time
            self.FREEZE_TRAIN_ACC_TIME += testing_time
        
        self.EXP_LOG.info(f"Completed testing with {correct} out of {total}.")
        self.EXP_LOG.info("Completed 'freeze_test' function.")
        self.EXP_LOG.info(f"Testing (freeze {purpose.value.lower()} acc) of epoch #{epoch} took {time_to_str(testing_time)}.")

        if purpose == Purposes.TEST_ACCURACY: self.TEST_LOG.info(f'Epoch Number: {epoch} || Dataset: {sname.value.lower().capitalize()} || Freeze Test Accuracy: {final_accuracy}')
        if purpose == Purposes.TRAIN_ACCURACY: self.TRAIN_LOG.info(f'Epoch Number: {epoch} || Dataset: {sname.value.lower().capitalize()} || Freeze Train Accuracy: {final_accuracy}')
        
        return final_accuracy
    
    

    ################################################################################################
    # Training and Testing Methods
    ################################################################################################
    def _training(self, 
                  train_data_loader: DataLoader, 
                  purpose: Purposes, 
                  epoch: int, sname: DataSetNames, 
                  phase: ExperimentPhases, 
                  visualize: bool = True
                  ) -> None:
        """
        METHOD
        Train model for 1 epoch
        @param
            train_data_loader: dataloader containing the training dataset
            purpose: name of set for logging purposes (test/train)
            epoch: epoch number of training iteration that is being tested on
            sname: dataset name
            phase: which part of experiment -> which training to do
        @return
            None
        """
        if phase == ExperimentPhases.RECONSTRUCTION:
            self._reconstruct_train(train_data_loader, purpose, epoch, sname, visualize)
        elif phase == ExperimentPhases.FREEZING_WEIGHTS:
            self._freeze_train(train_data_loader, purpose, epoch, sname, visualize)
    
    
    def _testing(self, 
                 test_data_loader: DataLoader, 
                 purpose: Purposes, 
                 epoch: int, 
                 sname: DataSetNames, 
                 phase: ExperimentPhases
                 ) -> Tuple[float, ...]:
        """
        METHOD
        Test model with test dataset and determine its accuracy
        @param
            test_data_loader: dataloader containing the testing dataset
            purpose: name of set for logging purposes (test/train)
            epoch: epoch number of training iteration that is being tested on
            sname: dataset name
            phase: which part of experiment -> which test to do
        @return
            accuracy: float value between [0, 1] to show accuracy model got on test
        """
        if phase == ExperimentPhases.RECONSTRUCTION:
            self._reconstruct_test(test_data_loader, purpose, epoch, sname)
        elif phase == ExperimentPhases.FREEZING_WEIGHTS:
            self._freeze_test(test_data_loader, purpose, epoch, sname)
    
    
    
    ################################################################################################
    # Running Experiment
    ################################################################################################
    def run(self) -> Tuple[float, ...]:
        """
        METHOD
        Runs the experiment
        @param
            None
        @return
            (...): tuple of final testing and training accuracies
        """
        # Start timer
        self.START_TIME = time.time()
        self.EXP_LOG.info("Start of experiment.")
        
        torch.device(self.device) # NOTE: Should this line be here or used where we create the experiment itself
        self.PRINT_LOG.info(f"local_machine: {self.local_machine}.")
        
        # Logging training parameters
        self.EXP_LOG.info("Started logging of experiment parameters.")
        self.PARAM_LOG.info(f"Experiment Type: {self.experiment_type.value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Input Dimension: {self.model.input_dim}")
        self.PARAM_LOG.info(f"Hebbian Layer Dimension: {self.model.heb_dim}")
        self.PARAM_LOG.info(f"Outout Dimension: {self.model.output_dim}")
        self.PARAM_LOG.info(f"Hebbian Layer Lambda: {self.model.heb_param["lamb"]}")
        self.PARAM_LOG.info(f"Hebbian Layer Gamma: {self.model.heb_param["gam"]}")
        self.PARAM_LOG.info(f"Hebbian Layer Epsilon: {self.model.heb_param["eps"]}")
        self.PARAM_LOG.info(f"Learning Rule: {self.model.heb_param["learn"].value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Inhibition Rule: {self.model.heb_param["inhib"].value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Function Type: {self.model.heb_param["func"].value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Network Learning Rate: {self.model.lr}")
        self.PARAM_LOG.info(f"Number of Epochs: {self.epochs}")
        self.PARAM_LOG.info(f"Start time of experiment: {time.strftime('%Y-%m-%d %Hh:%Mm:%Ss', time.localtime(self.START_TIME))}")
        
        self.EXP_LOG.info("Completed logging of experiment parameters.")
        
        # Get input layer class of model
        input_layer: InputLayer = self.model.get_module(LayerNames.INPUT)
        input_class: Type[InputLayer] = globals()[input_layer.__class__.__name__]
        
        # Training dataset
        train_data_set: TensorDataset = input_class.setup_data(self.train_data, self.train_label, self.train_fname, 60000)
        train_data_loader: DataLoader = DataLoader(train_data_set, batch_size=self.batch_size, shuffle=True)
        self.EXP_LOG.info("Completed setup for training dataset and dataloader.")

        # Testing dataset
        test_data_set: TensorDataset = input_class.setup_data(self.test_data, self.test_label, self.test_fname, 10000)
        test_data_loader: DataLoader = DataLoader(test_data_set, batch_size=self.batch_size, shuffle=True)
        self.EXP_LOG.info("Completed setup for testing dataset and dataloader.")
        
        # Extented training dataset
        e_train_data_set: TensorDataset = input_class.setup_data(self.e_train_data, self.e_train_label, self.e_train_fname, 60000)
        e_train_data_loader: DataLoader = DataLoader(e_train_data_set, batch_size=self.batch_size, shuffle=True)
        self.EXP_LOG.info("Completed setup for e-training dataset and dataloader.")
        
        # Extended testing dataset
        e_test_data_set: TensorDataset = input_class.setup_data(self.e_test_data, self.e_test_label, self.e_test_fname, 10000)
        e_test_data_loader: DataLoader = DataLoader(e_test_data_set, batch_size=self.batch_size, shuffle=True)
        self.EXP_LOG.info("Completed setup for e-testing dataset and dataloader.")
        
        
        self.EXP_LOG.info("Started training and testing loops.")
        
        # Reconstruction -> Hebbian layer training and testing
        for epoch in range(0, self.epochs):
            # Testing accuracy
            self._reconstruct_test(test_data_loader, Purposes.TEST_ACCURACY, epoch, DataSetNames.MNIST)
            self._reconstruct_test(e_test_data_loader, Purposes.TEST_ACCURACY, epoch, DataSetNames.E_MNIST)
            
            # Training accuracy
            self._reconstruct_test(train_data_loader, Purposes.TRAIN_ACCURACY, epoch, DataSetNames.MNIST)
            self._reconstruct_test(e_train_data_loader, Purposes.TRAIN_ACCURACY, epoch, DataSetNames.E_MNIST)
            
            # Training
            self._reconstruct_train(train_data_loader, epoch)
        
        # Freezing weights -> training classification    
        for epoch in range(0, self.epochs):
            # Testing accuracy
            self._freeze_test(test_data_loader, Purposes.TEST_ACCURACY, epoch, DataSetNames.MNIST)
            self._freeze_test(e_test_data_loader, Purposes.TEST_ACCURACY, epoch, DataSetNames.E_MNIST)
            
            # Training accuracy
            self._freeze_test(train_data_loader, Purposes.TRAIN_ACCURACY, epoch, DataSetNames.MNIST)
            self._freeze_test(e_train_data_loader, Purposes.TRAIN_ACCURACY, epoch, DataSetNames.E_MNIST)
            
            # Training
            self._freeze_train(e_train_data_loader, epoch)
            
        
        self.EXP_LOG.info("Completed training of model.")        
        self.model.visualize_weights(self.RESULT_PATH, self.epochs, 'final')
        self.EXP_LOG.info("Visualize weights of model after training.")
        
        # Final testing of model
        test_acc_mnist = self._freeze_test(test_data_loader, Purposes.TEST_ACCURACY, self.epochs, DataSetNames.MNIST)
        train_acc_mnist = self._freeze_test(train_data_loader, Purposes.TRAIN_ACCURACY, self.epochs, DataSetNames.MNIST)
        test_acc_emnist = self._freeze_test(e_test_data_loader, Purposes.TEST_ACCURACY, self.epochs, DataSetNames.E_MNIST)
        train_acc_emnist = self._freeze_test(e_train_data_loader, Purposes.TRAIN_ACCURACY, self.epochs, DataSetNames.E_MNIST)
        rec_cos_test_mnist, rec_norm_test_mnist = self._reconstruct_test(test_data_loader, Purposes.TEST_ACCURACY, self.epochs, DataSetNames.MNIST)
        rec_cos_test_emnist, rec_norm_test_emnist = self._reconstruct_test(e_test_data_loader, Purposes.TEST_ACCURACY, self.epochs, DataSetNames.E_MNIST)
        rec_cos_train_mnist, rec_norm_train_mnist = self._reconstruct_test(train_data_loader, Purposes.TRAIN_ACCURACY, self.epochs, DataSetNames.MNIST)
        rec_cos_train_emnist, rec_norm_train_emnist = self._reconstruct_test(e_train_data_loader, Purposes.TRAIN_ACCURACY, self.epochs, DataSetNames.E_MNIST)
        self.EXP_LOG.info("Completed final testing methods.")
        
        # Logging final parameters of experiment 
        self.PARAM_LOG.info(f"Training accuracy of model after training for {self.epochs} epochs: {train_acc_mnist} (MNIST)")
        self.PARAM_LOG.info(f"Testing accuracy of model after training for {self.epochs} epochs: {test_acc_mnist} (MNIST)")
        self.PARAM_LOG.info(f"Training accuracy of model after training for {self.epochs} epochs: {train_acc_emnist} (E-MNIST)")
        self.PARAM_LOG.info(f"Testing accuracy of model after training for {self.epochs} epochs: {test_acc_emnist} (E-MNIST)")
        
        self.PARAM_LOG.info(f"Reconstruction training accuracy of model after training for {self.epochs} epochs: cos = {rec_cos_train_mnist}, norm = {rec_norm_train_mnist} (MNIST)")
        self.PARAM_LOG.info(f"Reconstruction testing accuracy of model after training for {self.epochs} epochs:  cos ={rec_cos_test_mnist}, norm = {rec_norm_test_mnist} (MNIST)")
        self.PARAM_LOG.info(f"Reconstruction training accuracy of model after training for {self.epochs} epochs: cos = {rec_cos_train_emnist}, norm = {rec_norm_train_emnist} (E-MNIST)")
        self.PARAM_LOG.info(f"Reconstruction testing accuracy of model after training for {self.epochs} epochs: cos = {rec_cos_test_emnist}, norm = {rec_norm_test_emnist} (E-MNIST)")
        
        # End timer
        self.END_TIME = time.time()
        self.DURATION = self.END_TIME - self.START_TIME
        self.EXP_LOG.info(f"The experiment took {time_to_str(self.DURATION)} to be completed.")
        self.PARAM_LOG.info(f"End time of experiment: {time.strftime('%Y-%m-%d %Hh:%Mm:%Ss', time.localtime(self.END_TIME))}")
        self.PARAM_LOG.info(f"Runtime of experiment: {time_to_str(self.DURATION)}")
        self.PARAM_LOG.info(f"Total train time of experiment: {time_to_str(self.TRAIN_TIME)}")
        self.PARAM_LOG.info(f"Reconstruction train time of experiment: {time_to_str(self.REC_TRAIN_TIME)}")
        self.PARAM_LOG.info(f"Freeze train time of experiment: {time_to_str(self.FREEZE_TRAIN_TIME)}")
        self.PARAM_LOG.info(f"Total test time (test acc) of experiment: {time_to_str(self.TEST_ACC_TIME)}")
        self.PARAM_LOG.info(f"Reconstruction test time (test acc) of experiment: {time_to_str(self.REC_TEST_ACC_TIME)}")
        self.PARAM_LOG.info(f"Freeze test time (test acc) of experiment: {time_to_str(self.FREEZE_TEST_ACC_TIME)}")
        self.PARAM_LOG.info(f"Total test time (train acc) of experiment: {time_to_str(self.TRAIN_ACC_TIME)}")
        self.PARAM_LOG.info(f"Reconstruction test time (train acc) of experiment: {time_to_str(self.REC_TRAIN_ACC_TIME)}")
        self.PARAM_LOG.info(f"Freeze test time (train acc) of experiment: {time_to_str(self.FREEZE_TRAIN_ACC_TIME)}")
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