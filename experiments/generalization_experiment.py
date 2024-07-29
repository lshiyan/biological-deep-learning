# Built-in imports
import os
from ast import Module
import time
from typing import Tuple, Type, Union

# Pytorch imports
import torch
from torch import linalg as LA
import torch.nn as nn
from torch.nn import Module
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset

# Custom defined model imports
from interfaces.experiment import Experiment
from interfaces.layer import NetworkLayer
from interfaces.network import Network
from layers.input_layer import InputLayer

from layers.base.data_setup_layer import DataSetupLayer

# Utils imports
from utils.experiment_constants import DataSets, ExperimentPhases, LayerNames, Purposes
from utils.experiment_logger import *
from utils.experiment_parser import *
from utils.experiment_timer import *

import random


class GeneralizationExperiment(Experiment):
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
            
            REC_SAMPLES (int): number of samples seen in reconstruction training
            FREEZE_SAMPLES (int): number of samples seen in freezing weights training
            
            train_data_set (TensorDataset): training dataset
            train_data_loader (DataLoader): training dataloader
            test_data_set (TensorDataset): testing dataset
            test_data_loader (DataLoader): testing dataloader
            
            e_train_data_set (TensorDataset): e_training dataset
            e_train_data_loader (DataLoader): e_training dataloader
            e_test_data_set (TensorDataset): e_testing dataset
            e_test_data_loader (DataLoader): e_testing dataloader   
    """
    ################################################################################################
    # Constructor Method
    ################################################################################################
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

        self.REC_TRAIN_TIME: float = 0
        self.REC_TRAIN_ACC_TIME: float = 0
        self.REC_TEST_ACC_TIME: float = 0
        self.FREEZE_TRAIN_TIME: float = 0
        self.FREEZE_TRAIN_ACC_TIME: float = 0
        self.FREEZE_TEST_ACC_TIME: float = 0
        
        self.REC_SAMPLES: int = 0
        self.FREEZE_SAMPLES: int = 0
        
        self.ext_data_name = args.ext_data_name.upper()
        
        dataset_mapping = {member.name.upper(): member for member in DataSets}
        self.dataset = dataset_mapping[self.data_name.upper()]
        self.ext_dataset = dataset_mapping[self.data_name.upper()]

        # Select random letter classes
        letter_labels = list(range(0, 26))
        original_class = sorted(random.sample(letter_labels, 10))
        updated_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        filter_classes = dict(zip(original_class, updated_class))
        
        # Convert Dictionary class into letters
        letter_class = [chr(65 + key) for key in original_class]
        self.letter_log = dict(zip(updated_class, letter_class))
        
        # Dataset Information
        self.train_data = args.train_data
        self.train_label = args.train_label
        self.test_data = args.test_data
        self.test_label = args.test_label
        self.train_size = args.train_size
        self.test_size = args.test_size
        self.classes = args.classes
        self.train_fname = args.train_fname
        self.test_fname = args.test_fname
        
        # EXT-Dataset Information
        self.ext_train_data = args.ext_train_data
        self.ext_train_label = args.ext_train_label
        self.ext_test_data = args.ext_test_data
        self.ext_test_label = args.ext_test_label
        self.ext_train_size = args.ext_train_size
        self.ext_test_size = args.ext_test_size
        self.ext_classes = args.ext_classes
        self.ext_train_fname = args.ext_train_fname
        self.ext_test_fname = args.ext_test_fname
        
        # Get input layer class of model
        input_layer: Module = self.model.get_module(LayerNames.INPUT)
        input_class: Type[InputLayer] = globals()[input_layer.__class__.__name__]
        
        # Training Dataset Setup
        self.train_data_set: TensorDataset = input_class.setup_data(self.train_data, self.train_label, self.train_fname, self.train_size, self.dataset)
        self.train_data_loader: DataLoader = DataLoader(self.train_data_set, batch_size=self.batch_size, shuffle=True)
        self.EXP_LOG.info("Completed setup for training dataset and dataloader.")
        
        # Testing Dataset Setup
        self.test_data_set: TensorDataset = input_class.setup_data(self.test_data, self.test_label, self.test_fname, self.test_size, self.dataset)
        self.test_data_loader: DataLoader = DataLoader(self.test_data_set, batch_size=self.batch_size, shuffle=True)
        self.EXP_LOG.info("Completed setup for testing dataset and dataloader.")
        
        # Training Dataset Setup
        self.ext_train_data_set: TensorDataset = input_class.setup_data(self.ext_train_data, self.ext_train_label, self.ext_train_fname, self.ext_train_size, self.ext_dataset)
        self.ext_train_data_loader: DataLoader = DataLoader(self.ext_train_data_set, batch_size=self.batch_size, shuffle=True)
        self.ext_train_data_loader = input_class.filter_data_loader(self.ext_train_data_loader, filter_classes)
        self.EXP_LOG.info("Completed setup for ext-training dataset and dataloader.")
        
        # Testing Dataset Setup
        self.ext_test_data_set: TensorDataset = input_class.setup_data(self.ext_test_data, self.ext_test_label, self.ext_test_fname, self.ext_test_size, self.ext_dataset)
        self.ext_test_data_loader: DataLoader = DataLoader(self.ext_test_data_set, batch_size=self.batch_size, shuffle=True)
        self.ext_test_data_loader = input_class.filter_data_loader(self.ext_test_data_loader, filter_classes)
        self.EXP_LOG.info("Completed setup for ext-testing dataset and dataloader.")


    ################################################################################################
    # Phase 1 Training and Testing: Reconstruction (Hebbian Layer)
    ################################################################################################    
    def _reconstruct_train(self, 
                           train_data_loader: DataLoader, 
                           epoch: int, 
                           dname: str, 
                           visualize: bool = True
                           ) -> None:
        """
        METHOD
        Reconstruction training of model for 1 epoch
        @param
            train_data_loader: dataloader with the training data
            epoch : training epoch current training loop is at
            sname: dataset name
            visualize: if the weights of model are willing to be visualized
        @return
            None
        """
        if visualize: self.model.visualize_weights(self.RESULT_PATH, epoch, 'rec_learning')
        
        train_start: float = time.time()
        self.EXP_LOG.info(f"Started 'reconstruct_train' function with {dname.upper()}.")

        # Epoch and batch set up
        train_batches_per_epoch: int = len(train_data_loader)
        self.EXP_LOG.info(f"This training batch is epoch #{epoch} with {train_batches_per_epoch} batches of size {self.batch_size} in this epoch.")

        # Loop through training batches
        for inputs, labels in train_data_loader:  
            # Test model at intervals of samples seen
            if self.check_test(self.REC_SAMPLES):
                self._testing(self.test_data_loader, Purposes.TEST_ACCURACY, self.data_name, ExperimentPhases.RECONSTRUCTION)
                self._testing(self.train_data_loader, Purposes.TRAIN_ACCURACY, self.data_name, ExperimentPhases.RECONSTRUCTION)
                self._testing(self.ext_test_data_loader, Purposes.TEST_ACCURACY, self.ext_data_name, ExperimentPhases.RECONSTRUCTION)
                self._testing(self.ext_train_data_loader, Purposes.TRAIN_ACCURACY, self.ext_data_name, ExperimentPhases.RECONSTRUCTION)
             
            # Move input and targets to device
            inputs, labels = inputs.to(self.device).float(), one_hot(labels, self.model.output_dim).squeeze().to(self.device).float()
            
            # Forward pass
            self.model.train()
            self.EXP_LOG.info("Set the model to training mode.")
            self.model(inputs, clamped_output=labels, reconstruct=True)
            
            # Increment samples seen
            self.REC_SAMPLES += 1
            
        train_end: float = time.time()
        training_time: float = train_end - train_start
        self.TRAIN_TIME += training_time
        self.REC_TRAIN_TIME += training_time
            
        self.EXP_LOG.info(f"Training of epoch #{epoch} took {time_to_str(training_time)}.")
        self.EXP_LOG.info("Completed 'reconstruct_train' function.")
    
    
    def _reconstruct_test(self, 
                          test_data_loader: DataLoader, 
                          purpose: Purposes,  
                          dname: str,
                          visualize: bool = True,
                          ) -> Tuple[float, float]:
        """
        METHOD
        Reconstruction testing of model with test dataset and determine its accuracy
        @param
            test_data_loader: dataloader containing the testing dataset
            purpose: name of set for logging purposes (test/train)
            dname: dataset name
        @return
            accuracy: float value between [0, 1] to show accuracy model got on test
        """
        test_start: float = time.time()
        self.EXP_LOG.info(f"Started 'reconstruct_test' function with {dname.upper()}.")

        # Epoch and batch set up
        test_batches_per_epoch = len(test_data_loader)
        self.EXP_LOG.info(f"This testing is done after samples seen #{self.REC_SAMPLES} with {test_batches_per_epoch} batches of size {self.batch_size} in this epoch.")
        
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
            total: int = len(test_data_loader) * self.batch_size

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
                total_norm_error += curr_error_squared.item()
                
                # Cosine Similarity
                cur_cos_error = cos(inputs, reconstruct_input)
                scalar_cos_error = cur_cos_error.mean()
                total_cos_error += scalar_cos_error.item()
                
            cos_error = round(total_cos_error / total, 4)
            norm_error = round(total_norm_error / total, 4)
                
        test_end = time.time()
        testing_time = test_end - test_start

        if purpose == Purposes.TEST_ACCURACY: 
            self.TEST_ACC_TIME += testing_time
            self.REC_TEST_ACC_TIME += testing_time
        if purpose == Purposes.TRAIN_ACCURACY: 
            self.TRAIN_ACC_TIME += testing_time
            self.REC_TRAIN_ACC_TIME += testing_time
        
        self.EXP_LOG.info("Completed 'reconstruct_test' function.")
        self.EXP_LOG.info(f"Reconstruction Testing ({purpose.value.lower()} acc) of sample #{self.REC_SAMPLES} took {time_to_str(testing_time)}.")

        if purpose == Purposes.TEST_ACCURACY: self.TEST_LOG.info(f'Reconstruction Samples Seen: {self.REC_SAMPLES} || Dataset: {dname.upper()} || Test Accuracy: cos-sim = {cos_error}, norm = {norm_error}')
        if purpose == Purposes.TRAIN_ACCURACY: self.TRAIN_LOG.info(f'Reconstruction Samples Seen: {self.REC_SAMPLES} || Dataset: {dname.upper()} || Train Accuracy: cos-sim = {cos_error}, norm = {norm_error}')
        
        if visualize: self.model.visualize_weights(self.RESULT_PATH, self.REC_SAMPLES, f'rec_{purpose.name.lower()}')
        
        return (cos_error, norm_error)
    
    
    
    ################################################################################################
    # Phase 2 Training and Testing: Freezing Weights (Classification Layer)
    ################################################################################################
    def _freeze_train(self, 
                      train_data_loader: DataLoader, 
                      epoch: int, 
                      dname: str, 
                      visualize: bool = True
                      ) -> None:
        """
        METHOD
        Freezing weights training of model for 1 epoch
        @param
            train_data_loader: dataloader with the training data
            epoch : training epoch current training loop is at
            dname: dataset name
            visualize: if the weights of model are willing to be visualized
        @return
            None
        """
        if visualize: self.model.visualize_weights(self.RESULT_PATH, epoch, 'freeze_learning')
        
        train_start: float = time.time()
        self.EXP_LOG.info(f"Started 'freeze_train' function with {dname.upper()}.")

        # Epoch and batch set up
        train_batches_per_epoch: int = len(train_data_loader)
        self.EXP_LOG.info(f"This training batch is epoch #{epoch} with {train_batches_per_epoch} batches of size {self.batch_size} in this epoch.")

        # Loop through training batches
        for inputs, labels in train_data_loader: 
            # Test model at intervals of samples seen
            if self.check_test(self.FREEZE_SAMPLES):
                self._testing(self.ext_test_data_loader, Purposes.TEST_ACCURACY, self.ext_data_name, ExperimentPhases.FREEZING_WEIGHTS)
                self._testing(self.ext_train_data_loader, Purposes.TRAIN_ACCURACY, self.ext_data_name, ExperimentPhases.FREEZING_WEIGHTS)
          
            # Move input and targets to device
            inputs, labels = inputs.to(self.device).float(), one_hot(labels, self.model.output_dim).squeeze().to(self.device).float()
            
            # Forward pass
            self.model.train()
            self.EXP_LOG.info("Set the model to training mode.")
            self.model(inputs, clamped_output=labels, freeze=True)
     
            # Increment samples seen
            self.FREEZE_SAMPLES += 1

        train_end: float = time.time()
        training_time: float = train_end - train_start
        self.TRAIN_TIME += training_time
        self.FREEZE_TRAIN_TIME += training_time
            
        self.EXP_LOG.info(f"Training of epoch #{epoch} took {time_to_str(training_time)}.")
        self.EXP_LOG.info("Completed 'freeze_train' function.")
      
        
    def _freeze_test(self, 
                     test_data_loader: DataLoader, 
                     purpose: Purposes,  
                     dname: str,
                     visualize: bool = True,
                     ) -> float:
        """
        METHOD
        Freezing weights testing of model with test dataset and determine its accuracy
        @param
            test_data_loader: dataloader containing the testing dataset
            purpose: name of set for logging purposes (test/train)
            dname: dataset name
        @return
            accuracy: float value between [0, 1] to show accuracy model got on test
        """
        test_start: float = time.time()
        self.EXP_LOG.info(f"Started 'freeze_test' function with {dname.upper()}.")

        # Epoch and batch set up
        test_batches_per_epoch = len(test_data_loader)
        self.EXP_LOG.info(f"This testing is done after samples seen #{self.FREEZE_SAMPLES} with {test_batches_per_epoch} batches of size {self.batch_size} in this epoch.")
        
        # Set the model to evaluation mode - important for layers with different training / inference behaviour
        self.model.eval()
        self.EXP_LOG.info("Set the model to testing mode.")

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode 
        final_accuracy: float = 0

        with torch.no_grad():
            correct: int = 0
            total: int = len(test_data_loader)

            # Loop thorugh testing batches
            for inputs, labels in test_data_loader:
                # Move input and targets to device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Inference
                predictions: torch.Tensor = self.model(inputs)
                
                # Evaluates performance of model on testing dataset
                correct += (predictions.argmax(-1) == labels).type(torch.float).sum()

            final_accuracy = round(correct/total, 4)
                
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
        self.EXP_LOG.info(f"Testing (freeze {purpose.value.lower()} acc) of sample #{self.FREEZE_SAMPLES} took {time_to_str(testing_time)}.")

        if purpose == Purposes.TEST_ACCURACY: self.TEST_LOG.info(f'Samples Seen: {self.FREEZE_SAMPLES} || Dataset: {dname.upper()} || Freeze Test Accuracy: {final_accuracy}')
        if purpose == Purposes.TRAIN_ACCURACY: self.TRAIN_LOG.info(f'Samples Seen: {self.FREEZE_SAMPLES} || Dataset: {dname.upper()} || Freeze Train Accuracy: {final_accuracy}')
        

        if visualize: self.model.visualize_weights(self.RESULT_PATH, self.REC_SAMPLES + self.FREEZE_SAMPLES, f'freeze_{purpose.name.lower()}')

        return final_accuracy
    
    

    ################################################################################################
    # Training and Testing Methods
    ################################################################################################
    def _training(self, 
                  train_data_loader: DataLoader, 
                  epoch: int, 
                  dname: str, 
                  phase: ExperimentPhases, 
                  visualize: bool = True
                  ) -> None:
        """
        METHOD
        Train model for 1 epoch
        @param
            train_data_loader: dataloader containing the training dataset
            epoch: epoch number of training iteration that is being tested on
            sname: dataset name
            phase: which part of experiment -> which training to do
        @return
            None
        """
        if phase == ExperimentPhases.RECONSTRUCTION:
            self._reconstruct_train(train_data_loader, epoch, dname, visualize)
        elif phase == ExperimentPhases.FREEZING_WEIGHTS:
            self._freeze_train(train_data_loader, epoch, dname, visualize)
    
    
    def _testing(self, 
                 test_data_loader: DataLoader, 
                 purpose: Purposes, 
                 dname: str, 
                 phase: ExperimentPhases,
                 visualize: bool = True,
                 ) -> Union[float, Tuple[float, ...]]:
        """
        METHOD
        Test model with test dataset and determine its accuracy
        @param
            test_data_loader: dataloader containing the testing dataset
            purpose: name of set for logging purposes (test/train)
            sname: dataset name
            phase: which part of experiment -> which test to do
        @return
            accuracy: float value between [0, 1] to show accuracy model got on test
        """
        if phase == ExperimentPhases.RECONSTRUCTION:
            return self._reconstruct_test(test_data_loader, purpose, dname, visualize)
        elif phase == ExperimentPhases.FREEZING_WEIGHTS:
            return self._freeze_test(test_data_loader, purpose, dname, visualize)
        else:
            raise NameError(f"Invalid phase {phase}.")
    
    
    
    ################################################################################################
    # Loggings
    ################################################################################################
    def _param_start_log(self):
        self.EXP_LOG.info("Started logging of experiment parameters.")
        
        self.PARAM_LOG.info(f"Experiment Type: {self.experiment_type.value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Device: {self.device.upper()}")
        self.PARAM_LOG.info(f"Input Dimension: {self.model.input_dim}")
        self.PARAM_LOG.info(f"Hebbian Layer Dimension: {self.model.heb_dim}")
        self.PARAM_LOG.info(f"Outout Dimension: {self.model.output_dim}")
        self.PARAM_LOG.info(f"Hebbian Layer Lambda: {self.model.heb_lamb}")
        self.PARAM_LOG.info(f"Hebbian Layer Gamma: {self.model.heb_gam}")
        self.PARAM_LOG.info(f"Hebbian Layer Epsilon: {self.model.heb_eps}")
        self.PARAM_LOG.info(f"Hebbian Layer Sigmoid K: {self.model.sig_k}")
        self.PARAM_LOG.info(f"Learning Rule: {self.model.learn.value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Inhibition Rule: {self.model.inhib.value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Weight Growth: {self.model.growth.value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Weight Decay: {self.model.weight_decay.value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Bias Update: {self.model.bias_update.value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Network Learning Rate: {self.model.lr}")
        self.PARAM_LOG.info(f"Alpha: {self.model.alpha}")
        self.PARAM_LOG.info(f"Beta: {self.model.beta}")
        self.PARAM_LOG.info(f"Sigma: {self.model.sigma}")
        self.PARAM_LOG.info(f"Mu: {self.model.mu}")
        self.PARAM_LOG.info(f"Param Init: {self.model.init.value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Selected classes: {self.letter_log}")
        self.PARAM_LOG.info(f"Start time of experiment: {time.strftime('%Y-%m-%d %Hh:%Mm:%Ss', time.localtime(self.START_TIME))}")
        
        self.EXP_LOG.info("Completed logging of experiment parameters.")
    
    
    def _param_end_log(self):
        self.PARAM_LOG.info(f"End time of experiment: {time.strftime('%Y-%m-%d %Hh:%Mm:%Ss', time.localtime(self.END_TIME))}")
        self.PARAM_LOG.info(f"Runtime of experiment: {time_to_str(self.DURATION if self.DURATION != None else 0)}")
        self.PARAM_LOG.info(f"Total train time of experiment: {time_to_str(self.TRAIN_TIME)}")
        self.PARAM_LOG.info(f"Reconstruction train time of experiment: {time_to_str(self.REC_TRAIN_TIME)}")
        self.PARAM_LOG.info(f"Freeze train time of experiment: {time_to_str(self.FREEZE_TRAIN_TIME)}")
        self.PARAM_LOG.info(f"Total test time (test acc) of experiment: {time_to_str(self.TEST_ACC_TIME)}")
        self.PARAM_LOG.info(f"Reconstruction test time (test acc) of experiment: {time_to_str(self.REC_TEST_ACC_TIME)}")
        self.PARAM_LOG.info(f"Freeze test time (test acc) of experiment: {time_to_str(self.FREEZE_TEST_ACC_TIME)}")
        self.PARAM_LOG.info(f"Total test time (train acc) of experiment: {time_to_str(self.TRAIN_ACC_TIME)}")
        self.PARAM_LOG.info(f"Reconstruction test time (train acc) of experiment: {time_to_str(self.REC_TRAIN_ACC_TIME)}")
        self.PARAM_LOG.info(f"Freeze test time (train acc) of experiment: {time_to_str(self.FREEZE_TRAIN_ACC_TIME)}")
    
    
    def _final_test_log(self, results) -> None:
        rec_cos_train_mnist = results[2]
        rec_cos_test_mnist = results[4]
        rec_cos_train_emnist = results[6]
        rec_cos_test_emnist = results[8]
        
        rec_norm_train_mnist = results[3]
        rec_norm_test_mnist = results[5]
        rec_norm_train_emnist = results[7]
        rec_norm_test_emnist = results[9]
        
        freeze_train_acc_emnist = results[0]
        freeze_test_acc_emnist = results[1]
        
        self.PARAM_LOG.info(f"Reconstruction training accuracy of model after training for {self.epochs} epochs: cos = {rec_cos_train_mnist}, norm = {rec_norm_train_mnist} ({self.data_name})")
        self.PARAM_LOG.info(f"Reconstruction testing accuracy of model after training for {self.epochs} epochs:  cos = {rec_cos_test_mnist}, norm = {rec_norm_test_mnist} ({self.data_name})")
        self.PARAM_LOG.info(f"Reconstruction training accuracy of model after training for {self.epochs} epochs: cos = {rec_cos_train_emnist}, norm = {rec_norm_train_emnist} ({self.ext_data_name})")
        self.PARAM_LOG.info(f"Reconstruction testing accuracy of model after training for {self.epochs} epochs: cos = {rec_cos_test_emnist}, norm = {rec_norm_test_emnist} ({self.ext_data_name})")
        
        self.PARAM_LOG.info(f"Freezing training accuracy of model after training for {self.epochs} epochs: {freeze_train_acc_emnist} ({self.ext_data_name})")
        self.PARAM_LOG.info(f"Freezing testing accuracy of model after training for {self.epochs} epochs: {freeze_test_acc_emnist} ({self.ext_data_name})")
    
    
    
    ################################################################################################
    # Running Experiment
    ################################################################################################
    def _experiment(self) -> None:
        torch.device(self.device)
        
        self.EXP_LOG.info("Started training and testing loops.")
        
        # Reconstruction -> Hebbian layer training and testing
        for epoch in range(0, self.epochs):
            self._training(self.train_data_loader, epoch, self.data_name, ExperimentPhases.RECONSTRUCTION)
            self._testing(self.train_data_loader, Purposes.TRAIN_ACCURACY, self.data_name, ExperimentPhases.RECONSTRUCTION)
            self._testing(self.test_data_loader, Purposes.TEST_ACCURACY, self.data_name, ExperimentPhases.RECONSTRUCTION)
            self._testing(self.ext_train_data_loader, Purposes.TRAIN_ACCURACY, self.ext_data_name, ExperimentPhases.RECONSTRUCTION)
            self._testing(self.ext_test_data_loader, Purposes.TEST_ACCURACY, self.ext_data_name, ExperimentPhases.RECONSTRUCTION)
        
        # Reset test_sample
        self.test_sample = 0
        
        # Freezing weights -> training classification    
        for epoch in range(0, self.epochs):
            self._training(self.ext_train_data_loader, epoch, self.ext_data_name, ExperimentPhases.FREEZING_WEIGHTS)
            self._testing(self.ext_train_data_loader, Purposes.TRAIN_ACCURACY, self.ext_data_name, ExperimentPhases.FREEZING_WEIGHTS)
            self._testing(self.ext_test_data_loader, Purposes.TEST_ACCURACY, self.ext_data_name, ExperimentPhases.FREEZING_WEIGHTS)

        self.EXP_LOG.info("Completed training of model.")        
        self.model.visualize_weights(self.RESULT_PATH, self.REC_SAMPLES + self.FREEZE_SAMPLES, 'final')
        self.EXP_LOG.info("Visualize weights of model after training.")
        
    
    def _final_test(self) -> Tuple[float, ...]:
        rec_cos_test_mnist, rec_norm_test_mnist = self._testing(self.test_data_loader, Purposes.TEST_ACCURACY, self.data_name, ExperimentPhases.RECONSTRUCTION) # type: ignore
        rec_cos_test_emnist, rec_norm_test_emnist = self._testing(self.ext_test_data_loader, Purposes.TEST_ACCURACY, self.ext_data_name, ExperimentPhases.RECONSTRUCTION) # type: ignore
        rec_cos_train_mnist, rec_norm_train_mnist = self._testing(self.train_data_loader, Purposes.TRAIN_ACCURACY, self.data_name, ExperimentPhases.RECONSTRUCTION) # type: ignore
        rec_cos_train_emnist, rec_norm_train_emnist = self._testing(self.ext_train_data_loader, Purposes.TRAIN_ACCURACY, self.ext_data_name, ExperimentPhases.RECONSTRUCTION) # type: ignore
        
        freeze_test_acc_emnist: float = self._testing(self.ext_test_data_loader, Purposes.TEST_ACCURACY, self.ext_data_name, ExperimentPhases.FREEZING_WEIGHTS) # type: ignore
        freeze_train_acc_emnist: float = self._testing(self.ext_train_data_loader, Purposes.TRAIN_ACCURACY, self.ext_data_name, ExperimentPhases.FREEZING_WEIGHTS) # type: ignore
        
        self.EXP_LOG.info("Completed final testing methods.")
        
        return ( 
            freeze_train_acc_emnist, 
            freeze_test_acc_emnist, 
            rec_cos_train_mnist, 
            rec_norm_train_mnist, 
            rec_cos_test_mnist, 
            rec_norm_test_mnist, 
            rec_cos_train_emnist, 
            rec_norm_train_emnist, 
            rec_cos_test_emnist, 
            rec_norm_test_emnist
        )
        
