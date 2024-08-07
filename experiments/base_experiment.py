# Built-in imports
import os
import time
from typing import Tuple, Type, Union

# Pytorch imports
import torch
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



class BaseExperiment(Experiment):
    """
    CLASS
    Experiment for base training and testing of model
    @instance attr.
        Experiment ATTR.
            model (Network): model used in experiment
            batch_size (int): size of each batch of data
            epochs (int): number of epochs to train
            test_sample (int): interval at which testing will be done
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
        OWN ATTR.
            data_name (str): name of dataset
            train_data (str): path to train data
            train_label (str): path to train label
            train_fname (str): path to train filename
            test_data (str): path to test data
            test_label (str): path to test label
            test_fname (str): path to test filename
            
            SAMPLES (int): number of samples seen in training
            
            train_data_set (TensorDataset): training dataset
            train_data_loader (DataLoader): training dataloader
            test_data_set (TensorDataset): testing dataset
            test_data_loader (DataLoader): testing dataloader
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
        self.SAMPLES: int = 0
        
        dataset_mapping = {member.name.upper(): member for member in DataSets}
        self.dataset = dataset_mapping[self.data_name.upper()]
        
        self.train_data = args.train_data
        self.train_label = args.train_label
        self.test_data = args.test_data
        self.test_label = args.test_label
        self.train_size = args.train_size
        self.test_size = args.test_size
        self.classes = args.classes
        self.train_fname = args.train_fname
        self.test_fname = args.test_fname
        
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



    ################################################################################################
    # Phase 1 Training and Testing: Base (Hebbian and Classification Layers)
    ################################################################################################
    def _base_train(self, 
                    train_data_loader: DataLoader, 
                    epoch: int, 
                    dname: str, 
                    visualize: bool = True
                    ) -> None:
        """
        METHOD
        Base training of model for 1 epoch
        @param
            train_data_loader: dataloader with the training data
            epoch : training epoch current training loop is at
            dname: dataset name
            visualize: if the weights of model should be visualized
        @return
            None
        """
        if visualize: self.model.visualize_weights(self.RESULT_PATH, epoch, 'learning')

        train_epoch_start: float = self.TRAIN_TIME
        
        train_start: float = time.time()
        self.EXP_LOG.info(f"Started 'base_train' function with {dname.upper()}.")

        # Epoch and batch set up
        train_batches_per_epoch: int = len(train_data_loader)
        self.EXP_LOG.info(f"This training batch is epoch #{epoch} with {train_batches_per_epoch} batches of size {self.batch_size} in this epoch.")

        # Loop through training batches
        for inputs, labels in train_data_loader:
            # Test model at intervals of samples seen
            if self.check_test(self.SAMPLES):
                # Pause train timer and add to total time
                train_pause_time: float = time.time()
                self.TRAIN_TIME += train_pause_time - train_start

                self._testing(self.test_data_loader, Purposes.TEST_ACCURACY, self.data_name, ExperimentPhases.BASE)
                self._testing(self.train_data_loader, Purposes.TRAIN_ACCURACY, self.data_name, ExperimentPhases.BASE)

                # Restart train timer
                train_start = time.time()

            # Move input and targets to device
            inputs, labels = inputs.to(self.device).float(), one_hot(labels, self.model.output_dim).squeeze().to(self.device).float()
            
            # Forward pass
            self.model.train()
            # self.EXP_LOG.info("Set the model to training mode.")
            self.model(inputs, clamped_output=labels)
            
            # Increment samples seen
            self.SAMPLES += 1
        
        train_end: float = time.time()
        self.TRAIN_TIME += train_end - train_start
        train_epoch_end: float = self.TRAIN_TIME
        training_time = train_epoch_end - train_epoch_start
            
        self.EXP_LOG.info(f"Training of epoch #{epoch} took {time_to_str(training_time)}.")
        self.EXP_LOG.info("Completed 'base_train' function.")
    
    
    def _base_test(self, 
                   test_data_loader: DataLoader, 
                   purpose: Purposes,  
                   dname: str,
                   visualize: bool = True,
                   ) -> float:
        """
        METHOD
        Test model with test dataset and determine its accuracy
        @param
            test_data_loader: dataloader containing the testing dataset
            purpose: name of set for logging purposes (test/train)
            dname: dataset name
            last: is it final test
        @return
            accuracy: float value between [0, 1] to show accuracy model got on test
        """
        test_start: float = time.time()
        self.EXP_LOG.info(f"Started 'base_test' function with {dname.upper()}.")

        # Epoch and batch set up
        test_batches_per_epoch = len(test_data_loader)
        self.EXP_LOG.info(f"This testing is done after samples seen #{self.SAMPLES} with {test_batches_per_epoch} batches of size {self.batch_size} in this epoch.")
        
        # Set the model to evaluation mode - important for layers with different training / inference behaviour
        self.model.eval()
        self.EXP_LOG.info("Set the model to testing mode.")

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode 
        final_accuracy: float = 0

        with torch.no_grad():
            correct: int = 0
            total: int = len(test_data_loader) * self.batch_size

            # Loop thorugh testing batches
            for inputs, labels in test_data_loader:
                # Move input and targets to device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Inference
                predictions: torch.Tensor = self.model(inputs)
                
                # Evaluates performance of model on testing dataset
                correct += (predictions.argmax(-1) == labels).type(torch.float).sum().item()

            final_accuracy = round(correct/total, 4)
                
        test_end = time.time()
        testing_time = test_end - test_start

        if purpose == Purposes.TEST_ACCURACY: self.TEST_ACC_TIME += testing_time
        if purpose == Purposes.TRAIN_ACCURACY: self.TRAIN_ACC_TIME += testing_time
        
        self.EXP_LOG.info(f"Completed testing with {correct} out of {total}.")
        self.EXP_LOG.info("Completed 'base_test' function.")
        self.EXP_LOG.info(f"Testing ({purpose.value.lower()} acc) of sample #{self.SAMPLES} took {time_to_str(testing_time)}.")
        
        if purpose == Purposes.TEST_ACCURACY: self.TEST_LOG.info(f'Samples Seen: {self.SAMPLES} || Dataset: {dname.upper()} || Test Accuracy: {final_accuracy}')
        if purpose == Purposes.TRAIN_ACCURACY: self.TRAIN_LOG.info(f'Samples Seen: {self.SAMPLES} || Dataset: {dname.upper()} || Train Accuracy: {final_accuracy}')
        
        if visualize: self.model.visualize_weights(self.RESULT_PATH, self.SAMPLES, purpose.name.lower())
        
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
            dname: dataset name
            phase: which part of experiment -> which training to do
        @return
            None
        """
        if phase == ExperimentPhases.BASE:
            self._base_train(train_data_loader, epoch, dname, visualize)
          
        
    def _testing(self, 
                 test_data_loader: DataLoader, 
                 purpose: Purposes, 
                 dname: str, 
                 phase: ExperimentPhases,
                 visualize: bool = True,
                 ) -> float:
        """
        METHOD
        Test model with test dataset and determine its accuracy
        @param
            test_data_loader: dataloader containing the testing dataset
            purpose: name of set for logging purposes (test/train)
            dname: dataset name
            phase: which part of experiment -> which test to do
        @return
            accuracy: float value between [0, 1] to show accuracy model got on test
        """
        if phase == ExperimentPhases.BASE:
            return self._base_test(test_data_loader, purpose, dname, visualize)
        else:
            return 0
    


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
        self.PARAM_LOG.info(f"Hebbian Learning Rule: {self.model.heb_learn.value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Hebbian Inhibition Rule: {self.model.heb_inhib.value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Hebbian Weight Growth: {self.model.heb_growth.value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Hebbian Bias Update: {self.model.heb_bias_update.value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Hebbian Focus: {self.model.heb_focus.value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Hebbian Activation: {self.model.heb_act.value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Classification Learning Rule: {self.model.class_learn.value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Classification Weight Growth: {self.model.class_growth.value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Classification Bias Update: {self.model.class_bias_update.value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Classification Focus: {self.model.class_focus.value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Classification Activation: {self.model.class_act.value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Network Learning Rate: {self.model.lr}")
        self.PARAM_LOG.info(f"Sigmoid Constant: {self.model.sig_k}")
        self.PARAM_LOG.info(f"Alpha: {self.model.alpha}")
        self.PARAM_LOG.info(f"Beta: {self.model.beta}")
        self.PARAM_LOG.info(f"Sigma: {self.model.sigma}")
        self.PARAM_LOG.info(f"Mu: {self.model.mu}")
        self.PARAM_LOG.info(f"Param Init: {self.model.init.value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Start time of experiment: {time.strftime('%Y-%m-%d %Hh:%Mm:%Ss', time.localtime(self.START_TIME))}")
        
        self.EXP_LOG.info("Completed logging of experiment parameters.")
    
    
    def _param_end_log(self):
        self.PARAM_LOG.info(f"End time of experiment: {time.strftime('%Y-%m-%d %Hh:%Mm:%Ss', time.localtime(self.END_TIME))}")
        self.PARAM_LOG.info(f"Runtime of experiment: {time_to_str(self.DURATION if self.DURATION != None else 0)}")
        self.PARAM_LOG.info(f"Total train time of experiment: {time_to_str(self.TRAIN_TIME)}")
        self.PARAM_LOG.info(f"Total test time (test acc) of experiment: {time_to_str(self.TEST_ACC_TIME)}")
        self.PARAM_LOG.info(f"Total test time (train acc) of experiment: {time_to_str(self.TRAIN_ACC_TIME)}")
    
    
    def _final_test_log(self, results) -> None:
        test_acc, train_acc = results
        self.PARAM_LOG.info(f"Training accuracy of model after training for {self.epochs} epochs: {train_acc}")
        self.PARAM_LOG.info(f"Testing accuracy of model after training for {self.epochs} epochs: {test_acc}")
    
    
    
    ################################################################################################
    # Running Experiment
    ################################################################################################    
    def _experiment(self) -> None:
        torch.device(self.device)

        self.EXP_LOG.info("Started training and testing loops.")
        
        for epoch in range(0, self.epochs):
            self._training(self.train_data_loader, epoch, self.data_name, ExperimentPhases.BASE)
            self._testing(self.test_data_loader, Purposes.TEST_ACCURACY, self.data_name, ExperimentPhases.BASE)
            self._testing(self.train_data_loader, Purposes.TRAIN_ACCURACY, self.data_name, ExperimentPhases.BASE)
        
        self.EXP_LOG.info("Completed training of model.")        
        self.model.visualize_weights(self.RESULT_PATH, self.SAMPLES, 'final')
        self.EXP_LOG.info("Visualize weights of model after training.")
        
    
    def _final_test(self) -> Tuple[float, ...]:
        test_acc: float = self._testing(self.test_data_loader, Purposes.TEST_ACCURACY, self.data_name, ExperimentPhases.BASE, visualize=False)
        train_acc: float = self._testing(self.train_data_loader, Purposes.TRAIN_ACCURACY, self.data_name, ExperimentPhases.BASE, visualize=False)

        self.DEBUG_LOG.info(f"Current normalized weights: {self.model.get_module(LayerNames.HIDDEN).normalized_weights}")  

        test_dictionary = {}

        # First Set of Value_list tests
        values_list = [
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,17,66,138,255,253,169,138,23,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,120,228,252,252,253,252,252,252,158,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,108,252,252,252,252,190,252,252,252,252,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,43,233,252,252,252,116,5,135,252,252,252,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,43,178,253,252,221,43,2,0,5,54,232,252,210,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,93,253,255,249,115,0,0,0,0,0,136,251,255,154,0,0,0,0,0,0,0,0,0,0,0,0,0,0,166,252,253,185,0,0,0,0,0,0,0,209,253,206,0,0,0,0,0,0,0,0,0,0,0,0,0,19,220,252,253,92,0,0,0,0,0,0,0,116,253,206,0,0,0,0,0,0,0,0,0,0,0,0,0,70,252,252,192,17,0,0,0,0,0,0,0,116,253,223,25,0,0,0,0,0,0,0,0,0,0,0,0,122,252,252,63,0,0,0,0,0,0,0,0,116,253,252,69,0,0,0,0,0,0,0,0,0,0,0,0,132,253,253,0,0,0,0,0,0,0,0,0,116,255,253,69,0,0,0,0,0,0,0,0,0,0,0,0,184,252,252,0,0,0,0,0,0,0,0,0,116,253,252,69,0,0,0,0,0,0,0,0,0,0,0,0,184,252,252,0,0,0,0,0,0,0,0,0,116,253,240,50,0,0,0,0,0,0,0,0,0,0,0,0,184,252,252,0,0,0,0,0,0,0,0,0,210,253,112,0,0,0,0,0,0,0,0,0,0,0,0,0,48,232,252,158,0,0,0,0,0,0,0,0,230,232,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,93,253,244,50,0,0,0,0,0,0,155,253,168,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,34,164,253,113,0,0,0,0,0,66,236,231,42,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32,222,240,134,0,0,38,91,234,252,137,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,25,177,240,207,103,233,252,252,176,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,54,179,252,137,137,54,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,254,109,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,87,252,82,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,135,241,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,45,244,150,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,84,254,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,202,223,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32,254,216,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,95,254,195,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,140,254,77,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,57,237,205,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,124,255,165,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,171,254,81,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,232,215,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,120,254,159,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,151,254,142,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,228,254,66,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,251,254,66,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,141,254,205,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,215,254,121,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,198,176,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,116,125,171,255,255,150,93,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,169,253,253,253,253,253,253,218,30,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,169,253,253,253,213,142,176,253,253,122,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,52,250,253,210,32,12,0,6,206,253,140,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,77,251,210,25,0,0,0,122,248,253,65,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,18,0,0,0,0,209,253,253,65,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,117,247,253,198,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,76,247,253,231,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,253,253,144,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,176,246,253,159,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,25,234,253,233,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,198,253,253,141,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,78,248,253,189,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,200,253,253,141,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,134,253,253,173,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,248,253,253,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,248,253,253,43,20,20,20,20,5,0,5,20,20,37,150,150,150,147,10,0,0,0,0,0,0,0,0,0,248,253,253,253,253,253,253,253,168,143,166,253,253,253,253,253,253,253,123,0,0,0,0,0,0,0,0,0,174,253,253,253,253,253,253,253,253,253,253,253,249,247,247,169,117,117,57,0,0,0,0,0,0,0,0,0,0,118,123,123,123,166,253,253,253,155,123,123,41,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12,41,146,146,48,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12,129,253,253,253,250,163,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,253,253,253,253,253,253,229,70,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,101,253,252,145,102,107,237,253,247,128,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,181,253,167,0,0,0,61,235,253,253,163,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,253,43,0,0,0,0,58,193,253,253,164,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,187,253,32,0,0,0,0,0,55,236,253,253,86,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,146,253,32,0,100,190,87,87,87,147,253,253,123,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,94,253,78,40,248,253,253,253,253,253,253,253,223,84,15,0,0,0,0,0,0,0,0,0,0,0,0,0,14,92,12,35,240,253,253,253,253,253,253,253,253,253,244,89,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,75,161,179,253,253,253,253,253,253,253,253,253,209,43,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,16,16,39,38,16,16,145,243,253,253,185,48,0,0,0,0,0,0,0,0,0,0,0,0,0,20,58,0,0,0,0,0,0,0,0,58,209,253,253,183,0,0,0,0,0,0,0,0,0,0,0,0,77,221,247,79,0,0,0,0,0,0,0,0,13,219,253,240,72,0,0,0,0,0,0,0,0,0,0,0,90,247,253,252,57,0,0,0,0,0,0,0,0,53,251,253,191,0,0,0,0,0,0,0,0,0,0,0,0,116,253,253,59,0,0,0,0,0,0,0,0,99,252,253,145,0,0,0,0,0,0,0,0,0,0,0,0,14,188,253,221,158,38,0,0,0,0,111,211,246,253,253,145,0,0,0,0,0,0,0,0,0,0,0,0,0,12,221,246,253,251,249,249,249,249,253,253,253,253,200,19,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,95,183,228,253,253,253,253,253,253,195,124,23,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,37,138,74,126,88,37,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,168,91,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,234,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,51,254,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,178,31,0,0,0,0,0,51,254,81,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,94,254,83,0,0,0,0,0,87,254,54,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,160,254,56,0,0,0,0,0,189,238,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,227,168,2,0,0,0,0,0,194,236,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,55,254,114,0,0,0,0,0,16,235,167,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,115,254,50,0,0,0,0,0,103,254,105,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,221,236,75,156,180,190,252,252,253,254,114,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,254,254,254,252,211,179,179,179,246,254,247,94,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,217,239,117,22,0,0,0,0,226,254,242,197,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,18,0,0,0,0,0,27,243,207,46,37,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,99,254,132,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,116,254,67,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,116,254,61,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,116,254,61,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,174,255,100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,187,254,83,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,115,176,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,69,152,237,254,254,255,254,252,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,87,164,237,253,254,218,138,83,39,154,254,135,0,0,0,0,0,0,0,0,0,0,0,0,0,0,138,246,253,254,216,167,54,5,0,0,0,100,191,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,233,254,169,53,6,0,0,0,0,0,0,35,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,174,254,94,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,245,221,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,142,254,149,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,151,254,112,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,226,242,32,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12,240,203,44,44,44,44,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,70,254,254,254,254,254,254,205,85,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,37,184,169,133,133,162,212,254,254,166,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,51,177,254,125,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,27,209,254,104,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,63,209,254,194,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,0,10,137,244,254,198,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,87,122,147,223,254,247,127,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,118,250,210,248,254,252,199,49,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,138,254,254,254,250,201,72,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,79,167,197,87,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,49,180,253,244,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,184,252,252,232,164,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,66,234,252,136,38,56,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,236,252,176,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,76,252,252,56,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,139,253,173,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,212,252,69,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,116,253,240,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,157,253,206,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,230,253,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,231,255,180,138,180,253,255,253,222,97,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,230,253,252,252,252,252,211,252,252,252,117,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,230,253,240,183,89,69,7,69,171,252,252,85,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,135,253,153,0,0,0,0,0,13,215,252,116,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,116,253,206,0,0,0,0,0,0,155,252,116,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,106,255,211,7,0,0,0,0,49,233,253,116,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,159,252,154,9,0,0,30,197,252,252,95,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,17,227,252,154,70,81,228,252,227,130,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,50,227,252,252,253,252,185,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,179,252,190,117,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,21,0,0,0,11,92,173,253,254,253,254,253,62,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,142,0,21,102,213,252,233,151,131,131,253,252,142,0,0,0,0,0,0,0,0,0,0,0,0,0,0,163,152,152,214,233,183,102,0,0,0,0,132,253,255,50,0,0,0,0,0,0,0,0,0,0,0,0,82,243,253,252,131,30,0,0,0,0,0,0,10,212,253,131,0,0,0,0,0,0,0,0,0,0,0,0,123,203,102,20,0,0,0,0,0,0,0,0,0,203,255,151,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,162,253,151,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,102,254,151,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,102,253,151,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,163,254,91,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,203,253,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,203,203,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,21,223,122,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,72,253,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,233,252,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,254,233,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,41,253,151,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,123,254,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,21,223,233,30,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,51,253,183,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,51,252,20,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,75,0,98,185,178,94,19,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,111,195,238,94,0,208,249,254,254,116,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,50,107,197,246,183,25,0,0,81,245,254,249,91,0,0,0,0,0,0,0,0,0,0,0,0,18,84,230,254,254,221,86,0,0,1,125,253,254,178,53,0,0,0,0,0,0,0,0,0,0,0,0,0,133,254,254,217,118,4,0,0,62,202,254,241,131,8,0,0,0,0,0,0,0,0,0,0,0,0,0,107,244,254,213,45,0,0,0,62,240,254,220,29,0,0,0,0,0,0,0,0,0,0,0,0,0,0,44,246,254,209,49,0,0,0,31,241,254,221,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,95,254,254,55,0,0,0,17,198,254,218,28,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,27,219,254,233,144,39,42,204,254,205,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,115,248,254,254,244,233,254,223,24,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,84,168,245,254,254,254,207,115,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,236,254,230,163,237,244,211,80,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,99,254,254,99,0,0,37,225,254,130,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,71,254,229,12,0,0,0,2,170,254,51,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,96,254,254,18,0,0,0,0,81,254,198,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,80,254,254,18,0,0,0,0,131,254,119,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,214,254,78,0,0,0,1,183,244,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,253,223,24,0,2,126,254,178,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,87,238,222,119,177,254,217,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,154,196,196,101,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,36,56,137,201,199,95,37,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,45,152,234,254,254,254,254,254,250,211,151,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,46,153,240,254,254,227,166,133,251,200,254,229,225,104,0,0,0,0,0,0,0,0,0,0,0,0,0,153,234,254,254,187,142,8,0,0,191,40,198,246,223,253,21,0,0,0,0,0,0,0,0,0,0,8,126,253,254,233,128,11,0,0,0,0,210,43,70,254,254,254,21,0,0,0,0,0,0,0,0,0,0,72,243,254,228,54,0,0,0,0,3,32,116,225,242,254,255,162,5,0,0,0,0,0,0,0,0,0,0,75,240,254,223,109,138,178,178,169,210,251,231,254,254,254,232,38,0,0,0,0,0,0,0,0,0,0,0,9,175,244,253,255,254,254,251,254,254,254,254,254,252,171,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,136,195,176,146,153,200,254,254,254,254,150,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,162,254,254,241,99,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,118,250,254,254,90,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,100,242,254,254,211,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,54,241,254,254,242,59,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131,254,254,244,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,249,254,254,152,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12,228,254,254,208,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,78,255,254,254,66,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,209,254,254,137,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,227,255,233,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,113,255,108,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        ]

        # Add tensors to the dictionary and convert them to Float tensors
        for i in range(10):
            test_dictionary[i] = torch.tensor(values_list[i], dtype=torch.float)


        # Print the dictionary to verify
        for key, value in test_dictionary.items():
            self.DEBUG_LOG.info(f"Key: {key}, Value: {value}")

        self.DEBUG_LOG.info(f"====================================================================================================================================================================================")
        self.DEBUG_LOG.info(f"====================================================================================================================================================================================")
        self.DEBUG_LOG.info(f"====================================================================================================================================================================================")
        self.DEBUG_LOG.info(f"====================================================================================================================================================================================")
        self.DEBUG_LOG.info(f"====================================================================================================================================================================================")
        self.DEBUG_LOG.info(f"====================================================================================================================================================================================")

        normalized_weights = self.model.get_module(LayerNames.HIDDEN).normalized_weights  # Assuming it's a torch tensor

        max_activations = {}

        for key, tensor in test_dictionary.items():
            # Compute activations
            activations = torch.matmul(normalized_weights, tensor)

            # Find the max activation value and its index
            max_activation_value, max_activation_index = torch.max(activations, dim=0)

            max_activations[key] = (max_activation_value.item(), max_activation_index.item())
            self.DEBUG_LOG.info(f"Digit: {key}, Max Activation Value: {max_activation_value.item()}, Max Activation Index: {max_activation_index.item()}")
            self.DEBUG_LOG.info(f"Digit: {key}, Activation Values: {activations}")


        self.EXP_LOG.info("Completed final testing methods.")
        return (test_acc, train_acc)
    
    