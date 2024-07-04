# Built-in imports
import time
from typing import Tuple, Type, Union

# Pytorch imports
import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset

# Custom defined model imports
from interfaces.experiment import Experiment
from interfaces.layer import NetworkLayer
from interfaces.network import Network
from layers.input_layer import InputLayer

from layers.base.data_setup_layer import DataSetupLayer

# Utils imports
from utils.experiment_constants import ExperimentPhases, LayerNames, Purposes
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
        self.data_name: str = args.data_name
        self.train_data: str = args.train_data
        self.train_label: str = args.train_label
        self.train_fname: str = args.train_filename
        self.test_data: str = args.test_data
        self.test_label: str = args.test_label
        self.test_fname: str = args.test_filename
        
        self.SAMPLES: int = 0
        
        # Get input layer class of model
        input_layer: NetworkLayer = self.model.get_module(LayerNames.INPUT)
        input_class: Type[InputLayer] = globals()[input_layer.__class__.__name__]
        
        # Training dataset
        self.train_data_set: TensorDataset = input_class.setup_data(self.train_data, self.train_label, self.train_fname, 60000)
        self.train_data_loader: DataLoader = DataLoader(self.train_data_set, batch_size=self.batch_size, shuffle=True)
        self.EXP_LOG.info("Completed setup for training dataset and dataloader.")

        # Testing dataset
        self.test_data_set: TensorDataset = input_class.setup_data(self.test_data, self.test_label, self.test_fname, 10000)
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
        
        train_start: float = time.time()
        self.EXP_LOG.info(f"Started 'base_train' function with {dname.upper()}.")

        # Epoch and batch set up
        train_batches_per_epoch: int = len(train_data_loader)
        self.EXP_LOG.info(f"This training batch is epoch #{epoch} with {train_batches_per_epoch} batches of size {self.batch_size} in this epoch.")

        # Loop through training batches
        for inputs, labels in train_data_loader:
            # Test model at intervals of samples seen
            if self.check_test(self.SAMPLES):
                self._testing(self.test_data_loader, Purposes.TEST_ACCURACY, self.data_name, ExperimentPhases.BASE)
                self._testing(self.train_data_loader, Purposes.TRAIN_ACCURACY, self.data_name, ExperimentPhases.BASE)

            # Move input and targets to device
            inputs, labels = inputs.to(self.device).float(), one_hot(labels, self.model.output_dim).squeeze().to(self.device).float()
            
            # Forward pass
            self.model.train()
            self.EXP_LOG.info("Set the model to training mode.")
            self.model(inputs, clamped_output=labels)
            
            # Increment samples seen
            self.SAMPLES += 1
        
        train_end: float = time.time()
        training_time: float = train_end - train_start
        self.TRAIN_TIME += training_time
            
        self.EXP_LOG.info(f"Training of epoch #{epoch} took {time_to_str(training_time)}.")
        self.EXP_LOG.info("Completed 'base_train' function.")
    
    
    def _base_test(self, 
                   test_data_loader: DataLoader, 
                   purpose: Purposes,  
                   dname: str,
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
            total: int = len(test_data_loader)

            # Loop thorugh testing batches
            for inputs, labels in test_data_loader:
                # Move input and targets to device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Inference
                predictions: torch.Tensor = self.model(inputs)
                
                # Evaluates performance of model on testing dataset
                correct += (predictions.argmax(-1) == labels).type(torch.float).sum()

            final_accuracy = correct/total
                
        test_end = time.time()
        testing_time = test_end - test_start

        if purpose == Purposes.TEST_ACCURACY: self.TEST_ACC_TIME += testing_time
        if purpose == Purposes.TRAIN_ACCURACY: self.TRAIN_ACC_TIME += testing_time
        
        self.EXP_LOG.info(f"Completed testing with {correct} out of {total}.")
        self.EXP_LOG.info("Completed 'base_test' function.")
        self.EXP_LOG.info(f"Testing ({purpose.value.lower()} acc) of sample #{self.SAMPLES} took {time_to_str(testing_time)}.")
        
        if purpose == Purposes.TEST_ACCURACY: self.TEST_LOG.info(f'Samples Seen: {self.SAMPLES} || Dataset: {dname.upper()} || Test Accuracy: {final_accuracy}')
        if purpose == Purposes.TRAIN_ACCURACY: self.TRAIN_LOG.info(f'Samples Seen: {self.SAMPLES} || Dataset: {dname.upper()} || Train Accuracy: {final_accuracy}')
        
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
            return self._base_test(test_data_loader, purpose, dname)
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
        self.PARAM_LOG.info(f"Hebbian Layer Lambda: {self.model.heb_param['lamb']}")
        self.PARAM_LOG.info(f"Hebbian Layer Gamma: {self.model.heb_param['gam']}")
        self.PARAM_LOG.info(f"Hebbian Layer Epsilon: {self.model.heb_param['eps']}")
        self.PARAM_LOG.info(f"Hebbian Layer Sigmoid K: {self.model.heb_param["sig_k"]}")
        self.PARAM_LOG.info(f"Learning Rule: {self.model.heb_param['learn'].value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Inhibition Rule: {self.model.heb_param['inhib'].value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Weight Growth: {self.model.heb_param['growth'].value.lower().capitalize()}")
        self.PARAM_LOG.info(f"Network Learning Rate: {self.model.lr}")
        self.PARAM_LOG.info(f"Number of Epochs: {self.epochs}")
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
        
        self.EXP_LOG.info("Completed training of model.")        
        self.model.visualize_weights(self.RESULT_PATH, self.epochs, 'final')
        self.EXP_LOG.info("Visualize weights of model after training.")
        
    
    def _final_test(self) -> Tuple[float, ...]:
        test_acc: float = self._testing(self.test_data_loader, Purposes.TEST_ACCURACY, self.data_name, ExperimentPhases.BASE)
        train_acc: float = self._testing(self.train_data_loader, Purposes.TRAIN_ACCURACY, self.data_name, ExperimentPhases.BASE)
        self.EXP_LOG.info("Completed final testing methods.")
        return (test_acc, train_acc)
    
    