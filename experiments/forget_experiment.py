# Built-in imports
import os
from argparse import Namespace
import ast
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

# Checking Tool Import
from collections import Counter
import matplotlib.pyplot as plt

class ForgetExperiment(Experiment):

#####################################################
# STAGE 1: experiment set up
#####################################################

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
        
        # Input layer class of model
        input_layer: Module = self.model.get_module(LayerNames.INPUT)
        self.input_class: Type[InputLayer] = globals()[input_layer.__class__.__name__]
        
        # Dataset setup
        self.train_data_set: TensorDataset = self.input_class.setup_data(self.train_data, self.train_label, self.train_fname, self.train_size, self.dataset)
        self.test_data_set: TensorDataset = self.input_class.setup_data(self.test_data, self.test_label, self.test_fname, self.test_size, self.dataset)

        # Subexperiment scope list set up
        # Convert the string argument to a list of lists
        print(args.sub_experiment_scope_list)
        self.sub_experiment_scope_list = ast.literal_eval(args.sub_experiment_scope_list)

        # Dataloader setup
        self.sub_experiemnts_train_dataloader_list: list[DataLoader] = self._setup_dataloaders(self.train_data_set, self.sub_experiment_scope_list)
        self.sub_experiemnts_test_dataloader_list: list[DataLoader] = self._setup_dataloaders(self.test_data_set, self.sub_experiment_scope_list)

        # Other attributes set up 
        self.testing_test_dataloader_list: list[DataLoader] = []
        self.TOTAL_SAMPLES: int = 1
        self.SUB_EXP_SAMPLES: int  = 1
        self.curr_folder_path: str = self.RESULT_PATH


    def _setup_dataloaders(self, input_dataset: TensorDataset, sub_experiment_scope_list: list[ list[int] ] ) -> list[DataLoader]:

        result: list[ DataLoader ] = []

        entire_dataloader: DataLoader = DataLoader(input_dataset, batch_size=self.batch_size, shuffle=True)

        for curr_subexperiment_labels in sub_experiment_scope_list:

            label_filter_dictionary = dict(zip(curr_subexperiment_labels, curr_subexperiment_labels))

            curr_sub_experiment_dataloader = self.input_class.filter_data_loader(entire_dataloader, label_filter_dictionary)

            result.append(curr_sub_experiment_dataloader)
        
        return result



#TODO set up the folder logic for both forget experiment and other types of generic experiemnts
    def _setup_result_folder(self, result_path: str) -> None:

        sub_experiment_scope_list = [[0,1],[2,3],[4,5],[6,7],[8,9]]
        os.makedirs(f"{self.RESULT_PATH}", exist_ok=True)
        
        for label_value_list in sub_experiment_scope_list:
            
            # Create the subdirectory name
            subdirectory_name = f"{self.data_name}_{'_'.join(map(str, label_value_list))}"
            subdirectory_path = os.path.join(result_path, subdirectory_name)

            # Create the main subdirectory
            os.makedirs(subdirectory_path, exist_ok=True)
            
            # Create the 'hidden' and 'output' subdirectories
            os.makedirs(os.path.join(subdirectory_path, 'Hidden'), exist_ok=True)
            os.makedirs(os.path.join(subdirectory_path, 'Output'), exist_ok=True)
            

#####################################################
# STAGE 2: training and evaluation
#####################################################

    def _experiment(self) -> None:

        for step in range(len(self.sub_experiment_scope_list)):

            curr_train_dataloader: DataLoader = self.sub_experiemnts_train_dataloader_list[step]
            curr_test_dataloader: DataLoader = self.sub_experiemnts_test_dataloader_list[step]
            self.curr_folder_path: str = os.path.join(self.RESULT_PATH, f"{self.data_name}_{'_'.join(map(str, self.sub_experiment_scope_list[step]))}")

            self.testing_test_dataloader_list.append(curr_test_dataloader)

            for epoch in range(self.epochs):

                self._training(curr_train_dataloader, epoch, self.data_name, ExperimentPhases.FORGET)

            self.SUB_EXP_SAMPLES = 1




    def _training(self, 
                  train_data_loader: DataLoader, 
                  epoch: int, 
                  dname: str, 
                  phase: ExperimentPhases, 
                  visualize: bool = True
                  ) -> None:
        
        sub_experiment_name = self.curr_folder_path.split('/')[-1]  # Assumes '/' as the path separator.
        
        if visualize: self.model.visualize_weights(self.curr_folder_path, epoch, f"learning for {sub_experiment_name}")

        train_epoch_start: float = self.TRAIN_TIME
        
        train_start: float = time.time()
        self.EXP_LOG.info(f"Started '_training' function with {dname.upper()}.")

        # Epoch and Batch set up
        train_batches_per_epoch: int = len(train_data_loader)
        self.EXP_LOG.info(f"This training batch is epoch #{epoch} with {train_batches_per_epoch} batches of size {self.batch_size} in this epoch.")

        need_test: bool = True

        for inputs, labels in train_data_loader: 

            if need_test:
                # Pause train timer and add to total time
                train_pause_time: float = time.time()
                self.TRAIN_TIME += train_pause_time - train_start

                self._testing(train_data_loader, Purposes.TRAIN_ACCURACY, epoch, self.data_name, ExperimentPhases.FORGET)

                for curr_test_dataloader in self.testing_test_dataloader_list:

                    self._testing(curr_test_dataloader, Purposes.TEST_ACCURACY, epoch, self.data_name, ExperimentPhases.FORGET)
                
                need_test = False

                # Restart train timer
                train_start = time.time()

            # Move input and targets to device
            inputs, labels = inputs.to(self.device).float(), one_hot(labels, self.model.output_dim).squeeze().to(self.device).float()
            
            # Forward pass
            self.model.train()
            self.model(inputs, clamped_output=labels)
            
            # Increment samples seen
            self.TOTAL_SAMPLES += 1
            self.SUB_EXP_SAMPLES += 1
        
        train_end: float = time.time()
        self.TRAIN_TIME += train_end - train_start
        train_epoch_end: float = self.TRAIN_TIME
        training_time = train_epoch_end - train_epoch_start
        
        self.EXP_LOG.info(f"Training of epoch #{epoch} took {time_to_str(training_time)}.")
        self.EXP_LOG.info("Completed '_training' function for forget experiment")

    def _testing(self, 
                 test_data_loader: DataLoader, 
                 purpose: Purposes, 
                 epoch: int, 
                 dname: str, 
                 phase: ExperimentPhases,
                 visualize: bool = True,
                 ) -> Union[float, Tuple[float, ...]]:
        
        test_start: float = time.time()
        self.EXP_LOG.info(f"Started '_testing' function with {dname.upper()}.")

        sub_experiment_name = self.curr_folder_path.split('/')[-1]  # Assumes '/' as the path separator.
        
        # Epoch and batch set up
        test_batches_per_epoch = len(test_data_loader)
        self.EXP_LOG.info(f"Sub-experiemnt to be tested is {sub_experiment_name} -- Number of current experiment samples seen is {self.SUB_EXP_SAMPLES} -- Number of total experiment samples seen is {self.TOTAL_SAMPLES}")
        self.EXP_LOG.info(f"This testing is with {test_batches_per_epoch} batches of size {self.batch_size} in this epoch.")
        
        # Set the model to evaluation mode - important for layers with different training / inference behaviour
        self.model.eval()
        self.EXP_LOG.info("Set the model to testing mode.")

        final_accuracy: float = 0

        with torch.no_grad():
            
            correct_test_count: int = 0

            total_test_count: int = len(test_data_loader)

            for inputs, labels in test_data_loader:

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Inference
                predictions: torch.Tensor = self.model(inputs)
                
                # Evaluates performance of model on testing dataset
                correct_test_count += (predictions.argmax(-1) == labels).type(torch.float).sum()

            final_accuracy = correct_test_count/total_test_count
                
        test_end = time.time()
        testing_time = test_end - test_start

        if purpose == Purposes.TEST_ACCURACY: 
            self.TEST_LOG.info(f'Current Experiment: {sub_experiment_name} || Current Subexperiment Samples Seen: {self.SUB_EXP_SAMPLES} || Total Samples Seen: {self.TOTAL_SAMPLES} || Test Accuracy: {final_accuracy}')
        
        if purpose == Purposes.TRAIN_ACCURACY: 
            self.TRAIN_LOG.info(f'Current Experiment: {sub_experiment_name} || Current Subexperiment Samples Seen: {self.SUB_EXP_SAMPLES} || Total Samples Seen: {self.TOTAL_SAMPLES} || Train Accuracy: {final_accuracy}')
        
        self.EXP_LOG.info(f"Completed testing with {correct_test_count} out of {total_test_count}.")
        self.EXP_LOG.info("Completed '_testing' function.")
        self.EXP_LOG.info(f"Testing ({purpose.value.lower()} acc) of sample #{self.SUB_EXP_SAMPLES} in current subexperiment took {time_to_str(testing_time)}.")

        if visualize: 
            self.model.visualize_weights(self.curr_folder_path, self.SUB_EXP_SAMPLES, purpose.name.lower())

        return final_accuracy


#####################################################
# STAGE 3: final testing and report
#####################################################

    def _final_test(self):

        list_of_train_accuracy: list = []
        list_of_test_accuracy: list = []

        for step in range(len(self.sub_experiment_scope_list)):

            curr_train_dataloader: DataLoader = self.sub_experiemnts_train_dataloader_list[step]
            curr_test_dataloader: DataLoader = self.sub_experiemnts_test_dataloader_list[step]
            self.curr_folder_path: str = self.RESULT_PATH + f"{self.data_name}_{'_'.join(map(str, self.sub_experiment_scope_list[step]))}"


            temp_test_acc: Union[float, Tuple[float, ...]] = self._testing(curr_test_dataloader, 
                                                                           Purposes.TEST_ACCURACY, 
                                                                           0,
                                                                           self.data_name, 
                                                                           ExperimentPhases.FORGET,
                                                                           visualize=False)
            temp_train_acc: Union[float, Tuple[float, ...]] = self._testing(curr_train_dataloader, 
                                                                            Purposes.TEST_ACCURACY, 
                                                                            0,
                                                                            self.data_name, 
                                                                            ExperimentPhases.FORGET, 
                                                                            visualize=False)

            list_of_test_accuracy.append(temp_test_acc)
            list_of_train_accuracy.append(temp_train_acc)

            self.SUB_EXP_SAMPLES = 1

        full_list_of_accuracy = list_of_test_accuracy + list_of_train_accuracy
        return full_list_of_accuracy

#Concatenating the lists
#    So, the entries will be the order as follows:
#    TEST ACCURACY SECTION
#        digits 0 and 1 test accuracy
#        digits 2 and 3 test accuracy
#        digits 4 and 5 test accuracy
#        digits 6 and 7 test accuracy
#        digits 8 and 9 test accuracy
#    TRAIN ACCURACY SECTION
#        digits 0 and 1 train accuracy
#        digits 2 and 3 train accuracy
#        digits 4 and 5 train accuracy
#        digits 6 and 7 train accuracy
#        digits 8 and 9 train accuracy


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
        self.PARAM_LOG.info(f"Sub experiment scope list: {self.sub_experiment_scope_list}")
        self.PARAM_LOG.info(f"Start time of experiment: {time.strftime('%Y-%m-%d %Hh:%Mm:%Ss', time.localtime(self.START_TIME))}")
        
        self.EXP_LOG.info("Completed logging of experiment parameters.")

#TODO need to fix this to be more accurate and represent the training and testing time of each individual sub experiment
    def _param_end_log(self):
        self.PARAM_LOG.info(f"End time of experiment: {time.strftime('%Y-%m-%d %Hh:%Mm:%Ss', time.localtime(self.END_TIME))}")
        self.PARAM_LOG.info(f"Runtime of experiment: {time_to_str(self.DURATION if self.DURATION != None else 0)}")
        self.PARAM_LOG.info(f"Total train time of experiment: {time_to_str(self.TRAIN_TIME)}")
        self.PARAM_LOG.info(f"Total test time (test acc) of experiment: {time_to_str(self.TEST_ACC_TIME)}")
        self.PARAM_LOG.info(f"Total test time (train acc) of experiment: {time_to_str(self.TRAIN_ACC_TIME)}")
    
    
    def _final_test_log(self, results) -> None:

        test_acc_digit_0_1 = results[0]
        test_acc_digit_2_3 = results[1]
        test_acc_digit_4_5 = results[2]
        test_acc_digit_6_7 = results[3]
        test_acc_digit_8_9 = results[4]
        train_acc_digit_0_1 = results[5]
        train_acc_digit_2_3 = results[6]
        train_acc_digit_4_5 = results[7]
        train_acc_digit_6_7 = results[8]
        train_acc_digit_8_9 = results[9]

        self.PARAM_LOG.info(f"Testing accuracy of model on digits 0 and 1 after training for {self.epochs} epochs: {test_acc_digit_0_1}")
        self.PARAM_LOG.info(f"Testing accuracy of model on digits 2 and 3 after training for {self.epochs} epochs: {test_acc_digit_2_3}")
        self.PARAM_LOG.info(f"Testing accuracy of model on digits 4 and 5 after training for {self.epochs} epochs: {test_acc_digit_4_5}")
        self.PARAM_LOG.info(f"Testing accuracy of model on digits 6 and 7 after training for {self.epochs} epochs: {test_acc_digit_6_7}")
        self.PARAM_LOG.info(f"Testing accuracy of model on digits 8 and 9 after training for {self.epochs} epochs: {test_acc_digit_8_9}")
        self.PARAM_LOG.info(f"Training accuracy of model on digits 0 and 1 after training for {self.epochs} epochs: {train_acc_digit_0_1}")
        self.PARAM_LOG.info(f"Training accuracy of model on digits 2 and 3 after training for {self.epochs} epochs: {train_acc_digit_2_3}")
        self.PARAM_LOG.info(f"Training accuracy of model on digits 4 and 5 after training for {self.epochs} epochs: {train_acc_digit_4_5}")
        self.PARAM_LOG.info(f"Training accuracy of model on digits 6 and 7 after training for {self.epochs} epochs: {train_acc_digit_6_7}")
        self.PARAM_LOG.info(f"Training accuracy of model on digits 8 and 8 after training for {self.epochs} epochs: {train_acc_digit_8_9}")























