# Built-in imports
import os
from argparse import Namespace
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
        
        # Input layer class of model
        input_layer: Module = self.model.get_module(LayerNames.INPUT)
        self.input_class: Type[InputLayer] = globals()[input_layer.__class__.__name__]
        
        # Dataset setup
        self.train_data_set: TensorDataset = self.input_class.setup_data(self.train_data, self.train_label, self.train_fname, self.train_size, self.dataset)
        self.test_data_set: TensorDataset = self.input_class.setup_data(self.test_data, self.test_label, self.test_fname, self.test_size, self.dataset)

        # Subexperiment scope list set up
        self.sub_experiment_scope_list = args.sub_experiment_scope_list

        # Dataloader setup
        self.sub_experiemnts_train_dataloader_list: list[DataLoader] = self._setup_dataloaders(self.train_data_set, self.sub_experiment_scope_list)
        self.sub_experiemnts_test_dataloader_list: list[DataLoader] = self._setup_dataloaders(self.test_data_set, self.sub_experiment_scope_list)

        # Other attributes set up 
        self.testing_test_dataloader_list: list[DataLoader] = []
        self.TOTAL_SAMPLES: int = 1
        self.SUB_EXP_SAMPLES: int  = 1

    
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
        
        for label_value_list in self.sub_experiment_scope_list:
            
            # Create the subdirectory name
            subdirectory_name = f"{self.data_name}_{'_'.join(map(str, label_value_list))}"
            subdirectory_path = os.path.join(result_path, subdirectory_name)

            # Create the main subdirectory
            os.makedirs(subdirectory_path, exist_ok=True)
            
            # Create the 'hidden' and 'output' subdirectories
            os.makedirs(os.path.join(subdirectory_path, 'hidden'), exist_ok=True)
            os.makedirs(os.path.join(subdirectory_path, 'output'), exist_ok=True)
            

    












