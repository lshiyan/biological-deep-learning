# Built-in imports
import os
import time
import random
from typing import Tuple, Type, List
import numpy as np

# Pytorch imports
import torch
from torch.nn import Module
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset

# Custom imports
from interfaces.experiment import Experiment
from interfaces.network import Network
from layers.input_layer import InputLayer

# Utils imports
from utils.experiment_constants import DataSets, ExperimentPhases, Purposes, LayerNames
from utils.experiment_logger import *
from utils.experiment_parser import *
from utils.experiment_timer import *

# Custom dataset import
from cusom_bar_matrix_dataset.bar_matrix_dataset_generator import CustomBarMatrixDataset  # Ensure this file is in the same directory or properly imported

class BarGeneralizationExperiment(Experiment):
    """
    CLASS
    Experiment for testing generalization with bar matrices.
    """

    def __init__(self, model: Network, args: argparse.Namespace, name: str) -> None:
        """
        Constructor Method
        @param
            model: model to be trained and tested in experiment
            args: all arguments passed for experiment
            name: name of experiment
        @return
            None
        """
        super().__init__(model, args, name)
        self.SAMPLES: int = 0
        
        # Define dataset parameters

        self.data_matrix_size = args.data_matrix_size
        self.samples = args.samples
        self.forbidden_combinations = [(0, 1), (2, 3)]
        
        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Generate datasets
        self._setup_datasets()

    def _setup_datasets(self):
        """
        Set up training and test datasets.
        """
        self.EXP_LOG.info("Generating training and test datasets.")
        
        # Generate training dataset
        training_set = CustomBarMatrixDataset.generate_training_set(self.data_matrix_size, self.samples, self.forbidden_combinations)
        self.train_data_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True)
        
        # Generate test set one (single bars)
        test_set_one = CustomBarMatrixDataset.generate_test_set_one(self.data_matrix_size)
        self.test_set_one_loader = DataLoader(test_set_one, batch_size=self.batch_size, shuffle=False)
        
        # Generate test set two (forbidden combinations)
        test_set_two = CustomBarMatrixDataset.generate_test_set_two(self.data_matrix_size, self.forbidden_combinations)
        self.test_set_two_loader = DataLoader(test_set_two, batch_size=self.batch_size, shuffle=False)
        
        # Generate test set three (3+ bars per side)
        test_set_three = CustomBarMatrixDataset.generate_test_set_three(self.data_matrix_size)
        self.test_set_three_loader = DataLoader(test_set_three, batch_size=self.batch_size, shuffle=False)
        
        self.EXP_LOG.info("Completed setup for all datasets and dataloaders.")

    def _training(self, 
                train_data_loader: DataLoader, 
                epoch: int, 
                dname: str, 
                phase: ExperimentPhases, 
                visualize: bool = True
                ) -> None:
        if visualize: self.model.visualize_weights(self.RESULT_PATH, epoch, 'learning')

        train_epoch_start: float = self.TRAIN_TIME
        train_start: float = time.time()
        self.EXP_LOG.info(f"Started 'training' function with {dname.upper()}.")

        train_batches_per_epoch: int = len(train_data_loader)
        self.EXP_LOG.info(f"This training batch is epoch #{epoch} with {train_batches_per_epoch} batches of size {self.batch_size} in this epoch.")

        for inputs, col_labels, row_labels in train_data_loader:
            # Combine col_labels and row_labels into a single tensor
            labels = torch.cat((col_labels, row_labels), dim=-1)

            # Flatten the input matrix to match the model's expected input size
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the input tensor

            if self.check_test(self.SAMPLES):
                train_pause_time: float = time.time()
                self.TRAIN_TIME += train_pause_time - train_start

                self._generalization_test(self.test_set_one_loader, "Test Set One", "single_bars", ExperimentPhases.BASE)
                self._generalization_test(self.test_set_two_loader, "Test Set Two", "forbidden_combinations", ExperimentPhases.BASE)
                self._generalization_test(self.test_set_three_loader, "Test Set Three", "complex_patterns", ExperimentPhases.BASE)

                self._generalization_test(self.train_data_loader, "Training Set", "Training samples", ExperimentPhases.BASE)


                train_start = time.time()

            inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float()

            self.model.train()
            self.model(inputs, clamped_output=labels)

            self.SAMPLES += 1
        
        train_end: float = time.time()
        self.TRAIN_TIME += train_end - train_start
        train_epoch_end: float = self.TRAIN_TIME
        training_time = train_epoch_end - train_epoch_start
            
        self.EXP_LOG.info(f"Training of epoch #{epoch} took {time_to_str(training_time)}.")
        self.EXP_LOG.info("Completed 'training' function.")


    def _generalization_test(self, 
                            test_data_loader: DataLoader, 
                            purpose: str, 
                            dname: str, 
                            phase: ExperimentPhases, 
                            visualize: bool = True) -> float:
        test_start: float = time.time()
        self.EXP_LOG.info(f"Started '{purpose}' function with {dname.upper()}.")

        self.model.eval()
        self.EXP_LOG.info("Set the model to testing mode.")

        final_accuracy: float = 0

        with torch.no_grad():
            correct: int = 0
            total: int = len(test_data_loader) * self.batch_size

            for batch_idx, (inputs, col_labels, row_labels) in enumerate(test_data_loader):
                labels = torch.cat((col_labels, row_labels), dim=-1)
                num_bars = torch.sum(labels, dim=1).int()  # Count number of bars lit up in each sample
                
                inputs = inputs.view(inputs.size(0), -1)  # Flatten the input tensor
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                predictions: torch.Tensor = self.model(inputs)
                self.DEBUG_LOG.info(f"Batch {batch_idx}: Predictions Shape: {predictions.shape}, Predictions: {predictions}")

                top_k_preds = torch.zeros_like(predictions)
                
                for i in range(predictions.size(0)):
                    # Get indices of top 'num_bars' activations
                    _, top_k_indices = torch.topk(predictions[i], int(num_bars[i].item()))  # Explicitly convert to int
                    
                    # Set these indices to 1 in the prediction tensor
                    top_k_preds[i, top_k_indices] = 1

                    correct += int(torch.sum(top_k_preds == labels).item())

                self.DEBUG_LOG.info(f"Batch {batch_idx}: Top-k Preds: {top_k_preds}")
                self.DEBUG_LOG.info(f"Batch {batch_idx}: Labels: {labels}")
                self.DEBUG_LOG.info(f"Batch {batch_idx}: Correct Predictions: {correct}")

            final_accuracy = round(correct / (total * labels.size(1)), 4)

        test_end = time.time()
        testing_time = test_end - test_start

        self.EXP_LOG.info(f"Completed testing with {correct} out of {total}.")
        self.EXP_LOG.info(f"Testing ({purpose}) of sample #{self.SAMPLES} took {time_to_str(testing_time)}.")

        if purpose == "Test Set One":
            self.TEST_LOG.info(f'Samples Seen: {self.SAMPLES} || Dataset: {dname.upper()} || Test Set One Accuracy: {final_accuracy}')
        elif purpose == "Test Set Two":
            self.TEST_LOG.info(f'Samples Seen: {self.SAMPLES} || Dataset: {dname.upper()} || Test Set Two Accuracy: {final_accuracy}')
        elif purpose == "Test Set Three":
            self.TEST_LOG.info(f'Samples Seen: {self.SAMPLES} || Dataset: {dname.upper()} || Test Set Three Accuracy: {final_accuracy}')
        elif purpose == "Training Set":
            self.TRAIN_LOG.info(f'Samples Seen: {self.SAMPLES} || Dataset: {dname.upper()} || Training Accuracy: {final_accuracy}')

        if visualize:
            self.model.visualize_weights(self.RESULT_PATH, self.SAMPLES, purpose.lower())

        return final_accuracy

    def _experiment(self) -> None:
        torch.device(self.device)

        self.EXP_LOG.info("Started training and testing loops.")
        
        for epoch in range(0, self.epochs):
            self._training(self.train_data_loader, epoch, "training_set", ExperimentPhases.BASE)
            
            self._generalization_test(self.test_set_one_loader, "Test Set One", "single_bars", ExperimentPhases.BASE)
            self._generalization_test(self.test_set_two_loader, "Test Set Two", "forbidden_combinations", ExperimentPhases.BASE)
            self._generalization_test(self.test_set_three_loader, "Test Set Three", "complex_patterns", ExperimentPhases.BASE)

            self._generalization_test(self.train_data_loader, "Training Set", "Training samples", ExperimentPhases.BASE)
        
        self.EXP_LOG.info("Completed training of model.")        
        self.model.visualize_weights(self.RESULT_PATH, self.SAMPLES, 'final')
        self.EXP_LOG.info("Visualize weights of model after training.")

    def _final_test(self) -> Tuple[float, ...]:
        test_acc_one = self._generalization_test(self.test_set_one_loader, "Test Set One", "single_bars", ExperimentPhases.BASE, visualize=False)
        test_acc_two = self._generalization_test(self.test_set_two_loader, "Test Set Two", "forbidden_combinations", ExperimentPhases.BASE, visualize=False)
        test_acc_three = self._generalization_test(self.test_set_three_loader, "Test Set Three", "complex_patterns", ExperimentPhases.BASE, visualize=False)
        
        self.EXP_LOG.info("Completed final testing methods.")
        return (test_acc_one, test_acc_two, test_acc_three)


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
    