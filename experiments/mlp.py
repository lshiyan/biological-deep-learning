import torch
import torch.nn as nn 
import torch.optim as optim
import matplotlib.pyplot as plt

import argparse

from torch.utils.data import DataLoader
from data.data_loader import ImageDataSet
from models.hebbian_network import HebbianNetwork
from layers.scheduler import Scheduler


"""
Class to implement and run an experiment depending on the data and model chosen
"""
class MLPExperiment():
    """
    Constructor method to create an experiment
    @param
        args (argparse.ArgumentParser) = argument parser that has all the arguments passed to run.py
        num_epochs (int) = number of iterations
    @attr.
        model (nn.Module) = the model that will be used in the experiment
        num_epochs (int) = number of iterations
        args (argparse.ArgumentParser) = argument parser that has all the arguments
        data_name (str) = name of the dataset
        train_filename (str) = name of the train dataset
        test_filename (str) = name of the test dataset
    @return
        ___ (experiments.MLPExperiments) = new instance of MLPExperiment 
    """
    def __init__(self, args=None, num_epochs=3):
        # For testing when not ran in command line with run.py
        if args == None:
            parser = argparse.ArgumentParser(description='Biological deep learning')

            # Basic configurations.
            parser.add_argument('--is_training', type=bool, default=True, help='status')
            parser.add_argument('--data_name', type=str, default="MNIST")
            
            # Data Factory
            parser.add_argument('--train_data', type=str, default="data/mnist/train-images.idx3-ubyte")
            parser.add_argument('--train_label', type=str, default="data/mnist/train-labels.idx1-ubyte")
            parser.add_argument('--test_data', type=str, default="data/mnist/t10k-images.idx3-ubyte")
            parser.add_argument('--test_label', type=str, default="data/mnist/t10k-labels.idx1-ubyte")

            # CSV files generated
            parser.add_argument('--train_filename', type=str, default="data/mnist/mnist_train.csv")
            parser.add_argument('--test_filename', type=str, default="data/mnist/mnist_test.csv")

            # Dimension of each layer
            parser.add_argument('--input_dim', type=int, default=784)
            parser.add_argument('--heb_dim', type=int, default=64)
            parser.add_argument('--output_dim', type=int, default=10)

            # Hebbian layer hyperparameters
            parser.add_argument('--heb_lr', type=float, default=0.001)
            parser.add_argument('--heb_lamb', type=float, default=15)
            parser.add_argument('--heb_gam', type=float, default=0.99)

            # Classification layer hyperparameters
            parser.add_argument('--cla_lr', type=float, default=0.001)
            parser.add_argument('--cla_lamb', type=float, default=1)
            parser.add_argument('--cla_gam', type=float, default=0.99)

            # Shared hyperparameters
            parser.add_argument('--eps', type=float, default=10e-5)
            # Parse arguments
            args, _ = parser.parse_known_args()

        self.model = HebbianNetwork(args)
        self.num_epochs = num_epochs
        self.args = args
        self.data_name = args.data_name
        self.train_filename = args.train_filename
        self.test_filename = args.test_filename


    """
    Returns defined optimizer for gradiant descent
    @param
    @return
        optimizer (optim.Adam) = ADAM optimizer
    """    
    def optimizer(self):
        optimizer = optim.Adam(self.model.get_layer("Hebbian layer").parameters(), self.args.cla_lr)
        return optimizer
    

    """
    Returns cross entropy loss function
    @param
    @return
        loss_function (nn.CrossEntropy) = cross entropy loss function
    """
    # NOTE: what is the use of this function within our code???
    def loss_function(self):
        loss_function = nn.CrossEntropyLoss()
        return loss_function
    

    """
    Sets the scheduler for the feature detector layer of the network
    @param
    @return
        ___ (void) = no returns
    """
    def set_scheduler(self):
        self.model.set_scheduler()
    

    """
    Trains the experiment
    @param
    @return
        ___ (void) = no returns
    """
    def train(self):  
        data_set = self.model.get_layer("Input Layer").setup_train_data()
        data_loader = DataLoader(data_set, batch_size=1, shuffle=True)
        
        self.model.train()
        
        optimizer = self.optimizer()
        if self.args.heb_gam !=0 : self.set_scheduler()
        
        for _ in range(self.num_epochs):
            for _, data in enumerate(data_loader):
                inputs, labels = data
                self.model(inputs, clamped_output=self.one_hot_encode(labels, 10))
                optimizer.step()


    """
    Given a tensor of labels, returns a one hot encoded tensor for each label.
    @params
        labels (???) = set of labels
        num_classes (int) = number of classes
    @return
        one_hot_encoded.squeeze() (???) = ???
    """
    def one_hot_encode(self, labels, num_classes):
        one_hot_encoded = torch.zeros(len(labels), num_classes)
        one_hot_encoded.scatter_(1, labels.unsqueeze(1), 1)
    
        return one_hot_encoded.squeeze()


    """
    Visualizes the weights/features learned during training.
    @param
    @return
        ___ (void) = no returns
    """
    def visualize_weights(self):
        self.model.visualize_weights()
    

    """
    Test the model with the testing data
    @param
    @return
        correct / total (float) = accuracy of model on testing data
    """
    def test(self):
        data_set = self.model.get_layer("Input Layer").setup_test_data()
        data_loader = DataLoader(data_set, batch_size=1, shuffle=True)
        correct = 0
        total = 0
        for _, data in enumerate(data_loader):
            inputs, labels = data
            outputs = torch.argmax(self.model(inputs, None))
            if outputs.item() == labels.item():
                correct += 1
            total += 1
        return correct / total
    

    """
    Plots visually the exponential averages
    @param
    @return
        ___ (void) = no returns
    """
    def print_exponential_averages(self):
        A = torch.log(self.model.get_exponential_averages()).tolist()
        plt.scatter(range(len(A)), A)
        for i, (x, y) in enumerate(zip(range(len(A)), A)):
            plt.text(x, y, f'{i}', ha='center', va='bottom')
        plt.xlabel("Feature Selector")
        plt.ylabel("Log (Exponential Average)")
        plt.title("Logged Exponential Averages of Each Feature Selector")


    """
    Returns the number of weights that are active according to a certain threshold
    @param
        beta (float) = threshold to determine if a certain neuro is active or not
    @return
        self.model.active_weights (int) = number of active weights
    """    
    def active_weights(self, beta):
        return self.model.active_weights(beta)     