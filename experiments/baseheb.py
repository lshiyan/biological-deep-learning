import torch
import torch.nn as nn 
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.hebbian_network import HebbianNetwork
from experiments.experiment import Experiment


"""
Class to implement and run an experiment depending on the data and model chosen
"""
class BaseHebbianExperiment(Experiment):
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
        optimizer = optim.Adam(self.model.get_layer("Hebbian Layer").parameters(), self.args.cla_lr)
        return optimizer
    

    """
    Returns cross entropy loss function
    @param
    @return
        loss_function (nn.CrossEntropy) = cross entropy loss function
    """
    def loss_function(self):
        loss_function = nn.CrossEntropyLoss()
        return loss_function
    

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