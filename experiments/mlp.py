import torch
import torch.nn as nn 
import torch.optim as optim
import matplotlib.pyplot as plt

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
        args (argparse.ArgumentParser) = argument parser that has all the argumentd passed to run.py
        input_dimension (int) = number of inputs into model
        hidden_layer_dimension (int) = number of neurons in hidden layer
        output_dimension (int) = number of neurons in output layer
        lamb (float) = hyperparameter for latteral inhibition
        heb_lr (float) = learning rate for hebbian layer
        grad_lr (float) = learning rate for gradient descent for classification
        num_epochs (int) = number of iterations
        gamma (float) = decay factor -> factor to decay learning rate
        eps (float) = number to avoid division by 0
    @attr.
        model (nn.Module) = the model that will be used in the experiment
        args (argparse.ArgumentParser) = arguments passed to run.py
        num_epochs (int) = number of iterations
        grad_lr (float) = learning rate for gradient descend for classification
        heb_lr (float) = learning rate for hebbian layer
        gamma (float) = decay factor -> factor to decay learning rate
    """
    def __init__(self, args, input_dimension, hidden_layer_dimension, output_dimension, lamb=1, heb_lr=1, grad_lr=0.001, num_epochs=3, gamma=0, eps=10e-5):
        self.model = HebbianNetwork(input_dimension, hidden_layer_dimension, output_dimension, heb_lr=heb_lr, lamb=lamb, eps=eps) # TODO: For some reason my hebbian network is not processing batches together.
        self.args = args
        self.num_epochs = num_epochs
        self.grad_lr = grad_lr 
        self.heb_lr = heb_lr
        self.gamma = gamma

    """
    Returns ADAM optimizer for gradiant descent
    """    
    def optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), self.grad_lr)
        return optimizer
    
    """
    Sets the scheduler for the feature detector layer of the network
    """
    # NOTE: create scheduler within funciton? or pass scheduler as a paramter?
    def set_hebbian_scheduler(self):
        scheduler = Scheduler(self.heb_lr, 1000, self.gamma)
        self.model.set_scheduler_hebbian_layer(scheduler)
    
    """
    Returns cross entropy loss function
    """
    # NOTE: what is the use of this function within our code???
    def loss_function(self):
        loss_function = nn.CrossEntropyLoss()
        return loss_function
    
    """
    Trains the experiment
    """
    def train(self):  
        data_set=fashion_MNIST_set(self.args)
        # data_set=MNIST_set(self.args)
        data_loader=DataLoader(data_set, batch_size=1, shuffle=True)
        
        self.model.train()
        
        optimizer=self.optimizer()
        if self.gamma !=0 : self.set_hebbian_scheduler()
        
        for _ in range(self.num_epochs):
            for i, data in enumerate(data_loader):
                inputs, labels=data
                self.model(inputs, clamped_output=self.oneHotEncode(labels, 10))
                optimizer.step()
        
            
    # Given a tensor of labels, returns a one hot encoded tensor for each label.
    def oneHotEncode(self, labels, num_classes):
        one_hot_encoded = torch.zeros(len(labels), num_classes)
        one_hot_encoded.scatter_(1, labels.unsqueeze(1), 1)
    
        return one_hot_encoded.squeeze()
        
    # Visualizes the weights associated with the first feature detector layer.
    def visualizeWeights(self):
        self.model.visualize_weights()
    
    def test(self):
        data_set=fashion_MNIST_set(self.args, 0)
        data_loader=DataLoader(data_set, batch_size=1, shuffle=True)
        cor=0
        tot=0
        for _, data in enumerate(data_loader):
            inputs, labels=data
            outputs = torch.argmax(self.model(inputs, None, train=0))
            if outputs.item()==labels.item():
                cor+=1
            tot+=1
        return cor/tot
    
    def printExponentialAverages(self):
        A=torch.log(experiment.model.hebbian_layer.exponential_average).tolist()
        plt.scatter(range(len(A)), A)
        for i, (x, y) in enumerate(zip(range(len(A)), A)):
            plt.text(x, y, f'{i}', ha='center', va='bottom')
        plt.xlabel("Feature Selector")
        plt.ylabel("Log (Exponential Average)")
        plt.title("Logged Exponential Averages of Each Feature Selector")
        
    def activeClassifierWeights(self, beta):
        return self.model.classifier_layer.active_classifier_weights(beta)     