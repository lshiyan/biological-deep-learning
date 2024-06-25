from abc import ABC, abstractmethod
import torch


"""
Interface for all types of experiments
"""
class Experiment(ABC):
    """
    Constructor method to create an experiment
    @param
        args (argparse.ArgumentParser) = argument parser that has all the arguments passed to run.py
    @attr.
    @return
        * Can't return * 
    """
    def __init__(self, args=None):
        pass


    """
    Returns defined optimizer for gradiant descent
    @param
    @return
        optimizer (optim.Adam) = ADAM optimizer
    """    
    @abstractmethod
    def optimizer(self):
        pass
    

    """
    Returns cross entropy loss function
    @param
    @return
        loss_function (nn.CrossEntropy) = cross entropy loss function
    """
    @abstractmethod
    def loss_function(self):
        pass
    

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
    @abstractmethod
    def train(self):  
        pass


    """
    Test the model with the testing data
    @param
    @return
        correct / total (float) = accuracy of model on testing data
    """
    @abstractmethod
    def test(self):
        pass


    '''
    Abstract method for calculating reconstruction error
    @param
        data_loader is a DataLoader object
    '''
    def compute_reconstruction_error(self, data_loader: DataLoader) -> float:
        pass
    

    """
    Plots visually the exponential averages
    @param
    @return
        ___ (void) = no returns
    """
    def print_exponential_averages(self):
        pass


    """
    Returns the number of weights that are active according to a certain threshold
    @param
        beta (float) = threshold to determine if a certain neuro is active or not
    @return
        self.model.active_weights (int) = number of active weights
    """    
    def active_weights(self, beta):
        return self.model.active_weights(beta)   


    """
    Visualizes the weights/features learned during training.
    @param
    @return
        ___ (void) = no returns
    """
    def visualize_weights(self):
        self.model.visualize_weights()  


    """
    Given a tensor of labels, returns a one hot encoded tensor for each label.
    @params
        labels (???) = set of labels
        num_classes (int) = number of classes
    @return
        one_hot_encoded.squeeze() (???) = ???
    """
    # NOTE: Is there a reason we ceated our own one_hot funciton instead of using PyTorch's one?
    @classmethod
    def one_hot_encode(cls, labels, num_classes):
        one_hot_encoded = torch.zeros(len(labels), num_classes)
        one_hot_encoded.scatter_(1, labels.unsqueeze(1), 1)
    
        return one_hot_encoded.squeeze()