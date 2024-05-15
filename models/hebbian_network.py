from layers.hebbian_layer import HebbianLayer
from layers.classifier_layer import ClassifierLayer
import torch.nn as nn 

class HebbianNetwork(nn.Module):
    """
    Constructor method
    @param
        input_dimension (int) = number of inputs
        hidden_layer_dimension (int) = number of neurons in hidden layer
        output_dimension (int) = number of output neurons
        heb_lr (float) = learning rate of NN
        lamb (float) = hyperparameter for lateral neuron inhibition
        eps (float) = small value to avoid 0 division
    @attr.
        input_dimension (int) = number of inputs
        output_dimension (int) = number of output neurons
        hidden_layer_dimension (int) = number of neurons in hidden layer
        hebbian_layer (layers.HebbianLayer) = hidden NN layer based off hebbian learning
        classifier_layer (layer.HebbianLayer) = output layer used for classification
    """
    def __init__(self, input_dimension, hidden_layer_dimension, output_dimension, heb_lr=1, lamb=1, eps=10e-5):
        super(HebbianNetwork, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_layer_dimension = hidden_layer_dimension
        self.hebbian_layer = HebbianLayer(self.input_dimension, self.hidden_layer_dimension, False, lamb=lamb, heb_lr=heb_lr, eps=eps)
        self.classifier_layer = ClassifierLayer(self.hidden_layer_dimension, self.output_dimension, True, lamb=lamb, heb_lr=heb_lr)
    
    """
    Method to set scheduler to either the hebbian layer
    @param
        scheduler (layers.Scheduler) = a scheduler
    """
    def set_scheduler_hebbian_layer(self, scheduler, classifier):
        self.hebbian_layer.set_scheduler(scheduler)
    
    """
    Method to set scheduler to either the classification layer
    @param
        scheduler (layers.Scheduler) = a scheduler
    """
    def set_scheduler_classifier(self, scheduler):
        self.classifier_layer.set_scheduler(scheduler)

    """
    Method that defines how an input data flows throw the network
    @param
        x (torch.Tensor) = input data as a tensor
        clamped_out (???) = parameter to clamp the output #WTV this means
        train (bool) = true if in training
    """   
    def forward(self, x, clamped_output=None, train=True):
        x=self.hebbian_layer(x, clamped_output=None, train=train)
        x=self.classifier_layer(x, clamped_output=clamped_output, train=train)
        return x
    
    
    """
    Method to visualize the weights/features learned by each neuron during training
    """
    def visualizeWeights(self):
        self.hebbian_layer.visualize_weights(8, 8)
        self.classifier_layer.visualize_weights(2, 5)