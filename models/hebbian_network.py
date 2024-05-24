from layers.hebbian_layer import HebbianLayer
from layers.classifier_layer import ClassifierLayer
import torch.nn as nn 

class HebbianNetwork(nn.Module):
    """
    Hyperparameters that remain constant within the network
    @const
        INPUT_DIMENSION (int) = number of inputs to network
        HEBBIAN_LAYER_DIMENSION (int) = number of neurons in hebbian layer
        OUTPUT_DIMENSION (int) = number of output neurons from network

        HEBBIAN_LR (float) = learning rate of hebbian layer
        HEBBIAN_LAMBDA (float) = hyperparameter for lateral neuron inhibition
        HEBBIAN_GAMMA (float) = factor to decay learning rate of hebbian layer

        CLASSIFICATION_LR (float) = learning rate of classification layer
        CLASSIFICATION_LAMBDA (float) = hyperparameter for lateral neuron inhibition
        CLASSIFICATION_GAMMA (float) = factor to decay learning rate of classification layer
        
        EPS (float) = small value to avoid 0 division
    """
    # Number of 
    INPUT_DIMENSION = 784
    HEBBIAN_LAYER_DIMENSION = 64
    OUTPUT_DIMENSION = 10

    # Hebbian layer hyperparameters
    HEBBIAN_LR = 0.001
    HEBBIAN_LAMBDA = 15
    HEBBIAN_GAMMA = 0.99

    # Classification layer hyperparameters
    CLASSIFICATION_LR = 0.001
    CLASSIFICATION_LAMBDA = 1
    CLASSIFICATION_GAMMA = 0.99
    
    # Shared hyperparameter across layers
    EPS = 10e-5

    """
    Constructor method
    @attr.
        input_dimension (int) = number of inputs
        output_dimension (int) = number of output neurons
        hidden_layer_dimension (int) = number of neurons in hidden layer
        hebbian_layer (layers.HebbianLayer) = hidden NN layer based off hebbian learning
        classifier_layer (layer.HebbianLayer) = output layer used for classification
    """
    def __init__(self):
        super(HebbianNetwork, self).__init__()
        self.hebbian_layer = HebbianLayer(HebbianNetwork.INPUT_DIMENSION, HebbianNetwork.HEBBIAN_LAYER_DIMENSION, HebbianNetwork.HEBBIAN_LAMBDA, HebbianNetwork.HEBBIAN_LR, HebbianNetwork.HEBBIAN_GAMMA, HebbianNetwork.EPS)
        self.classifier_layer = ClassifierLayer(HebbianNetwork.HEBBIAN_LAYER_DIMENSION, HebbianNetwork.OUTPUT_DIMENSION, HebbianNetwork.CLASSIFICATION_LAMBDA, HebbianNetwork.CLASSIFICATION_LR, HebbianNetwork.CLASSIFICATION_GAMMA, HebbianNetwork.EPS)
    
    """
    Method to set scheduler to either the hebbian layer
    @param
        scheduler (layers.Scheduler) = a scheduler
    """
    def set_scheduler_hebbian_layer(self, scheduler):
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
        clamped_output (???) = parameter to clamp the output   # WTV this means
    """   
    def forward(self, x, clamped_output=None):
        x = self.hebbian_layer(x, clamped_output)
        x = self.classifier_layer(x)
        return x
    
    """
    Method to visualize the weights/features learned by each neuron during training
    """
    def visualize_weights(self):
        self.hebbian_layer.visualize_weights(8, 8)
        self.classifier_layer.visualize_weights(2, 5)