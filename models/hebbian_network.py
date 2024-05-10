from layers.hebbian_layer import HebbianLayer
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
    """
    def __init__(self, input_dimension, hidden_layer_dimension, output_dimension, heb_lr=1, lamb=1, eps=10e-5):
        super(HebbianNetwork, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_layer_dimension = hidden_layer_dimension
        self.hebbian_layer = HebbianLayer(self.input_dimension, self.hidden_layer_dimension, False, lamb=lamb, heb_lr=heb_lr, eps=eps)
        self.classifier_layer = HebbianLayer(self.hidden_layer_dimension, self.output_dimension, True, lamb=lamb, heb_lr=heb_lr)
    
    """
    Method to set scheduler to either the classification layer or the hebbian layer
    @param
        scheduler (layers.Scheduler) = a scheduler
        classifier (bool) = true if setting scheduler for classifier layer
    """
    # TODO: modify this and make it more reusable -> might need to change other parts of the code (hebbian_layer.py)
    def setScheduler(self, scheduler, classifier):
        if classifier:
            self.classifier_layer.setScheduler(scheduler)
        else:
            self.hebbian_layer.setScheduler(scheduler)

    """
    Method that defines how an input data flows throw the network
    @param
        x (torch.Tensor) = input data as a tensor
        clamped_out (???) = parameter to clamp the output #WTV this means
        train (int) = true if in training
    """   
    def forward(self, x, clamped_output=None, train=1):
        x=self.hebbian_layer(x, clamped_output=None, train=train)
        x=self.classifier_layer(x, clamped_output=clamped_output, train=train)
        return x
    
    
    # Method to visualize the weights/features learned by each neuron during training
    def visualizeWeights(self):
        self.hebbian_layer.visualizeWeights(classifier=0)
        self.classifier_layer.visualizeWeights(classifier=1)

# TODO: remove this from code
if __name__=="__main__":
    network = HebbianNetwork(784, 256, 10)
    print(network)