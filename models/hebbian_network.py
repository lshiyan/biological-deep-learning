from layers.hebbian_layer import HebbianLayer
import torch.nn as nn 

class HebbianNetwork(nn.Module):
    
    def __init__(self, input_dimension, hidden_layer_dimension, output_dimension, heb_lr=1, lamb=1, eps=10e-5):
        super(HebbianNetwork, self).__init__()
        self.input_dimension=input_dimension
        self.output_dimension=output_dimension
        self.hidden_layer_dimension=hidden_layer_dimension
        self.hebbian_layer=HebbianLayer(self.input_dimension, self.hidden_layer_dimension, False, lamb=lamb, heb_lr=heb_lr, eps=eps)
        self.classifier_layer=HebbianLayer(self.hidden_layer_dimension, self.output_dimension, True, lamb=lamb, 
                                           heb_lr=heb_lr)
    
    def setScheduler(self, scheduler, classifier):
        if classifier:
            self.classifier_layer.setScheduler(scheduler)
        else:
            self.hebbian_layer.setScheduler(scheduler)
            
    def forward(self, x, clamped_output=None, train=1):
        x=self.hebbian_layer(x, clamped_output=None, train=train)
        x=self.classifier_layer(x, clamped_output=clamped_output, train=train)
        return x
    
    def visualizeWeights(self):
        self.hebbian_layer.visualizeWeights(classifier=0)
        self.classifier_layer.visualizeWeights(classifier=1)
    
if __name__=="__main__":
    network=HebbianNetwork(784, 256, 10)
    print(network)