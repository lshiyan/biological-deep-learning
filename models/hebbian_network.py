from layers.hebbian_layer import HebbianLayer
import torch
import torch.nn as nn 

class HebbianNetwork(nn.Module):
    
    def __init__(self, input_dimension, hidden_layer_dimension, output_dimension, heb_lr=1, lamb=1):
        super(HebbianNetwork, self).__init__()
        self.input_dimension=input_dimension
        self.output_dimension=output_dimension
        self.hidden_layer_dimension=hidden_layer_dimension
        self.hebbian_layer=HebbianLayer(self.input_dimension, self.hidden_layer_dimension, lamb=lamb, heb_lr=heb_lr)
        self.classifier_layer=HebbianLayer(self.hidden_layer_dimension, self.output_dimension, lamb=lamb, heb_lr=heb_lr)
        
    def forward(self,x, clamped_output):
        x=self.hebbian_layer(x)
        x=self.classifier_layer(x, clamped_output)
        return x
    
    def visualizeWeights(self, num_choices):
        self.hebbian_layer.visualizeWeights(num_choices=num_choices)
    
if __name__=="__main__":
    network=HebbianNetwork(784, 256, 10)
    print(network)