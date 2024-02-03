from ..layers.hebbian_layer import HebbianLayer
import torch
import torch.nn as nn 

class HebbianNetwork():
    
    def __init__(self, input_dimension, hidden_layer_dimension, output_dimension):
        self.input_dimension=input_dimension
        self.output_dimension=output_dimension
        self.hidden_layer_dimension=hidden_layer_dimension
        self.hebbian_layer=HebbianLayer(self.input_dimension, self.hidden_layer_dimension)
        self.classifier_layer=nn.Linear(self.hidden_layer_dimension, self.output_dimension)
        
    def forward(self,x):
        x=self.hebbian_layer(x)
        x=self.classifier_layer(x)
        return x
    
if __name__=="__main__":
    network=HebbianNetwork(3,3)