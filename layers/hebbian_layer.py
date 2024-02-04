import torch
import torch.nn as nn
import math
from numpy import outer

#Hebbian learning layer that implements lateral inhibition in output. Not trained through supervision.
class HebbianLayer (nn.Module):
    def __init__(self, input_dimension, output_dimension, lamb=1, lr=0.001):
        super (HebbianLayer, self).__init__()
        self.input_dimension=input_dimension
        self.output_dimension=output_dimension
        self.lamb=lamb
        self.lr=lr
        
        self.fc=nn.Linear(self.input_dimension, self.output_dimension)
        for param in self.fc.parameters():
            param.requires_grad_(False)
    
    #Calculates lateral inhibition h_mu -> (h_mu)^(lambda)/ sum on i (h_mu_i)^(lambda)
    def inhibition(self, x):
        normalization_factor=0
        for ele in x:
            normalization_factor+= ele**self.lamb
        inhibited_x=[(ele**self.lamb)/normalization_factor for ele in x]
        return inhibited_x
    
    #Employs hebbian learning rule, Wij->alpha*y_i*x_j. Calculates outer product of input and output and adds it to matrix.
    def updateWeightsHebbian(self, input, output):
        weight=self.fc.weight
        outer_prod=torch.tensor(outer(output, input))
        weight=torch.add(weight, outer_prod)
        
                
    #Feed forward.
    def forward(self, x):
        print(x.size())
        input=x
        x=self.fc(x)
        x=self.inhibition(x)
        self.updateWeightsHebbian(input, x)    
        return x
    
"""if __name__=="__main__":
    hebbian_layer=HebbianLayer(3,3)
    x=torch.tensor([1.0,2.0,3.0])
    res=hebbian_layer(x)"""
