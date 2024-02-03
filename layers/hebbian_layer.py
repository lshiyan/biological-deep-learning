import torch
import torch.nn as nn
import math

class HebbianLayer (nn.Module):
    def __init__(self, input_dimension, output_dimension, lamb=1, lr=0.001):
        super (HebbianLayer, self).__init__()
        self.input_dimension=input_dimension
        self.output_dimension=output_dimension
        self.lamb=lamb
        self.lr=lr
        
        self.FC=nn.Linear(self.input_dimension, self.output_dimension)
    
    def inhibition(self, x):
        normalization_factor=0
        for ele in x:
            normalization_factor+= ele**self.lamb
        inhibited_x=[(ele**self.lamb)/normalization_factor for ele in x]
        return inhibited_x
    
    def updateWeights(self):
        return 
    
    def forward(self, x):
        input=x
        x=self.FC(x)
        x=self.inhibition(x)
        self.updateWeights()    
        return x
    
if __name__=="__main__":
    print("test")
