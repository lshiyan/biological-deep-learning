import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from numpy import outer


#Hebbian learning layer that implements lateral inhibition in output. Not trained through supervision.
class HebbianLayer (nn.Module):
    def __init__(self, input_dimension, output_dimension, lamb=1, heb_lr=1):
        super (HebbianLayer, self).__init__()
        self.input_dimension=input_dimension
        self.output_dimension=output_dimension
        self.lamb=lamb
        self.lr=heb_lr
        
        self.fc=nn.Linear(self.input_dimension, self.output_dimension)
  
        for param in self.fc.parameters():
            torch.nn.init.uniform_(param, a=0.0, b=1.0)
            param.requires_grad_(False)
    
    #Calculates lateral inhibition h_mu -> (h_mu)^(lambda)/ sum on i (h_mu_i)^(lambda)
    def inhibition(self, x):
        normalization_factor=0
        normalization_factor+= torch.sum(x ** self.lamb)
        print("Factor is:", normalization_factor)
        x/=normalization_factor
        print(x)
        return x
    
    #Employs hebbian learning rule, Wij->alpha*y_i*x_j. Calculates outer product of input and output and adds it to matrix.
    def updateWeightsHebbian(self, input, output):
        weight=self.fc.weight
        outer_prod=torch.tensor(outer(output, input))
        weight=torch.add(weight, outer_prod)
                
    #Feed forward.
    def forward(self, x):
        input=x
        x=self.fc(x)
        print("After FC Layer")
        print(x)
        x=self.inhibition(x)
        self.updateWeightsHebbian(input, x)    

        return x
    
    #Creates heatmap of randomly chosen feature selectors.
    def visualizeWeights(self, num_choices):
        weight=self.fc.weight
        random_indices = torch.randperm(self.fc.weight.size(0))[:num_choices]
        for ele in random_indices:#Scalar tensor
            idx=ele.item()
            random_feature_selector=weight[idx]
            heatmap=random_feature_selector.view(28,28)
            plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            plt.title("HeatMap of feature selector {} with lambda {}".format(idx, self.lamb))
            plt.colorbar()
            plt.show()
        return
    
if __name__=="__main__":
    test_layer=HebbianLayer(3,3,1)
    test_layer(torch.tensor([1,2,3], dtype=torch.float))
