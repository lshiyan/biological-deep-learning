import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from numpy import outer

#Hebbian learning layer that implements lateral inhibition in output. Not trained through supervision.
class HebbianLayer (nn.Module):
    def __init__(self, input_dimension, output_dimension, lamb=1.0, heb_lr=0.1, K=10):
        super (HebbianLayer, self).__init__()
        self.input_dimension=input_dimension
        self.output_dimension=output_dimension
        self.lamb=lamb
        self.alpha = heb_lr
        self.K=K
        self.fc=nn.Linear(self.input_dimension, self.output_dimension)
  
        for param in self.fc.parameters():
            param=torch.nn.init.uniform_(param, a=0.0, b=0.5)
            param.requires_grad_(False)
    
    #Calculates lateral inhibition h_mu -> (h_mu)^(lambda)/ sum on i (h_mu_i)^(lambda)
    def inhibition(self, x):        
        #print(torch.max(x ** self.lamb))
        x=torch.pow(x,self.lamb)
        if len(x.shape)==1:
            normalization_factor = torch.max(x)
        else:       
            normalization_factor = torch.max(x, dim=1, keepdim=True)
        x/=(normalization_factor)
        return x
    
    #Employs hebbian learning rule, Wij->alpha*y_i*x_j. 
    #Calculates outer product of input and output and adds it to matrix.
    def updateWeightsHebbian(self, input, output):
        x=torch.tensor(input, requires_grad=False, dtype=torch.float)
        y=torch.tensor(output, requires_grad=False, dtype=torch.float)


        outer_prod=torch.tensor(outer(y, x))
        torch.nn.functional.normalize(self.fc.weight,p=2, dim=1)
        self.fc.weight=nn.Parameter(torch.add(self.fc.weight, self.alpha * (outer_prod -1*self.fc.weight) ), requires_grad=False)
        torch.nn.functional.normalize(self.fc.weight,p=2, dim=1)

    #Feed forward.
    def forward(self, x, clamped_output=None,train = True):
        input=x
        if not train:
            return self.fc(x)
        x = self.fc(x)
        x = self.inhibition(x)
        #0x = x/torch.sum(x)
        if clamped_output is not None: #If we're clamping the output, i.e. a one hot of the label, update accordingly.
            self.updateWeightsHebbian(input, clamped_output)
        else: #If not, do hebbian update with usual output.
            #x=self.fc(x)
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
    test_layer=HebbianLayer(3,3,1.5)
    test_layer(torch.tensor([1,2,3], dtype=torch.float))
    
