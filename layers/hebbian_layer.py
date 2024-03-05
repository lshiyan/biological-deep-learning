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
            param=torch.nn.init.uniform_(param, a=-1, b=1)
            param.requires_grad_(False)
    
    #Calculates lateral inhibition h_mu -> (h_mu)^(lambda)/ sum on i (h_mu_i)^(lambda)
    def inhibition(self, x):        
        #print(torch.max(x ** self.lamb))
        x = torch.relu_(x)
        x = torch.pow(x,self.lamb)

        if len(x.shape)==1:
            normalization_factor = torch.max(x, dim=0, keepdim=True)
        else:       
            normalization_factor = torch.max(x, dim=1, keepdim=True)
        x = x/(normalization_factor.values + 1e-16)
        return x
    
    #Employs hebbian learning rule, Wij->alpha*y_i*x_j. 
    #Calculates outer product of input and output and adds it to matrix.
    def updateWeightsHebbian(self, input, output, clamped_output=None):
        x=input
        y=output
        z = clamped_output if clamped_output is not None else y
        outer_prod = torch.tensor(outer(z, x), dtype=torch.float)
        current_weights = self.fc.weight
        decay_strength = torch.tensor(y).unsqueeze(-1)
        max_decay = 0.1
        if len(decay_strength.shape) == 3:
            decay_strength = decay_strength.mean(0)
        max_strength = torch.max(decay_strength)
        decay_strength = 1 - max_decay * decay_strength/(max_strength + 1e-16)
        decayed_weights = current_weights * decay_strength
        hebbian_update = torch.add(decayed_weights, self.alpha * outer_prod)
        # normalized_weights = torch.nn.functional.normalize(hebbian_update,p=2, dim=1)
        self.fc.weight=nn.Parameter(hebbian_update, requires_grad=False)


    #Feed forward.
    def forward(self, x, clamped_output=None,train = True):
        input=x.clone()
        y = self.fc(x)
        y = self.inhibition(y)
        z = clamped_output if clamped_output is not None else y
        if not train:
            return y
        #0x = x/torch.sum(x)
        self.updateWeightsHebbian(input, y, clamped_output=clamped_output)
        return y
    
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
    
