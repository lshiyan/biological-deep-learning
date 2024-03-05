import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from numpy import outer
import warnings

warnings.filterwarnings("ignore")

#Hebbian learning layer that implements lateral inhibition in output. Not trained through supervision.
class HebbianLayer (nn.Module):
    def __init__(self, input_dimension, output_dimension, lamb=2, heb_lr=0.001, K=10):
        super (HebbianLayer, self).__init__()
        self.input_dimension=input_dimension
        self.output_dimension=output_dimension
        self.lamb=lamb
        self.alpha = heb_lr
        self.K=K
        self.fc=nn.Linear(self.input_dimension, self.output_dimension, bias=False)
  
        for param in self.fc.parameters():
            param=torch.nn.init.uniform_(param, a=0.0, b=0.5)
            param.requires_grad_(False)
            
        self.itensors=self.createITensors()
        
    #Creates identity tensors for Sanger's rule computation.
    def createITensors(self):
        itensors=torch.zeros(self.output_dimension, self.output_dimension, self.output_dimension)
        for i in range(0, self.output_dimension):
            identity = torch.eye(i+1)
            padded_identity = torch.nn.functional.pad(identity, (0, self.output_dimension - i-1, 0, 
                                                                 self.output_dimension - i-1))
            itensors[i]=padded_identity
        return itensors
    
    #Calculates lateral inhibition h_mu -> (h_mu)^(lambda)/ max on i (h_mu_i)^(lambda)
    def inhibition(self, x):
        max_ele=torch.max(x, dim=1).values.item()
        x=torch.pow(x,self.lamb) #Make sure that lamb is an integer power.
        x/=abs(max_ele)**self.lamb
        return x
    
    #Employs Sanger's Rule, deltaW_(ij)=alpha*x_j*y_i-alpha*y_i*sum(k=1 to i) (w_(kj)*y_k)
    #Calculates outer product of input and output and adds it to matrix.
    def updateWeightsHebbian(self, input, output, train):
        if train:
            x=torch.tensor(input.clone().detach(), requires_grad=False, dtype=torch.float).squeeze()
            y=torch.tensor(output.clone().detach(), requires_grad=False, dtype=torch.float).squeeze()
            outer_prod=torch.tensor(outer(y, x))
            initial_weight=self.fc.weight.clone().detach().transpose(0,1)
            self.fc.weight=nn.Parameter(torch.add(self.fc.weight, self.alpha*outer_prod), requires_grad=False)
            A=torch.einsum('jk, lkm, m -> lj', initial_weight, self.itensors, y)
            A=A*(y.unsqueeze(1))
            self.fc.weight=nn.Parameter(torch.sub(self.fc.weight, self.alpha*A), requires_grad=False)
            
    #Feed forward
    def forward(self, x, clamped_output=None, train=1):
        input=x.clone()
        if clamped_output is not None: #If we're clamping the output, i.e. a one hot of the label, update accordingly.
            self.updateWeightsHebbian(input, clamped_output, train)
            return clamped_output  
        else: #If not, do hebbian update with usual output.
            x=self.fc(x)
            x=self.inhibition(x) 
            self.updateWeightsHebbian(input, x, train)  
            return x
    
    #Creates heatmap of randomly chosen feature selectors.
    def visualizeWeights(self, num_choices):
        weight=self.fc.weight
        random_indices = torch.randperm(self.fc.weight.size(0))[:num_choices]
        for ele in random_indices:#Scalar tensor
            idx=ele.item()
            random_feature_selector=weight[idx]
            heatmap=random_feature_selector.view(int(math.sqrt(self.fc.weight.size(1))), 
                                                 int(math.sqrt(self.fc.weight.size(1))))
            plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            plt.title("HeatMap of feature selector {} with lambda {}".format(idx, self.lamb))
            plt.colorbar()
            plt.show()
        return
    
if __name__=="__main__":
    length_3_tensor = torch.tensor([2, 3, 4])
    tensor_3x3 = torch.tensor([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]])
    expanded_length_3_tensor = length_3_tensor.unsqueeze(1)
    print(tensor_3x3*expanded_length_3_tensor)
    
