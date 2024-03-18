import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from numpy import outer
import warnings

warnings.filterwarnings("ignore")

#Hebbian learning layer that implements lateral inhibition in output. Not trained through supervision.
class HebbianLayer (nn.Module):
    def __init__(self, input_dimension, output_dimension, classifier, lamb=2, heb_lr=0.001, gamma=0.99):
        super (HebbianLayer, self).__init__()
        self.input_dimension=input_dimension
        self.output_dimension=output_dimension
        self.lamb=lamb
        self.alpha = heb_lr
        self.fc=nn.Linear(self.input_dimension, self.output_dimension, bias=False)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.softplus=nn.Softplus()
        self.tanh=nn.Tanh()
        self.isClassifier=classifier
        self.scheduler=None
        
        self.exponential_average=torch.zeros(self.output_dimension)
        self.gamma=gamma
        
        self.itensors=self.createITensors()
        
        for param in self.fc.parameters():
            param=torch.nn.init.uniform_(param, a=0.0, b=1.0)
            param.requires_grad_(False)
            
    def setScheduler(self, scheduler):
        self.scheduler=scheduler
        
    #Creates identity tensors for Sanger's rule computation.
    def createITensors(self):
        itensors=torch.zeros(self.output_dimension, self.output_dimension, self.output_dimension, dtype=torch.float)
        for i in range(0, self.output_dimension):
            identity = torch.eye(i+1)
            padded_identity = torch.nn.functional.pad(identity, (0, self.output_dimension - i-1, 0, 
                                                                 self.output_dimension - i-1))
            itensors[i]=padded_identity
        return itensors
    
    #Calculates lateral inhibition h_mu -> (h_mu)^(lambda)/ max on i (h_mu_i)^(lambda)
    def inhibition(self, x):
        x=self.relu(x) 
        random_tensor=torch.rand_like(x)
        mask=random_tensor<0.001
        max_ele=torch.max(x).item()
        x=torch.pow(x, self.lamb)
        x/=abs(max_ele)**self.lamb
        #x=torch.where(x==0, mask, x)
        return x
    
    #Employs Sanger's Rule, deltaW_(ij)=alpha*x_j*y_i-alpha*y_i*sum(k=1 to i) (w_(kj)*y_k)
    #Calculates outer product of input and output and adds it to matrix.
    def updateWeightsHebbian(self, input, output, clamped_output=None, train=1):
        if train:
            x=torch.tensor(input.clone().detach(), requires_grad=False, dtype=torch.float).squeeze()
            y=torch.tensor(output.clone().detach(), requires_grad=False, dtype=torch.float).squeeze()
            outer_prod=torch.tensor(outer(y, x))
            initial_weight=torch.transpose(self.fc.weight.clone().detach(), 0,1)
            A=torch.einsum('jk, lkm, m -> lj', initial_weight, self.itensors, y)
            A=A*(y.unsqueeze(1))
            delta_weight=self.alpha*(outer_prod-A)
            self.fc.weight=nn.Parameter(torch.add(self.fc.weight, delta_weight), requires_grad=False)
            self.exponential_average=torch.add(self.gamma*self.exponential_average,(1-self.gamma)*y)
            
    #Decays the overused weights and increases the underused weights using tanh functions.
    def weightDecay(self):
        average=torch.mean(self.exponential_average).item()
        A=self.exponential_average/average
        above_average=torch.where(A>1, A, 0.0)
        below_average=torch.where(A<1, A, 0.0)
        above_average_mask= above_average != 0
        below_average_mask= below_average != 0
        above_average=torch.where(above_average_mask, self.tanh(-1*(above_average-1))+1, 0.0)
        below_average=torch.where(below_average_mask, 0.001*self.tanh(1000*below_average)+1, 0.0)
        tanhed_averages=torch.add(above_average, below_average)
        
        print(average)

        #self.fc.weight=nn.Parameter(self.fc.weight*(tanhed_averages.unsqueeze(1)), requires_grad=False)
        
    #Feed forward
    def forward(self, x, clamped_output=None, train=1):
        input=x.clone()
        if self.isClassifier:
            x=self.fc(x)
            x=self.inhibition(x)
            self.updateWeightsHebbian(input, x, train)  
            return x
        x=self.fc(x)
        x=self.inhibition(x)
        self.updateWeightsHebbian(input, x, train)  
        self.weightDecay() 
        if self.scheduler is not None:
            self.alpha=self.scheduler.step()
        return x
    
    #Creates heatmap of randomly chosen feature selectors.
    def visualizeWeights(self, num_choices):
        weight=self.fc.weight
        random_indices = torch.randperm(self.fc.weight.size(0))[:num_choices]
        for ele in range(64):#Scalar tensor
            idx=ele
            random_feature_selector=weight[ele]
            heatmap=random_feature_selector.view(int(math.sqrt(self.fc.weight.size(1))), 
                                                 int(math.sqrt(self.fc.weight.size(1))))
            plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            plt.title("HeatMap of feature selector {} with lambda {}".format(idx, self.lamb))
            plt.colorbar()
            plt.show()
        averages=torch.log(self.exponential_average).tolist()
        plt.plot(range(len(averages)), averages)
        plt.xtitle
        return
