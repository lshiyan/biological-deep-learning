import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from numpy import outer
import warnings

warnings.filterwarnings("ignore")

#Hebbian learning layer that implements lateral inhibition in output. Not trained through supervision.
class HebbianLayer (nn.Module):
    def __init__(self, args, input_dimension, output_dimension, classifier, lamb=2, heb_lr=0.001, gamma=0.99, eps=10e-5):
        super (HebbianLayer, self).__init__()
        self.input_dimension=input_dimension
        self.output_dimension=output_dimension
        self.lamb=lamb
        self.alpha = heb_lr
        self.fc=nn.Linear(self.input_dimension, self.output_dimension, bias=True)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.softplus=nn.Softplus()
        self.tanh=nn.Tanh()
        self.softmax=nn.Softmax()
        self.isClassifier=classifier
        self.scheduler=None
        self.eps=eps
        
        self.exponential_average=torch.zeros(self.output_dimension).to(args.device_id)
        self.gamma=gamma
        
        self.itensors=self.createITensors().to(args.device_id)
        
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
        max_ele=torch.max(x).item()
        x=torch.pow(x, self.lamb)
        x/=abs(max_ele)**self.lamb
        return x
    
    def updateWeightsHebbianClassifier(self, input, output, clamped_output=None, train=1):
        if train:
            u=output.squeeze()
            x=input.squeeze()
            y=torch.softmax(u, dim=0)
            A=None
            if clamped_output != None:
                outer_prod=torch.outer(clamped_output-y,x)
                u_times_y=torch.mul(u,y)
                A=outer_prod-self.fc.weight*(u_times_y.unsqueeze(1))
            else:
                A=torch.outer(y,x)
            A=self.fc.weight+self.alpha*A
            weight_maxes=torch.max(A, dim=1).values
            self.fc.weight=nn.Parameter(A/weight_maxes.unsqueeze(1), requires_grad=False)
            self.fc.weight[:, 0] = 0
        
    #Employs Sanger's Rule, deltaW_(ij)=alpha*x_j*y_i-alpha*y_i*sum(k=1 to i) (w_(kj)*y_k)
    #Calculates outer product of input and output and adds it to matrix.
    def updateWeightsHebbian(self, input, output, clamped_output=None, train=1):
        if train:
            x=input.squeeze()
            y=output.squeeze()
            outer_prod=torch.tensor(torch.outer(y, x))
            initial_weight=torch.transpose(self.fc.weight.clone().detach(), 0,1)
            A=torch.einsum('jk, lkm, m -> lj', initial_weight, self.itensors, y)
            A=A*(y.unsqueeze(1))
            delta_weight=self.alpha*(outer_prod-A)
            self.fc.weight=nn.Parameter(torch.add(self.fc.weight, delta_weight), requires_grad=False)
            self.exponential_average=torch.add(self.gamma*self.exponential_average,(1-self.gamma)*y)
        
    def updateBias(self, output, train=1):
        if train:
            y=output.clone().detach().squeeze()
            exponential_bias=torch.exp(-1*self.fc.bias)
            A=torch.mul(exponential_bias, y)-1
            A=self.fc.bias+self.alpha*A
            bias_maxes=torch.max(A, dim=0).values
            self.fc.bias=nn.Parameter(A/bias_maxes.item(), requires_grad=False)

    #Decays the overused weights and increases the underused weights using tanh functions.
    def weightDecay(self):
        average=torch.mean(self.exponential_average).item()
        A=self.exponential_average/average
        growth_factor_positive=self.eps*self.tanh(-self.eps*(A-1))+1
        growth_factor_negative=torch.reciprocal(growth_factor_positive)
        positive_weights=torch.where(self.fc.weight>0, self.fc.weight, 0.0)
        negative_weights=torch.where(self.fc.weight<0, self.fc.weight, 0.0)
        positive_weights=positive_weights*growth_factor_positive.unsqueeze(1)
        negative_weights=negative_weights*growth_factor_negative.unsqueeze(1)
        self.fc.weight=nn.Parameter(torch.add(positive_weights, negative_weights), requires_grad=False)
        if (self.fc.weight.isnan().any()):
            print("NAN WEIGHT")
    
    #Feed forward
    def forward(self, x, clamped_output=None, train=1):
        input=x.clone()
        if self.isClassifier:
            x=self.fc(x)
            self.updateWeightsHebbianClassifier(input, x, clamped_output=clamped_output, train=train)
            #self.updateBias(x, train=train)
            x=self.softmax(x)
            return x
        else:
            x=self.fc(x)
            x=self.inhibition(x)
            self.updateWeightsHebbian(input, x, train)
            #self.updateBias(x, train=train)
            self.weightDecay() 
            return x
    
    #Counts the number of active feature selectors (above a certain cutoff beta).
    def activeClassifierWeights(self, beta):
        weights=self.fc.weight
        active=torch.where(weights>beta, weights, 0.0)
        return active.nonzero().size(0)
    
    #Creates heatmap of randomly chosen feature selectors.
    def visualizeWeights(self, classifier=0):
        weight = self.fc.weight
        if classifier:
            fig, axes = plt.subplots(2, 5, figsize=(16, 8))
            for ele in range(10):  
                random_feature_selector = weight[ele]
                heatmap = random_feature_selector.view(int(math.sqrt(self.fc.weight.size(1))),
                                                        int(math.sqrt(self.fc.weight.size(1))))
                ax = axes[ele // 5, ele % 5]
                im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
                fig.colorbar(im, ax=ax)
                ax.set_title(f'Weight {ele}')
            plt.tight_layout()
            plt.show()
        else:
            fig, axes = plt.subplots(8, 8, figsize=(16, 16))
            for ele in range(self.output_dimension): 
                random_feature_selector = weight[ele]
                heatmap = random_feature_selector.view(int(math.sqrt(self.fc.weight.size(1))),
                                                        int(math.sqrt(self.fc.weight.size(1))))
                ax = axes[ele // 8, ele % 8]
                im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
                fig.colorbar(im, ax=ax)
                ax.set_title(f'Weight {ele}')
            plt.tight_layout()
            plt.show()
