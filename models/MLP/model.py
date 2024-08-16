import os 
import torch
import torch.nn as nn
from enum import Enum
import math
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time




class ImageType(Enum):
    Gray = 1
    RGB = 2

class ClassifierLearning(Enum):
    Supervised = 1
    Contrastive = 2
    Orthogonal = 3

class Learning(Enum):
    FullyOrthogonal = 1
    OrthogonalExclusive = 2

class WeightScale(Enum):
    WeightDecay = 1
    WeightNormalization = 2

class ClassifierLearning(Enum):
    Supervised = 1
    Contrastive = 2
    Orthogonal = 3

class Learning(Enum):
    FullyOrthogonal = 1
    OrthogonalExclusive = 2

class WeightScale(Enum):
    WeightDecay = 1
    WeightNormalization = 2




def oneHotEncode(labels, num_classes, device):
    one_hot_encoded = torch.zeros(len(labels), num_classes).to(device)
    one_hot_encoded.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot_encoded.squeeze()

def mlp_print_weight(model):
    for l in model.layers.values():
        print(l.feedforward.weight)




class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleDict()
        self.iteration = 3
        self.correct = 0
        self.tot = 0

    def save_acc(self, cor, seen):
        self.correct += cor
        self.tot += seen
    
    def reset_acc(self):
        self.correct = 0
        self.tot = 0
    
    def add_layer(self, name, layer):
        self.layers[name] = layer

    def visualize_weights(self):
        for layer_name, module in self.layers.items():
            module.visualize_weights()

    def forward(self, x, clamped):
        for layer in self.layers.values():
            #print(x)
            x = layer(x, clamped)
        return x
    
    def forward_test(self, x):
        for layer in self.layers.values():
            x = layer.forward_test(x)
        return x
    
    def set_iteration(self, i):
        self.iteration = i
    
    def forward_clamped(self, x, clamped):
        input = x.detach().clone()
        layers = list(self.layers.values())
        nb_layers = len(layers)
        hlist = [None]*nb_layers
        for iter in range(self.iteration):
            hlist[-1] = clamped
            x = input
            for idx in range(nb_layers-1):
                w_topdown = layers[idx+1].feedforward.weight.detach().clone()
                h_topdown = hlist[idx+1]
                x = layers[idx].TD_forward(x, w_topdown, h_topdown)
                x = layers[idx].TD_forward(x, w_topdown, h_topdown)
                hlist[idx] = x
        return hlist

    def TD_forward_test(self, x):
        input = x.detach().clone()
        layers = list(self.layers.values())
        nb_layers = len(layers)
        hlist = [None]*nb_layers
        for _ in range(self.iteration):
            x = input
            for idx in range(nb_layers-1):
                w = layers[idx+1].feedforward.weight.detach().clone()
                h_l = hlist[idx+1]
                x = layers[idx].TD_forward(x, w, h_l)
                hlist[idx] = x
            x_L = layers[-1].forward_test(hlist[-2])
            hlist[-1] = x_L
        return x_L
    
    def update_weights(self, input, label_clamped_hlist):
        layers = list(self.layers.values())
        nb_layers = len(layers)
        for idx in range(nb_layers):
            if idx == 0:
                if layers[idx].w_update == Learning.FullyOrthogonal:
                    layers[idx].update_weights_FullyOrthogonal(input.squeeze(0), label_clamped_hlist[idx].squeeze(0))
                else:
                    layers[idx].update_weights_OrthogonalExclusive(input.squeeze(0), label_clamped_hlist[idx].squeeze(0))

                if layers[idx].weight_mod == WeightScale.WeightDecay:
                    layers[idx].weight_decay()
                else:
                    layers[idx].normalize_weights()

                if layers[idx].w_update == Learning.FullyOrthogonal:
                    layers[idx].update_weights_FullyOrthogonal(input.squeeze(0), label_clamped_hlist[idx].squeeze(0))
                else:
                    layers[idx].update_weights_OrthogonalExclusive(input.squeeze(0), label_clamped_hlist[idx].squeeze(0))

                if layers[idx].weight_mod == WeightScale.WeightDecay:
                    layers[idx].weight_decay()
                else:
                    layers[idx].normalize_weights()

            elif idx == (nb_layers-1):
                if layers[idx].o_learning == ClassifierLearning.Supervised:
                    layers[idx].classifier_update_supervised(label_clamped_hlist[idx-1].squeeze(0), label_clamped_hlist[idx].squeeze(0))
                elif layers[idx].o_learning == ClassifierLearning.Orthogonal:
                    layers[idx].update_weights_FullyOrthogonal(label_clamped_hlist[idx-1].squeeze(0), label_clamped_hlist[idx].squeeze(0))
                if layers[idx].o_learning == ClassifierLearning.Supervised:
                    layers[idx].classifier_update_supervised(label_clamped_hlist[idx-1].squeeze(0), label_clamped_hlist[idx].squeeze(0))
                elif layers[idx].o_learning == ClassifierLearning.Orthogonal:
                    layers[idx].update_weights_FullyOrthogonal(label_clamped_hlist[idx-1].squeeze(0), label_clamped_hlist[idx].squeeze(0))


    def TD_forward(self, x, labels):
        input = x.detach().clone()
        all_hlist_label = self.forward_clamped(x, labels)
        self.update_weights(input, all_hlist_label)
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)



class Hebbian_Layer(nn.Module):
    def __init__(self, inputdim, outputdim, lr, lamb, w_decrease, gamma, eps, device, is_output_layer=False, output_learning=ClassifierLearning.Supervised, update=Learning.FullyOrthogonal, weight=WeightScale.WeightDecay):
        super(Hebbian_Layer, self).__init__()
        self.input_dim = inputdim
        self.output_dim = outputdim
        self.lr = lr
        self.lamb = lamb
        self.decrease = w_decrease
        self.gamma = gamma
        self.epsilon = eps
        self.is_output_layer = is_output_layer

        self.feedforward = nn.Linear(self.input_dim, self.output_dim, bias=False)

        for param in self.feedforward.parameters():
            param=torch.nn.init.uniform_(param, a=0.0, b=1.0)
            param.requires_grad_(False)
        
        self.exponential_average=torch.zeros(self.output_dim).to(device)
        self.o_learning = output_learning
        self.w_update = update
        self.weight_mod = weight
        self.device = device


    def inhibition(self, x):
        x=nn.ReLU()(x)
        max_ele=torch.max(x).item()
        x/=max_ele
        x=torch.pow(x, self.lamb)
        return x
        
    # Fully orthogonal Sanger variant
    def update_weights_FullyOrthogonal(self, input, output):
        x=input.detach().clone().squeeze()
        y=output.detach().clone().squeeze()
        initial_weight = self.feedforward.weight.detach().clone()
        outer_prod = torch.outer(y, x)
        ytw = torch.matmul(y.unsqueeze(0), initial_weight)
        norm_term = torch.outer(y.squeeze(0), ytw.squeeze(0))
        delta_weight = self.lr*(outer_prod - norm_term)
        new_weights = torch.add(initial_weight, delta_weight)
        self.feedforward.weight=nn.Parameter(new_weights, requires_grad=False)
        self.exponential_average=torch.add(self.gamma*self.exponential_average,(1-self.gamma)*y)
    
    def update_weights_OrthogonalExclusive(self, input: torch.Tensor, output: torch.Tensor):
        x: torch.Tensor = input.clone().detach().float().squeeze().to(self.device)
        x.requires_grad_(False)
        y: torch.Tensor = output.clone().detach().float().squeeze().to(self.device)
        y.requires_grad_(False)
        
        outer_prod: torch.Tensor = torch.einsum("i, j -> ij", y, x)
        
        weights: torch.Tensor = self.feedforward.weight.clone().detach().to(self.device)

        W_norms: torch.Tensor = torch.norm(weights, dim=1, keepdim=False).to(self.device)

        scaling_terms: torch.Tensor = W_norms.reshape(self.feedforward.weight.size(0),1) / (W_norms.reshape(1,self.feedforward.weight.size(0)) + 10e-8)

        remove_diagonal: torch.Tensor = torch.ones(self.feedforward.weight.size(0), self.feedforward.weight.size(0)) - torch.eye(self.feedforward.weight.size(0))
        remove_diagonal: torch.Tensor = remove_diagonal.reshape(self.feedforward.weight.size(0), self.feedforward.weight.size(0), 1).to(self.device)
        scaled_weight: torch.Tensor = scaling_terms.reshape(self.feedforward.weight.size(0), self.feedforward.weight.size(0), 1) * weights.reshape(1, self.feedforward.weight.size(0), self.feedforward.weight.size(1)).to(self.device)

        scaled_weight: torch.Tensor = scaled_weight * remove_diagonal

        norm_term: torch.Tensor = torch.einsum("i, k, ikj -> ij", y, y, scaled_weight)
        
        computed_rule: torch.Tensor = self.lr*(outer_prod - norm_term)

        self.exponential_average=torch.add(self.gamma*self.exponential_average,(1-self.gamma)*y)
        self.feedforward.weight=nn.Parameter(torch.add(computed_rule, weights), requires_grad=False)
    
    def classifier_update_contrastive(self, input, output, true_output):
        output = output.detach().clone().squeeze()
        input = input.detach().clone().squeeze()
        true_output = true_output.detach().clone().squeeze()
        outer = torch.outer(nn.ReLU()(true_output - output), input)
        self.feedforward.weight=nn.Parameter(torch.add(outer, self.feedforward.weight.detach().clone()), requires_grad=False)

    def classifier_update_supervised(self, input, output):
        output = output.detach().clone().squeeze()
        input = input.detach().clone().squeeze()
        outer = torch.outer(output, input)
        self.feedforward.weight=nn.Parameter(torch.add(outer, self.feedforward.weight.detach().clone()), requires_grad=False)
    
    def normalize_weights(self):
        new_weights = self.feedforward.weight.detach().clone()
        #print(new_weights.shape)
        #print(new_weights)
        new_weights_norm = torch.norm(new_weights, p=2, dim=1, keepdim=True)
        new_weights_norm = torch.norm(new_weights, p=2, dim=1, keepdim=True)
        #print(new_weights_norm)
        #print(new_weights_norm.shape)
        new_weights = new_weights / (new_weights_norm)
        new_weights = new_weights / (new_weights_norm)
        #print(new_weights)
        self.feedforward.weight=nn.Parameter(new_weights, requires_grad=False)
    
    def weight_decay(self):
        average=torch.mean(self.exponential_average).item()
        A=self.exponential_average/average
        growth_factor_positive=self.epsilon*nn.Tanh()(-self.epsilon*(A-1))+1
        growth_factor_negative=torch.reciprocal(growth_factor_positive)
        positive_weights=torch.where(self.feedforward.weight>0, self.feedforward.weight, 0.0)
        negative_weights=torch.where(self.feedforward.weight<0, self.feedforward.weight, 0.0)
        positive_weights=positive_weights*growth_factor_positive.unsqueeze(1)
        negative_weights=negative_weights*growth_factor_negative.unsqueeze(1)
        self.feedforward.weight=nn.Parameter(torch.add(positive_weights, negative_weights), requires_grad=False)

    def forward(self, x, clamped):
        x = x.reshape(-1).unsqueeze(0)
        input = x.detach().clone()
        #self.normalize_weights()
        x = self.feedforward(x)
        x = self.inhibition(x)
        if self.is_output_layer:
            if self.o_learning == ClassifierLearning.Contrastive:
                self.classifier_update_contrastive(input, x, clamped)
            elif self.o_learning == ClassifierLearning.Supervised:
                self.classifier_update_supervised(input, clamped)
            else:
                self.update_weights_FullyOrthogonal(input, clamped)
            if self.o_learning == ClassifierLearning.Contrastive:
                self.classifier_update_contrastive(input, x, clamped)
            elif self.o_learning == ClassifierLearning.Supervised:
                self.classifier_update_supervised(input, clamped)
            else:
                self.update_weights_FullyOrthogonal(input, clamped)
        else:
            if self.w_update == Learning.FullyOrthogonal:
                self.update_weights_FullyOrthogonal(input, x)
            else:
                self.update_weights_OrthogonalExclusive(input, x)
            if self.weight_mod == WeightScale.WeightDecay:
                self.weight_decay()
            else:
                self.normalize_weights()
            if self.w_update == Learning.FullyOrthogonal:
                self.update_weights_FullyOrthogonal(input, x)
            else:
                self.update_weights_OrthogonalExclusive(input, x)
            if self.weight_mod == WeightScale.WeightDecay:
                self.weight_decay()
            else:
                self.normalize_weights()
        return x
            
    def forward_test(self, x):
        x = x.reshape(-1).unsqueeze(0)
        x = self.feedforward(x)
        x = self.inhibition(x)
        return x
    
    def TD_forward(self, x, w, h_l):
        if h_l == None:
            h = self.feedforward(x)
        else :
            #print(self.feedforward(x))
            #print(torch.matmul(h_l, w))
            #print(self.feedforward(x))
            #print(torch.matmul(h_l, w))
            h = self.feedforward(x) + self.decrease*torch.matmul(h_l, w)
        return self.inhibition(h)
    


def MLPBaseline_Model(hsize, lamb, lr, e, wtd, gamma, nclasses, device, o, w, ws):
    mymodel = NeuralNet()
    heb_layer = Hebbian_Layer(784, hsize, lr, lamb, wtd, gamma, e, device=device, update=w, weight=ws)
    heb_layer2 = Hebbian_Layer(hsize, nclasses, lr, lamb, wtd, gamma, e, device=device, is_output_layer=True, output_learning=o)

    mymodel.add_layer('Hebbian1', heb_layer)
    mymodel.add_layer('Hebbian2', heb_layer2)

    return mymodel



def Save_Model(mymodel, dataset, device):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    foldername = os.getcwd() + '/SavedModels/MLP_FF_' + dataset + '_' + str(device) + '_' + timestr

    if not os.path.exists(foldername):
        os.mkdir(foldername)

    if not os.path.isfile(foldername + '/model'):
        torch.save(mymodel.state_dict(), foldername + '/model')

    view_weights(mymodel, foldername)



def MLPBaseline_Experiment(epoch, mymodel, dataloader, dataset, nclasses):
def MLPBaseline_Experiment(epoch, mymodel, dataloader, dataset, nclasses):

    mymodel.train()
    cn = 0
    for _ in range(epoch):
        for data in tqdm(dataloader):
            inputs, labels=data
            #print(inputs)
            mymodel.forward(inputs, oneHotEncode(labels, nclasses))
            #cn += 1
            #if cn == 10:
            #    return mymodel
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    foldername = os.getcwd() + '/SavedModels/MLP_FF_' + dataset + '_' + timestr
    os.mkdir(foldername)

    torch.save(mymodel.state_dict(), foldername + '/model')

    view_weights(mymodel, foldername)
    return mymodel

def TDBaseline_Experiment(epoch, mymodel, dataloader, dataset, nclasses):

    mymodel.train()
    cn = 0
    for _ in range(epoch):
        for data in tqdm(dataloader):
            inputs, labels=data
            #print(inputs)
            mymodel.TD_forward(inputs, oneHotEncode(labels, nclasses))
            #cn += 1
            #if cn == 10:
            #    return mymodel
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    foldername = os.getcwd() + '/SavedModels/TD_FF_' + dataset + '_' + timestr
    os.mkdir(foldername)

    torch.save(mymodel.state_dict(), foldername + '/model')

    view_weights(mymodel, foldername)
    return mymodel

def TDBaseline_Experiment(epoch, mymodel, dataloader, dataset, nclasses):

    mymodel.train()
    cn = 0
    for _ in range(epoch):
        for data in tqdm(dataloader):
            inputs, labels=data
            #print(inputs)
            mymodel.TD_forward(inputs, oneHotEncode(labels, nclasses))
            #cn += 1
            #if cn == 10:
            #    return mymodel
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    foldername = os.getcwd() + '/SavedModels/TD_FF_' + dataset + '_' + timestr
    os.mkdir(foldername)

    torch.save(mymodel.state_dict(), foldername + '/model')

    view_weights(mymodel, foldername)
    return mymodel


def view_weights(model, folder):
    for name, l in model.layers.items():
        visualize_weights(l, folder + '/' + name + '.png')

def visualize_weights(self, file):
    nb = int(math.ceil(math.sqrt(self.output_dim)))
    if not self.is_output_layer:
        fig, axes = plt.subplots(nb, nb, figsize=(32,32))
        nb_ele = self.output_dim
    else :
        nb_ele = self.feedforward.weight.size(0)
        fig, axes = plt.subplots(nb, nb, figsize=(32,32))
        
    weight = self.feedforward.weight.detach().to('cpu')
    weight = self.feedforward.weight.detach().to('cpu')
    for ele in range(nb_ele):
        random_feature_selector = weight[ele]
        heatmap = random_feature_selector.view(int(math.sqrt(weight.size(1))),
                                                int(math.sqrt(weight.size(1))))
        ax = axes[ele // nb, ele % nb]
        im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
        fig.colorbar(im, ax=ax)
        ax.set_title(f'Weight {ele}')

    plt.tight_layout()
    if not os.path.exists(file):
        plt.savefig(file)