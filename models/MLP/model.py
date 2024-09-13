import os 
import torch
import torch.nn as nn
from enum import Enum
import math
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time
import models.learning as L


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
    Softhebb = 3

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
    
    def add_layer(self, name, layer):
        self.layers[name] = layer

    def visualize_weights(self):
        for layer_name, module in self.layers.items():
            module.visualize_weights()

    """
    Used in Feedforward models
    """
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
    
    """
    Used in Topdown models
    """
    def forward_clamped(self, x, clamped):
        input = x.detach().clone()
        layers = list(self.layers.values())
        nb_layers = len(layers)
        hlist = [None]*nb_layers
        ulist = [None]*nb_layers
        for iter in range(self.iteration):
            hlist[-1] = clamped
            x = input
            for idx in range(nb_layers-1):
                w_topdown = layers[idx+1].feedforward.weight.detach().clone()
                h_topdown = hlist[idx+1]
                u, x = layers[idx].TD_forward(x, w_topdown, h_topdown)
                hlist[idx] = x
                ulist[idx] = u
        x_L = layers[-1].forward_test(hlist[-2])
        return hlist, ulist, x_L

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
                u, x = layers[idx].TD_forward(x, w, h_l)
                hlist[idx] = x
            x_L = layers[-1].forward_test(hlist[-2])
            hlist[-1] = x_L
        return x_L
    
    def update_weights(self, input, label_clamped_hlist, ulist, pred):
        layers = list(self.layers.values())
        nb_layers = len(layers)
        for idx in range(nb_layers):
            if idx == 0:
                if layers[idx].w_update == Learning.FullyOrthogonal:
                    layers[idx].update_weights_FullyOrthogonal(input, label_clamped_hlist[idx])
                elif layers[idx].w_update == Learning.Softhebb:
                    layers[idx].update_weight_softhebb(input, ulist[idx], label_clamped_hlist[idx])
                else:
                    layers[idx].update_weights_OrthogonalExclusive(input, label_clamped_hlist[idx])

                if layers[idx].weight_mod == WeightScale.WeightDecay:
                    layers[idx].weight_decay()
                else:
                    layers[idx].normalize_weights()

            elif idx == (nb_layers-1):
                if layers[idx].o_learning == ClassifierLearning.Supervised:
                    layers[idx].classifier_update_supervised(label_clamped_hlist[idx-1], label_clamped_hlist[idx])
                elif layers[idx].o_learning == ClassifierLearning.Orthogonal:
                    layers[idx].update_weights_FullyOrthogonal(label_clamped_hlist[idx-1], label_clamped_hlist[idx])
                elif layers[idx].o_learning == ClassifierLearning.Contrastive:
                    layers[idx].classifier_update_contrastive(label_clamped_hlist[idx-1], pred, label_clamped_hlist[idx])

    def TD_forward(self, x, labels):
        input = x.detach().clone()
        all_hlist_label, preact_hlist, pred = self.forward_clamped(x, labels)
        self.update_weights(input, all_hlist_label, preact_hlist, pred)
    
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
        y = output.squeeze()
        weight = self.feedforward.weight
        delta_w = L.update_weights_FullyOrthogonal(input, output, weight)
        delta_w = torch.mean(delta_w, dim=0)
        delta_weight = self.lr*(delta_w)
        new_weights = torch.add(weight, delta_weight)
        self.feedforward.weight=nn.Parameter(new_weights, requires_grad=False)
        self.exponential_average=torch.add(self.gamma*self.exponential_average,(1-self.gamma)*y)
    
    def update_weights_OrthogonalExclusive(self, input: torch.Tensor, output: torch.Tensor):
        y = output.squeeze()
        weight = self.feedforward.weight
        delta_w = L.update_weights_OrthogonalExclusive(input, output, weight, self.device)
        delta_w = torch.mean(delta_w, dim=0)
        computed_rule: torch.Tensor = self.lr*(delta_w)
        self.feedforward.weight=nn.Parameter(torch.add(computed_rule, weight), requires_grad=False)
        self.exponential_average=torch.add(self.gamma*self.exponential_average,(1-self.gamma)*y)
    
    def update_weight_softhebb(self, input, preac, postac):
        y = postac.squeeze()
        weight = self.feedforward.weight 
        delta_w = L.update_weight_softhebb(input, preac, postac, weight)
        delta_w = self.lr*delta_w
        self.feedforward.weight=nn.Parameter(torch.add(delta_w, weight), requires_grad=False)
        self.exponential_average=torch.add(self.gamma*self.exponential_average,(1-self.gamma)*y)
    
    def classifier_update_contrastive(self, input, output, true_output):
        output = output.squeeze()
        input = input.squeeze()
        true_output = true_output.squeeze()
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
        h = self.feedforward(x)
        x = self.inhibition(h)
        if self.is_output_layer:
            if self.o_learning == ClassifierLearning.Contrastive:
                self.classifier_update_contrastive(input, x, clamped)
            elif self.o_learning == ClassifierLearning.Supervised:
                self.classifier_update_supervised(input, clamped)
            else:
                self.update_weights_FullyOrthogonal(input, clamped)
        else:
            if self.w_update == Learning.FullyOrthogonal:
                self.update_weights_FullyOrthogonal(input, x)
            elif self.w_update == Learning.Softhebb:
                self.update_weight_softhebb(input, h, x)
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
            h = self.feedforward(x) + self.decrease*torch.matmul(h_l, w)
        return h, self.inhibition(h)
    


def MLPBaseline_Model(hsize, lamb, lr, e, wtd, gamma, nclasses, device, o, w, ws):
    mymodel = NeuralNet()
    heb_layer = Hebbian_Layer(784, hsize, lr, lamb, wtd, gamma, e, device=device, update=w, weight=ws)
    heb_layer2 = Hebbian_Layer(hsize, nclasses, lr, lamb, wtd, gamma, e, device=device, is_output_layer=True, output_learning=o)

    mymodel.add_layer('Hebbian1', heb_layer)
    mymodel.add_layer('Hebbian2', heb_layer2)

    return mymodel



def Save_Model(mymodel, dataset, device, topdown, acc):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if topdown:
        foldername = os.getcwd() + '/SavedModels4/MLP_TD_' + dataset + '_' + str(device) + '_' + str(acc) + '_' + timestr
    else:
        foldername = os.getcwd() + '/SavedModels4/MLP_FF_' + dataset + '_' + str(device) + '_' + str(acc) + '_' + timestr

    if not os.path.exists(foldername):
        os.mkdir(foldername)

    if not os.path.isfile(foldername + '/model'):
        torch.save(mymodel.state_dict(), foldername + '/model')

    view_weights(mymodel, foldername)


def MLPBaseline_Experiment(epoch, mymodel, dataloader, dataset, nclasses, device):

    mymodel.train()
    cn = 0
    for _ in range(epoch):
        for data in tqdm(dataloader):
            inputs, labels=data
            #print(inputs)
            mymodel.forward(inputs, oneHotEncode(labels, nclasses, device))
            #cn += 1
            #if cn == 10:
            #    return mymodel
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    foldername = os.getcwd() + '/SavedModels/MLP_FF_' + dataset + '_' + timestr
    os.mkdir(foldername)

    torch.save(mymodel.state_dict(), foldername + '/model')

    view_weights(mymodel, foldername)
    return mymodel

def TDBaseline_Experiment(epoch, mymodel, dataloader, dataset, nclasses, device):

    mymodel.train()
    cn = 0
    for _ in range(epoch):
        for data in tqdm(dataloader):
            inputs, labels=data
            #print(inputs)
            mymodel.TD_forward(inputs, oneHotEncode(labels, nclasses, device))
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