import os 
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import time
import models.learning as L
import models.helper_modules as M
from models.hyperparams import (LearningRule, WeightScale, oneHotEncode, InputProcessing,
                                Inhibition, WeightGrowth)
from models.MLP.layers import SoftHebbLayer, Hebbian_Layer


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
                if layers[idx].w_update == LearningRule.FullyOrthogonal:
                    layers[idx].update_weights_FullyOrthogonal(input, label_clamped_hlist[idx])
                elif layers[idx].w_update == LearningRule.Softhebb:
                    layers[idx].update_weight_softhebb(input, ulist[idx], label_clamped_hlist[idx])
                else:
                    layers[idx].update_weights_OrthogonalExclusive(input, label_clamped_hlist[idx])

                if layers[idx].weight_mod == WeightScale.WeightDecay:
                    layers[idx].weight_decay()
                else:
                    layers[idx].normalize_weights()

            elif idx == (nb_layers-1):
                if layers[idx].o_learning == LearningRule.Supervised:
                    layers[idx].classifier_update_supervised(label_clamped_hlist[idx-1], label_clamped_hlist[idx])
                elif layers[idx].o_learning == LearningRule.Orthogonal:
                    layers[idx].update_weights_FullyOrthogonal(label_clamped_hlist[idx-1], label_clamped_hlist[idx])
                elif layers[idx].o_learning == LearningRule.OutputContrastiveSupervised:
                    layers[idx].classifier_update_contrastive(label_clamped_hlist[idx-1], pred, label_clamped_hlist[idx])

    def TD_forward(self, x, labels):
        input = x.detach().clone()
        all_hlist_label, preact_hlist, pred = self.forward_clamped(x, labels)
        self.update_weights(input, all_hlist_label, preact_hlist, pred)
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)


class SoftNeuralNet(nn.Module):
    def __init__(self):
        super(SoftNeuralNet, self).__init__()
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
    def forward(self, x, clamped=None):
        for layer in self.layers.values():
            x = layer.forward(x, target=clamped)
        return x
    
    def forward_test(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def set_iteration(self, i):
        self.iteration = i
        
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def set_training_layers(self, layers_to_train):
        for layer in self.layers.values():
            if layer in layers_to_train:
                layer.train()
            else:
                layer.eval()


def MLPBaseline_Model(hsize, lamb, lr, e, wtd, gamma, nclasses, device, o, w, ws):
    mymodel = NeuralNet()
    heb_layer = Hebbian_Layer(784, hsize, lr, lamb, wtd, gamma, e, device=device, update=w, weight=ws)
    heb_layer2 = Hebbian_Layer(hsize, nclasses, lr, lamb, wtd, gamma, e, device=device, is_output_layer=True, output_learning=o)

    mymodel.add_layer('Hebbian1', heb_layer)
    mymodel.add_layer('Hebbian2', heb_layer2)

    return mymodel


def NewMLPBaseline_Model(hsize, lamb, w_lr, b_lr, l_lr, nclasses, device, initial_weight_norm=0.01):
    mymodel = SoftNeuralNet()
    heb_layer = SoftHebbLayer(inputdim=784, outputdim=hsize, w_lr=w_lr, b_lr=b_lr, l_lr=l_lr,
                              device=device, initial_lambda=lamb, initial_weight_norm=initial_weight_norm)
    
    heb_layer2 = SoftHebbLayer(hsize, nclasses, w_lr=w_lr, b_lr=b_lr, l_lr=l_lr, initial_lambda=lamb,
                               learningrule=LearningRule.SoftHebbOutputContrastive, is_output_layer=True, initial_weight_norm=initial_weight_norm)
    mymodel.add_layer('SoftHebbian1', heb_layer)
    mymodel.add_layer('SoftHebbian2', heb_layer2)

    return mymodel

def MultilayerSoftMLPModel(hsize, lamb, w_lr, b_lr, l_lr, nclasses, device):
    mymodel = SoftNeuralNet()
    heb_layer = SoftHebbLayer(inputdim=784, outputdim=512, w_lr=w_lr, b_lr=b_lr, l_lr=l_lr,
                              device=device, initial_lambda=lamb)
    heb_layer1 = SoftHebbLayer(inputdim=512, outputdim=512, w_lr=w_lr, b_lr=b_lr, l_lr=l_lr,
                              device=device, initial_lambda=lamb)
    heb_layer2 = SoftHebbLayer(inputdim=512, outputdim=512, w_lr=w_lr, b_lr=b_lr, l_lr=l_lr,
                              device=device, initial_lambda=lamb)
    heb_layer3 = SoftHebbLayer(inputdim=512, outputdim=512, w_lr=w_lr, b_lr=b_lr, l_lr=l_lr,
                              device=device, initial_lambda=lamb)
    heb_layer4 = SoftHebbLayer(512, nclasses, w_lr=w_lr, b_lr=b_lr, l_lr=l_lr, initial_lambda=lamb,
                               learningrule=LearningRule.SoftHebbOutputContrastive, is_output_layer=True)
    mymodel.add_layer('SoftHebbian1', heb_layer)
    mymodel.add_layer('SoftHebbian2', heb_layer1)
    mymodel.add_layer('SoftHebbian3', heb_layer2)
    mymodel.add_layer('SoftHebbian4', heb_layer3)
    mymodel.add_layer('SoftHebbian5', heb_layer4)

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


def view_weights(model, folder):
    for name, l in model.layers.items():
        visualize_weights(l, folder + '/' + name + '.png')


def visualize_weights(self, file):
    if hasattr(self, 'weight'):
        nb = int(math.ceil(math.sqrt(self.output_dim)))
        if not self.is_output_layer:
            fig, axes = plt.subplots(nb, nb, figsize=(32,32))
            nb_ele = self.output_dim
        else :
            nb_ele = self.weight.size(0)
            fig, axes = plt.subplots(nb, nb, figsize=(32,32))
            
        weight = self.weight.detach().to('cpu')
        weight = self.weight.detach().to('cpu')
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

    else:
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