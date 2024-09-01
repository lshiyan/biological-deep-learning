import os 
import torch
import torch.nn as nn
from enum import Enum
import math
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F



class ImageType(Enum):
    Gray = 1
    RGB = 2

class ClassifierLearning(Enum):
    Supervised = 1
    Contrastive = 2
    Orthogonal = 3
    SoftHebb = 4

class Learning(Enum):
    Sanger = 1
    OrthogonalExclusive = 2
    SoftHebb = 3

class WeightScale(Enum):
    WeightDecay = 1
    WeightNormalization = 2
    No = 3


def oneHotEncode(labels, num_classes, device):
    one_hot_encoded = torch.zeros(len(labels), num_classes).to(device)
    one_hot_encoded.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot_encoded.squeeze()

def update_weight_softhebb(input, preactivation, output, weight):
    # input_shape = batch, in_dim
    # output_shape = batch, out_dim = preactivation_shape
    # weight_shape = out_dim, in_dim
    b, indim = input.shape
    b, outdim = output.shape
    y = output.reshape(b, outdim, 1)
    x = input.reshape(b, 1, indim)
    deltas = y * (x - torch.matmul(preactivation, weight).reshape(b, 1, indim))
    delta = torch.mean(deltas, dim=0)
    return delta

class ConvolutionalNeuralNet(nn.Module):
    def __init__(self, device):
        super(ConvolutionalNeuralNet, self).__init__()
        self.layers = nn.ModuleDict()
        self.iteration = 3
        self.correct = 0
        self.tot = 0
        self.device = device
    
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
            true_value = clamped if layer.is_output_layer else None
            u, x = layer(x, clamped=true_value)
        return x
    
    def forward_test(self, x):
        for layer in self.layers.values():
            x = layer.forward_test(x)
        return x

    def TD_inference(self, input, clamped_output=None):
        with torch.no_grad():
            layers = list(self.layers.values())
            nb_layers = len(layers)
            hlist = [None] * nb_layers
            ulist = [None] * nb_layers
            hlist[-1] = clamped_output
            for _ in range(self.iteration):
                x = input.clone()
                for idx in range(nb_layers-1):
                    h_l = hlist[idx+1]
                    u, h = layers[idx].TD_forward(x, h_l)
                    hlist[idx] = h
                    ulist[idx] = u
                lastu, prediction = layers[-1].forward(hlist[-2])
                if clamped_output is None:
                    hlist[-1] = prediction
                ulist[-1] = lastu
            return ulist, hlist, prediction

    def initialize_TD_weights(self):
        layers = list(self.layers.values())
        nb_layers = len(layers)
        for i in range(nb_layers-2):
            layers[i].convolutionTD.weight = nn.Parameter(layers[i+1].convolution.weight.detach().clone(), requires_grad=False)

    def TD_forward(self, x, clamped):
        input = x.detach().clone()
        self.initialize_TD_weights()
        hlist, x_L = self.TD_inference(input, clamped)
        self.update_weights(x, hlist, x_L)
    
    def update_weights(self, inp, hlist, ulist, prediction):
        layers = list(self.layers.values())
        nb_layers = len(layers)
        hlist = [inp] + hlist
        for idx in range(nb_layers-1):
            i = hlist[idx]
            u = ulist[idx]
            o = hlist[idx+1]
            layers[idx].update_weights(i, None, o, u)
        i = hlist[-2].squeeze().squeeze().reshape(-1)
        layers[-1].update_weights(i, hlist[-1], prediction, ulist[-1])
        #layers[-1].weight_decay()
        
    def TD_forward_test(self, x):
        input = x.detach().clone()
        layers = list(self.layers.values())
        nb_layers = len(layers)
        hlist = [None]*nb_layers
        for _ in range(self.iteration):
            x = input
            for idx in range(nb_layers-1):
                h_l = hlist[idx+1]
                _, x = layers[idx].TD_forward(x, h_l)
                hlist[idx] = x
            x_l = layers[-2].TD_forward(hlist[-3], hlist[-1], layers[-1].feedforward.weight)
            #x_l = layers[-2].forward_test(hlist[-3])
            hlist[-2] = x_l
            x_L = layers[-1].forward_test(hlist[-2])
            hlist[-1] = x_L
        return hlist[-1]
    
    def set_iteration(self, i):
        self.iteration = i

    def view_map(self, fmaps):
        if fmaps.size(1) == 1:
            plt.imshow(fmaps[0][0], interpolation='nearest')
            plt.colorbar()
            plt.show()
        else:
            fig, axes = plt.subplots(2, fmaps.size(1)//2, figsize=(20,5))
            for i in range(fmaps.size(1)//2):
                img = fmaps[0][i]
                axes[0, i].imshow(img, interpolation='nearest')
            for i in range(fmaps.size(1)//2):
                img = fmaps[0][i+fmaps.size(1)//2]
                axes[1, i].imshow(img, interpolation='nearest')
            plt.show()
    
    def visualize_featuremaps(self, x):
        input = x.detach().clone()
        layers = list(self.layers.values())
        nb_layers = len(layers)
        hlist = [None]*nb_layers
        for _ in range(self.iteration):
            x = input
            for idx in range(nb_layers-2):
                h_l = hlist[idx+1]
                x = layers[idx].TD_forward(x, h_l)
                self.view_map(x)
                hlist[idx] = x
            x_l = layers[-2].TD_forward(hlist[-3], hlist[-1], layers[-1].feedforward.weight)
            hlist[-2] = x_l
            self.view_map(x_l)
            x_L = layers[-1].forward_test(hlist[-2])
            hlist[-1] = x_L


class Hebbian_Layer(nn.Module):
    def __init__(self, inputdim, outputdim, lr, lamb, rho, gamma, eps, device, is_output_layer=False,
                 output_learning=ClassifierLearning.Supervised, update=Learning.OrthogonalExclusive,
                 weight=WeightScale.WeightDecay, triangle=False):
        super(Hebbian_Layer, self).__init__()
        self.input_dim = inputdim
        self.output_dim = outputdim
        self.lr = lr
        self.lamb = lamb
        self.rho = rho
        self.gamma = gamma
        self.epsilon = eps
        self.is_output_layer = is_output_layer
        self.device = device
        self.triangle = False

        self.feedforward = nn.Linear(self.input_dim, self.output_dim, bias=False)

        for param in self.feedforward.parameters():
            param=torch.nn.init.uniform_(param, a=0.0, b=1.0)
            param.requires_grad_(False)
        
        self.exponential_average=torch.zeros(self.output_dim).to(self.device)
        self.o_learning = output_learning
        self.w_update = update
        self.weight_mod = weight


    def inhibition(self, x):
        if self.triangle:
            m = torch.mean(x)
            x = x - m
        x=torch.relu(x)
        max_ele=torch.max(x).item()
        x/=(max_ele + 1e-9)
        x=torch.pow(x, self.lamb)
        return x

    #SoftHebb
    def update_weights_softhebb(self, input, preactivation, output):
        with torch.no_grad():
            y = output.detach().clone().squeeze()
            initial_weight = self.feedforward.weight
            delta_w = update_weight_softhebb(input, preactivation, output, initial_weight)
            new_weights = initial_weight + self.lr *  delta_w
            self.feedforward.weight = nn.Parameter(new_weights, requires_grad=False)
            self.exponential_average = torch.add(self.gamma * self.exponential_average, (1 - self.gamma) * y)

    # Fully orthogonal Sanger variant
    def update_weights_FullyOrthogonal(self, input, output):
        with torch.no_grad():
            x = input.squeeze()
            y = output.squeeze()
            initial_weight = self.feedforward.weight.detach().clone()
            outer_prod = torch.outer(y, x)
            ytw = torch.matmul(y.unsqueeze(0), initial_weight)
            norm_term = torch.outer(y.squeeze(0), ytw.squeeze(0))
            delta_weight = self.lr * ((outer_prod - norm_term))
            new_weights = torch.add(initial_weight, delta_weight)
            self.feedforward.weight = nn.Parameter(new_weights, requires_grad=False)
            self.exponential_average = torch.add(self.gamma * self.exponential_average, (1 - self.gamma) * y)

    # Fully orthogonal Sanger variant
    def update_weights_FullyOrthogonal(self, input, output):
        with torch.no_grad():
            x=input.squeeze()
            y=output.squeeze()
            initial_weight = self.feedforward.weight.detach().clone()
            outer_prod = torch.outer(y, x)
            ytw = torch.matmul(y.unsqueeze(0), initial_weight)
            norm_term = torch.outer(y.squeeze(0), ytw.squeeze(0))
            delta_weight = self.lr*((outer_prod - norm_term))
            new_weights = torch.add(initial_weight, delta_weight)
            self.feedforward.weight=nn.Parameter(new_weights, requires_grad=False)
            self.exponential_average=torch.add(self.gamma*self.exponential_average,(1-self.gamma)*y)
    
    def update_weights_OrthogonalExclusive(self, input: torch.Tensor, output: torch.Tensor):
        with torch.no_grad():
            x: torch.Tensor = input.float().squeeze().to(self.device)
            x.requires_grad_(False)
            y: torch.Tensor = output.float().squeeze().to(self.device)
            y.requires_grad_(False)

            outer_prod: torch.Tensor = torch.einsum("i, j -> ij", y, x)

            weights: torch.Tensor = self.feedforward.weight.to(self.device)

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
        with torch.no_grad():
            output = output.squeeze()
            input = input.squeeze()
            true_output = true_output.squeeze()
            outer = torch.outer(torch.relu(true_output - output), input)
            self.feedforward.weight=nn.Parameter(torch.add(outer, self.feedforward.weight.detach().clone()), requires_grad=False)

    def classifier_update_supervised(self, input, output):
        output = output.detach().clone().squeeze()
        input = input.detach().clone().squeeze()
        outer = torch.outer(output, input)
        self.feedforward.weight=nn.Parameter(torch.add(outer, self.feedforward.weight.detach().clone()), requires_grad=False)
    
    def update_weights(self, input, preactivation, prediction,  true_output=None):
        if self.is_output_layer:
            if self.o_learning == ClassifierLearning.Contrastive:
                self.classifier_update_contrastive(input, prediction, true_output)
            elif self.o_learning == ClassifierLearning.Supervised:
                self.classifier_update_supervised(input, true_output)
            elif self.o_learning == ClassifierLearning.SoftHebb:
                self.update_weights_softhebb(input, preactivation, true_output)
        elif self.w_update == Learning.OrthogonalExclusive:
                self.update_weights_OrthogonalExclusive(input, prediction)
        elif self.w_update == Learning.SoftHebb:
            self.update_weights_softhebb(input, preactivation, prediction)
        else:
            raise ValueError("Not Implemented")

    def normalize_weights(self):
        new_weights = self.feedforward.weight.detach().clone()
        #print(new_weights.shape)
        #print(new_weights)
        new_weights_norm = torch.norm(new_weights, p=2, dim=1, keepdim=True)
        #print(new_weights_norm)
        #print(new_weights_norm.shape)
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

    def forward(self, x, clamped=None):
        with torch.no_grad():
            #self.normalize_weights()
            if len(x.shape) > 2:
                x = x.reshape(x.shape[0], -1)
            u = self.feedforward(x)
            y = self.inhibition(x)
            self.update_weights(x, u, y, true_output=clamped)
            return u, y
            
    def forward_test(self, x):
        x = x.reshape(-1).unsqueeze(0)
        x = self.feedforward(x)
        x = self.inhibition(x)
        return x
    
    def TD_forward(self, x, w, h_l):
        if h_l is None:
            u = self.feedforward(x)
        else :
            #print(self.feedforward(x))
            #print(torch.matmul(h_l, w))
            u = self.feedforward(x) + self.rho * torch.matmul(h_l, w)
        y = self.inhibition(u)
        return u, y
    


class ConvolutionHebbianLayer(nn.Module):
    def __init__(self, input_shape, kernel, stride, in_ch, out_ch, lambd, lr, gamma, epsilon, rho, device, padding=0,
                 weightlearning=Learning.OrthogonalExclusive, weightscaling=WeightScale.WeightDecay, triangle=False,
                 is_output_layer=False):
        super(ConvolutionHebbianLayer, self).__init__()
        self.input_shape = input_shape  # (height, width)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.in_channel = in_ch
        self.out_channel = out_ch
        self.lamb = lambd
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.rho = rho
        self.device = device
        self.triangle = triangle
        self.is_output_layer = is_output_layer

        self.w_learning = weightlearning
        self.w_scaling = weightscaling

        self.output_shape = (self._output_formula_(input_shape[0]), self._output_formula_(input_shape[1]))
        self.nb_tiles = self.output_shape[0] * self.output_shape[1]

        self.convolution = nn.Conv2d(stride=self.stride, kernel_size=self.kernel, padding=self.padding,
                                     in_channels=self.in_channel, out_channels=self.out_channel, bias=False)
        self.convolutionTD = None
        self.unfold = nn.Unfold(self.kernel, dilation=1, padding=self.padding, stride=self.stride)
        
        for param in self.convolution.parameters():
            param=torch.nn.init.uniform_(param, a=0, b=1)
            param.requires_grad_(False)

        self.exponential_average=torch.zeros(self.out_channel).to(device)
    @staticmethod
    def _generic_output_formula_(size, kernel, padding, dilation, stride):
        return int((size + 2 * padding - dilation * (kernel -1) - 1)/stride + 1)

    def _output_formula_(self, size):
        return self._generic_output_formula_(size, kernel=self.kernel, padding=self.padding, dilation=1,
                                             stride=self.stride)

    def Set_TD(self, inc, outc, kernel, stride):
        self.convolutionTD = nn.ConvTranspose2d(inc, outc, kernel, stride, padding=self.padding)

    def inhibition(self, output):
        
        # output : [batch=1, outchannel, newheight, newwidth]
        if self.triangle:
            m = torch.mean(output, dim=1, keepdim=True)
            output = output - m
        output = torch.relu(output)

        max_ele, _ = torch.max(output, dim=1, keepdim=True)

        output /= (max_ele + 1e-9)

        output=torch.pow(output, self.lamb)

        return output
    
    """
    Unfold the weights from the convolution module to correct dimensions

    Input : [output_channels, input_channels, kernel, kernel]
    Output : [output_channels, input_channels*kernel*kernel]

    --------------------

    With correct dimensions, we could reuse the weight update rules from MLP section

    """
    def unfold_weights(self):
        
        # [output_channels, input_channels, kernel, kernel]
        initial_weight = self.convolution.weight.detach().clone()

        # [output_channels, input_channels*kernel*kernel]
        flattened_weight = initial_weight.reshape(initial_weight.size(0), -1)

        return flattened_weight.to(self.device)
    

    """
    Fold the weights from the convolution module to correct dimensions

    Input : [output_channels, input_channels*kernel*kernel]
    Output : [output_channels, input_channels, kernel, kernel]

    --------------------

    We need to fold the weights back to initial dimensions to replace the convolution's weight parameter

    """
    def fold_weights(self, unfolded):
        
        reshaped = unfolded.reshape(self.out_channel, self.in_channel, self.kernel, self.kernel)

        return reshaped

    def update_weights_orthogonalexclusive(self, input: torch.Tensor, output: torch.Tensor):
        weights = self.unfold_weights()

        x: torch.Tensor = input.clone().detach().float().to(self.device)
        y: torch.Tensor = output.clone().detach().float().to(self.device)
        
        outer_prod: torch.Tensor = torch.einsum("ai, aj -> aij", y, x).to(self.device)

        W_norms: torch.Tensor = torch.norm(weights, dim=1, keepdim=False).to(self.device)

        scaling_terms: torch.Tensor = W_norms.reshape(weights.size(0),1) / (W_norms.reshape(1,weights.size(0)) + 10e-8)

        remove_diagonal: torch.Tensor = torch.ones(weights.size(0), weights.size(0)) - torch.eye(weights.size(0))
        remove_diagonal: torch.Tensor = remove_diagonal.reshape(weights.size(0), weights.size(0), 1).to(self.device)

        scaled_weight: torch.Tensor = scaling_terms.reshape(weights.size(0), weights.size(0), 1) * weights.reshape(1, weights.size(0), weights.size(1)).to(self.device)

        scaled_weight: torch.Tensor = scaled_weight * remove_diagonal

        norm_term: torch.Tensor = torch.einsum("ai, ak, ikj -> ij", y, y, scaled_weight)

        computed_rule: torch.Tensor = self.lr*(outer_prod - norm_term)

        avg_delta_weight = torch.mean(computed_rule, dim=0)

        new_weights = torch.add(self.unfold_weights(), avg_delta_weight)

        new_weights = self.fold_weights(new_weights)

        self.convolution.weight=nn.Parameter(torch.relu(new_weights), requires_grad=False)
        
        self.exponential_average=torch.add(self.gamma*self.exponential_average,(1-self.gamma)*torch.mean(output, dim=0))
    

    def update_weights_FullyOrthogonal(self, input, output):
        
        initial_weight = self.unfold_weights()

        outer_prod = torch.einsum('ij,ik->ijk', output, input)

        ytw = torch.matmul(output.unsqueeze(1), initial_weight).squeeze(1)

        norm_term = torch.einsum('ij,ik->ijk', output, ytw)

        delta_weight = self.lr*(outer_prod - norm_term)

        avg_delta_weight = torch.mean(delta_weight, dim=0)

        new_weights = torch.add(initial_weight, avg_delta_weight)

        new_weights = self.fold_weights(new_weights)

        self.convolution.weight=nn.Parameter(torch.relu(new_weights), requires_grad=False)
        
        self.exponential_average=torch.add(self.gamma*self.exponential_average,(1-self.gamma)*torch.mean(output, dim=0))

    def update_weight_softhebb(self, input, preactivation, output):
        with torch.no_grad():
            W = self.unfold_weights()
            x = self.unfold(input).permute(0, 2, 1)
            batch_dim, n_tiles, in_dim = x.shape
            x = x.reshape(batch_dim * n_tiles , in_dim)
            batch_dim, out_dim, out_h, out_w = output.shape
            assert out_h * out_w == n_tiles
            def flatten_out(z):
                o = z.permute(0, 2, 3, 1)
                o = o.reshape( batch_dim * out_h * out_w, out_dim)
                return o
            u = flatten_out(preactivation)
            y = flatten_out(output)
            delta_w = update_weight_softhebb(x, u, y, W)
            new_W = W + self.lr * delta_w
            new_W = self.fold_weights(new_W)
            self.convolution.weight = nn.Parameter(new_W, requires_grad=False)
            if self.convolutionTD:
                self.convolutionTD.weight = self.convolution.weight



    def normalize_weights(self):

        new_weights = self.unfold_weights()

        out_dim, in_dim = new_weights.shape
        new_weights_norm = torch.norm(new_weights, p=2, dim=1, keepdim=True) / np.sqrt(in_dim)
        new_weights = new_weights / (new_weights_norm + 1e-8)

        new_weights = self.fold_weights(new_weights)

        self.convolution.weight=nn.Parameter(new_weights, requires_grad=False)

    def weight_decay(self):
        average=torch.mean(self.exponential_average).item()
        A=self.exponential_average/average

        growth_factor_positive=self.epsilon*nn.Tanh()(-self.epsilon*(A-1))+1
        growth_factor_negative=torch.reciprocal(growth_factor_positive)

        positive_weights=torch.where(self.convolution.weight>0, self.convolution.weight, 0.0)
        negative_weights=torch.where(self.convolution.weight<0, self.convolution.weight, 0.0)
        
        positive_weights=positive_weights*(growth_factor_positive.reshape(self.out_channel, 1,1,1))
        negative_weights=negative_weights*(growth_factor_negative.reshape(self.out_channel, 1,1,1))
        
        self.convolution.weight=nn.Parameter(torch.add(positive_weights, negative_weights), requires_grad=False)

    
    def get_flattened_input(self, x):
        convolved_input = self.unfold(x)
        # convolved_input : [nb_tiles, input_channels, kernel, kernel]
        convolved_input = convolved_input.reshape(1, self.in_channel, self.nb_tiles, self.kernel, self.kernel).permute(0,2,1,3,4).squeeze(0)

        # flattened_input is [nb_tiles, input_channel*kernel*kernel]
        flattened_input = convolved_input.reshape(self.nb_tiles, -1)

        return flattened_input

    def get_flattened_output(self, output):

        # flattened_output is [nb_tiles, output_channel]
        flattened_output = output.squeeze(0).permute(1,2,0).reshape(-1, self.out_channel)

        return flattened_output


    def update_weights(self, input, preactivation, prediction,  true_output=None):
        if self.w_learning == Learning.OrthogonalExclusive:
            self.update_weights_orthogonalexclusive(self.get_flattened_input(input), self.get_flattened_output(prediction))
        elif self.w_learning == Learning.SoftHebb:
            y = prediction if true_output is None else prediction
            self.update_weight_softhebb(input, preactivation, y)
        else:
            raise ValueError("Not Implemented")

        
        if self.w_scaling == WeightScale.WeightDecay:
            self.weight_decay()
        elif self.w_scaling == WeightScale.WeightNormalization:
            self.normalize_weights()
        else:
            pass

    # x is torch.Size([batch_size=1, channel, height, width])
    def forward(self, x, clamped=None):
        
        # output is [batch=1, output_channel, new_height, new_width]
        u = self.convolution(x)

        output = self.inhibition(u)

        self.update_weights(x, u, output,  true_output=clamped)
        
        return u, output


    def forward_test(self, x):
        output = self.convolution(x)
        output = self.inhibition(output)
        return output
    

    def TD_forward(self, x, h_l, w=None):
        if h_l == None:
            output = self.forward_test(x)
        elif w == None:
            td_component = self.convolutionTD.forward(h_l)
            ff_component = self.convolution.forward(x)
            output = self.rho*td_component+ff_component
            #print(self.rho*td_component)
            #print(ff_component)
            output = self.inhibition(output)
        else :
            ff_component = self.convolution.forward(x)
            td_component = torch.matmul(h_l, w)
            td_component = td_component.reshape_as(ff_component)
            #print(self.rho*td_component)
            #print(ff_component)
            output = self.rho*td_component+ff_component
            output = self.inhibition(output)
        return output
    
    def visualize_filters(self, w1):
        fig, axes = plt.subplots(w1.size(1), w1.size(0), figsize=(20, 15))

        if w1.size(1) == 1:
            for i2 in range(w1.size(0)):
                img = w1[i2].squeeze(0)
                axes[i2].imshow(img, interpolation='nearest')
        elif w1.size(0) == 1:
            for i2 in range(w1.size(1)):
                img = w1[0][i2]
                axes[i2].imshow(img, interpolation='nearest')
        else:
            for i in range(w1.size(1)):
                for i2 in range(w1.size(0)):
                    img = w1[i2][i]
                    axes[i][i2].imshow(img, interpolation='nearest')
        plt.show()
        
    def visualize_weights(self):
        self.visualize_filters(self.convolution.weight)


def CNNBaseline_Model(inputshape, kernels, channels, strides=None, padding=None, lambd=1, lr=0.005,
                      gamma=0.99, epsilon=0.01, rho=1e-3, nbclasses=10, topdown=True, device=None,
                      learningrule=Learning.OrthogonalExclusive,
                      weightscaling=WeightScale.WeightNormalization, outputlayerrule=ClassifierLearning.Contrastive,
                      triangle=False):
    if padding is None:
        padding = [0] * len(kernels)
    if strides is None:
        strides = [1] * len(kernels)

    mycnn = ConvolutionalNeuralNet(device)
    input_channels = inputshape[0]
    input_size = inputshape[1:]

    layer_id = 1
    for kernel, out_chan, stride, pad in zip(kernels, channels, strides, padding):
        convlayer = ConvolutionHebbianLayer(input_size, kernel, stride, input_channels, out_chan, lambd, lr, gamma,
                                            epsilon, rho, device=device, padding=pad, weightlearning=learningrule,
                                            weightscaling=weightscaling, triangle=triangle)
        if topdown:
            convlayer.Set_TD(inc=out_chan, outc=input_channels, kernel=kernel, stride=stride)
        mycnn.add_layer(f"CNNLayer{layer_id}", convlayer)
        layer_id += 1
        input_channels = out_chan
        input_size = convlayer.output_shape
        print(f"New Image Dimensions after Convolution : {((out_chan,) + input_size)}")

    fc_inputdim = convlayer.nb_tiles * out_chan
    print("Fully connected layer input dim : " + str(fc_inputdim))
    
    heblayer = Hebbian_Layer(inputdim=fc_inputdim, outputdim=nbclasses, lr=lr, lamb=lambd, rho=rho,
                             gamma=gamma, eps=epsilon, device=device, is_output_layer=True,
                             output_learning=outputlayerrule, triangle=triangle)

    #heblayer2 = Hebbian_Layer(inputdim=hsize, outputdim=nbclasses, lr=lr, lamb=lambd, w_decrease=10, gamma=gamma, )

    mycnn.add_layer("HebLayer", heblayer)

    return mycnn



def Save_Model(mymodel, dataset, rank, topdown, v_input, device, acc):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if topdown:
        foldername = os.getcwd() + '/SavedModels_3/CNN_TD_' + dataset + '_' + str(rank) + '_' + str(acc) + '_' + timestr
    else:
        foldername = os.getcwd() + '/SavedModels_3/CNN_FF_' + dataset + '_' + str(rank) + '_' + str(acc) + '_' + timestr

    if not os.path.exists(foldername):
        os.mkdir(foldername)

    if not os.path.isfile(foldername + '/model'):
        torch.save(mymodel.state_dict(), foldername + '/model')

    view_filters(mymodel, foldername)
    if topdown:
        view_TD_weights(mymodel, foldername, v_input, device)
    else:
        view_ff_weights(mymodel, foldername, v_input, device)



def CNNBaseline_Experiment(epoch, mymodel, dataloader, dataset, nclasses, imgtype):

    mymodel.train()
    cn = 0
    for _ in range(epoch):
        for data in tqdm(dataloader):
            inputs, labels=data
            #print(inputs)
            if imgtype == ImageType.Gray:
                inputs = inputs.reshape(1,1,28,28)
            mymodel.forward(inputs, oneHotEncode(labels, nclasses, mymodel.device).unsqueeze(0))
            #cn += 1
            #if cn == 10:
            #    return mymodel
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    foldername = os.getcwd() + '/SavedModels/CNN_FF_' + dataset + '_' + timestr
    os.mkdir(foldername)

    torch.save(mymodel.state_dict(), foldername + '/model')

    #view_filters(mymodel, foldername)
    #view_ff_weights(mymodel, foldername, dataloader)
    return mymodel

def CNNTD_Experiment(epoch, mymodel, dataloader, dataset, nclasses, imgtype):
    mymodel.train()
    cn = 0
    for _ in range(epoch):
        for data in tqdm(dataloader):
            inputs, labels=data
            #print(inputs)
            if imgtype == ImageType.Gray:
                inputs = inputs.reshape(1,1,28,28)
            mymodel.TD_forward(inputs, oneHotEncode(labels, nclasses, mymodel.device))
            #return mymodel
            #cn += 1
            #if cn == 10:
            #    return mymodel
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    foldername = os.getcwd() + '/SavedModels/MLP_TD_' + dataset + '_' + timestr
    os.mkdir(foldername)

    torch.save(mymodel.state_dict(), foldername + '/model')

    view_filters(mymodel, foldername)
    view_TD_weights(mymodel, foldername, dataloader)
    return mymodel



def CNN_Baseline_test(mymodel, data_loader, imgtype, topdown):
    yes = 0
    tot = 0
    da = {}
    for data in tqdm(data_loader):
        inputs, labels = data
        if imgtype == ImageType.Gray:
            inputs = inputs.reshape(1,1,28,28)
        if topdown:
            output = torch.argmax(mymodel.TD_forward_test(inputs))
        else:
            output = torch.argmax(mymodel.forward_test(inputs))
        #return mymodel
        if labels.item() not in da:
            da[labels.item()] = (0, 0, 0)
        if output.item() not in da:
            da[output.item()] = (0,0,0)
        r, t, w = da[labels.item()]
        if output.item() == labels.item():
            yes += 1
            da[labels.item()] = (r+1, t+1, w)
        else :
            r1, t1, w1 = da[output.item()]
            da[output.item()] = (r1, t1, w1+1) 
            da[labels.item()] = (r, t+1, w)
        tot += 1
    return (yes/tot), da

def view_filters(v_model, folder):
    w1 = v_model.layers['CNNLayer1'].convolution.weight.detach().to('cpu')
    w2 = v_model.layers['CNNLayer2'].convolution.weight.detach().to('cpu')
    w3 = v_model.layers['CNNLayer3'].convolution.weight.detach().to('cpu')
    visualize_filters(w1, folder + '/Conv1.png')
    visualize_filters(w2, folder + '/Conv2.png')
    visualize_filters(w3, folder + '/Conv3.png')
    view_classifier(v_model, folder + '/Classifier.png')

def view_classifier(v_model, file):
    nb = int(math.ceil(math.sqrt(10)))

    nb_ele = v_model.layers['HebLayer'].feedforward.weight.size(0)
    fig, axes = plt.subplots(nb, nb, figsize=(32,32))
        
    weight = v_model.layers['HebLayer'].feedforward.weight.detach().to('cpu')
    sq = int(math.ceil(math.sqrt(weight.size(1))))
    weight = F.pad(weight, (0, sq**2 - weight.size(1)))

    for ele in range(nb_ele):
        random_feature_selector = weight[ele]
        heatmap = random_feature_selector.view(int(math.sqrt(weight.size(1))),
                                                int(math.sqrt(weight.size(1))))
        ax = axes[ele // nb, ele % nb]
        im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
        fig.colorbar(im, ax=ax)
        ax.set_title(f'Weight {ele}')

    plt.tight_layout()
    plt.savefig(file)

def visualize_filters(w1, file):
    fig, axes = plt.subplots(w1.size(1), w1.size(0), figsize=(20, 15))

    if w1.size(1) == 1:
        for i2 in range(w1.size(0)):
            img = w1[i2].squeeze(0)
            axes[i2].imshow(img, interpolation='nearest')
    elif w1.size(0) == 1:
        for i2 in range(w1.size(1)):
            img = w1[0][i2]
            axes[i2].imshow(img, interpolation='nearest')
    else:
        for i in range(w1.size(1)):
            for i2 in range(w1.size(0)):
                img = w1[i2][i]
                axes[i][i2].imshow(img, interpolation='nearest')
    plt.savefig(file)



def view_ff_weights(v_model, folder, v_input, device):
    inputs, data = v_input[0], v_input[1]
    inputs = inputs.reshape(1,1,28,28).to(device)
    fmaps = v_model.layers['CNNLayer1'].forward_test(inputs)
    fmaps2 = v_model.layers['CNNLayer2'].forward_test(fmaps)
    fmaps3 = v_model.layers['CNNLayer3'].forward_test(fmaps2)

    view_map(fmaps, folder + '/Fmap1.png')
    view_map(fmaps2, folder + '/Fmap2.png')
    view_map(fmaps3, folder + '/Fmap3.png')


def view_TD_weights(self, folder, v_input, device):
    inputs, data = v_input[0], v_input[1]
    inputs = inputs.reshape(1,1,28,28).to(device)
    layers = list(self.layers.values())
    nb_layers = len(layers)
    hlist = [None]*nb_layers
    for _ in range(self.iteration):
        x = inputs
        for idx in range(nb_layers-2):
            h_l = hlist[idx+1]
            x = layers[idx].TD_forward(x, h_l)
            hlist[idx] = x
        x_l = layers[-2].TD_forward(hlist[-3], hlist[-1], layers[-1].feedforward.weight)
        hlist[-2] = x_l
        x_L = layers[-1].forward_test(hlist[-2])
        hlist[-1] = x_L

    for idx in range(len(hlist)-1):
        view_map(hlist[idx], folder + '/Fmap' + str(idx+1) + '.png')
    

def view_map(fmaps, file):
    fmaps = fmaps.detach().to('cpu')
    if fmaps.size(1) == 1:
        plt.imshow(fmaps[0][0], interpolation='nearest')
        plt.colorbar()
        plt.savefig(file)
    else:
        fig, axes = plt.subplots(2, fmaps.size(1)//2, figsize=(20,5))
        for i in range(fmaps.size(1)//2):
            img = fmaps[0][i]
            axes[0, i].imshow(img, interpolation='nearest')
        for i in range(fmaps.size(1)//2):
            img = fmaps[0][i+fmaps.size(1)//2]
            axes[1, i].imshow(img, interpolation='nearest')
        plt.savefig(file)
