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



class ConvolutionalNeuralNet(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNet, self).__init__()
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
            x = layer(x, clamped)
        return x
    
    def forward_test(self, x):
        for layer in self.layers.values():
            x = layer.forward_test(x)
        return x

    def TD_inference(self, x, clamped):
        input = x.detach().clone()
        layers = list(self.layers.values())
        nb_layers = len(layers)
        hlist = [None]*nb_layers
        hlist[-1] = clamped
        for _ in range(self.iteration):
            x = input
            for idx in range(nb_layers-2):
                h_l = hlist[idx+1]
                x = layers[idx].TD_forward(x, h_l)
                hlist[idx] = x
            x_l = layers[-2].TD_forward(hlist[-3], hlist[-1], layers[-1].feedforward.weight)
            hlist[-2] = x_l
            x_ = layers[-1].forward_test(hlist[-2])
        return hlist, x_

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
    
    def update_weights(self, inp, hlist, x_L):
        layers = list(self.layers.values())
        nb_layers = len(layers)
        hlist = [inp] + hlist
        for idx in range(nb_layers-1):
            i = hlist[idx]
            o = hlist[idx+1]
            layers[idx].update_weights(i, o)
        i = hlist[-2].squeeze().squeeze().reshape(-1)
        layers[-1].update_weights(i, hlist[-1], x_L)
        #layers[-1].weight_decay()
        
    def TD_forward_test(self, x):
        input = x.detach().clone()
        layers = list(self.layers.values())
        nb_layers = len(layers)
        hlist = [None]*nb_layers
        for _ in range(self.iteration):
            x = input
            for idx in range(nb_layers-2):
                h_l = hlist[idx+1]
                x = layers[idx].TD_forward(x, h_l)
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
    
    def visualize_featuremaps(self):
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
        self.device = device

        self.feedforward = nn.Linear(self.input_dim, self.output_dim, bias=False)

        for param in self.feedforward.parameters():
            param=torch.nn.init.uniform_(param, a=0.0, b=1.0)
            param.requires_grad_(False)
        
        self.exponential_average=torch.zeros(self.output_dim).to(self.device)
        self.o_learning = output_learning
        self.w_update = update
        self.weight_mod = weight


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
        delta_weight = self.lr*((outer_prod - norm_term))
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
    
    def update_weights(self, input, output, pred):
        if self.is_output_layer:
            if self.o_learning == ClassifierLearning.Contrastive:
                self.classifier_update_contrastive(input, pred, output)
            elif self.o_learning == ClassifierLearning.Supervised:
                self.classifier_update_supervised(input, pred)
            else:
                self.update_weights_OrthogonalExclusive(input, pred)

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
        else:
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
            h = self.feedforward(x) + self.decrease*torch.matmul(h_l, w)
        return self.inhibition(h)
    


class ConvolutionHebbianLayer(nn.Module):
    def __init__(self, input_size, kernel, stride, in_ch, out_ch, lambd, lr, gamma, epsilon, rho, device, weightlearning=Learning.FullyOrthogonal, weightscaling=WeightScale.WeightDecay):
        super(ConvolutionHebbianLayer, self).__init__()
        self.in_size = input_size
        self.kernel = kernel
        self.stride = stride
        self.in_channel = in_ch
        self.out_channel = out_ch
        self.lamb = lambd
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.rho = rho
        self.device = device

        self.w_learning = weightlearning
        self.w_scaling = weightscaling

        self.nb_tiles = int(((self.in_size-self.kernel)/self.stride+1)**2)

        self.convolution = nn.Conv2d(stride=self.stride, kernel_size=self.kernel, in_channels=self.in_channel, out_channels=self.out_channel, bias=False)
        
        for param in self.convolution.parameters():
            param=torch.nn.init.uniform_(param, a=0, b=1)
            param.requires_grad_(False)

        self.exponential_average=torch.zeros(self.out_channel).to(device)
    
    def Set_TD(self, inc, outc, kernel, stride):
        self.convolutionTD = nn.ConvTranspose2d(inc, outc, kernel, stride)

    def inhibition(self, output):
        
        # output : [batch=1, outchannel, newheight, newwidth]
        output = nn.ReLU()(output)

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

        self.convolution.weight=nn.Parameter(nn.ReLU()(new_weights), requires_grad=False)
        
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

        self.convolution.weight=nn.Parameter(nn.ReLU()(new_weights), requires_grad=False)
        
        self.exponential_average=torch.add(self.gamma*self.exponential_average,(1-self.gamma)*torch.mean(output, dim=0))

    
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

        convolved_input = x.unfold(2, self.kernel, self.stride)
        
        convolved_input = convolved_input.unfold(3, self.kernel, self.stride)
        
        # convolved_input : [nb_tiles, input_channels, kernel, kernel]
        convolved_input = convolved_input.reshape(1, self.in_channel, self.nb_tiles, self.kernel, self.kernel).permute(0,2,1,3,4).squeeze(0)

        # flattened_input is [nb_tiles, input_channel*kernel*kernel]
        flattened_input = convolved_input.reshape(self.nb_tiles, -1)

        return flattened_input

    def get_flattened_output(self, output):

        # flattened_output is [nb_tiles, output_channel]
        flattened_output = output.squeeze(0).permute(1,2,0).reshape(-1, self.out_channel)

        return flattened_output


    def update_weights(self, inp, out):
        if self.w_learning == Learning.FullyOrthogonal:
            self.update_weights_FullyOrthogonal(self.get_flattened_input(inp), self.get_flattened_output(out))
        else:
            self.update_weights_orthogonalexclusive(self.get_flattened_input(inp), self.get_flattened_output(out))
        
        if self.w_scaling == WeightScale.WeightDecay:
            self.weight_decay()
        else:
            self.normalize_weights()

    # x is torch.Size([batch_size=1, channel, height, width])
    def forward(self, x, clamped):
        
        # output is [batch=1, output_channel, new_height, new_width]
        output = self.convolution(x)

        output = self.inhibition(output)

        self.update_weights(x, output)
        
        return output


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


def CNNBaseline_Model(inputsize, kernel, stride, inchannel, outchannel, lambd, lr, gamma, epsilon, rho, nbclasses, topdown, device, wl, ws, o):

    mycnn = ConvolutionalNeuralNet()
    convlayer1 = ConvolutionHebbianLayer(input_size=inputsize, kernel=kernel[0], stride=stride[0], in_ch=inchannel[0], out_ch=outchannel[0], lambd=lambd, lr=lr, gamma=gamma, epsilon=epsilon, rho=rho, device=device, weightlearning=wl, weightscaling=ws)
    hinputsize = (inputsize-kernel[0])/stride[0]+1
    print("New Image Dimensions after Convolution : " + str(int(hinputsize)))
    if topdown:
        convlayer1.Set_TD(inc=outchannel[1], outc=inchannel[1], kernel=kernel[1], stride=stride[1])

    convlayer2 = ConvolutionHebbianLayer(input_size=hinputsize, kernel=kernel[1], stride=stride[1], in_ch=inchannel[1], out_ch=outchannel[1], lambd=lambd, lr=lr, gamma=gamma, epsilon=epsilon, rho=rho, device=device, weightlearning=wl, weightscaling=ws)
    hinputsize2 = (hinputsize-kernel[1])/stride[1]+1
    print("New Image Dimensions after Convolution : " + str(int(hinputsize2)))
    if topdown:
        convlayer2.Set_TD(inc=outchannel[2], outc=inchannel[2], kernel=kernel[2], stride=stride[2])

    convlayer3 = ConvolutionHebbianLayer(input_size=hinputsize2, kernel=kernel[2], stride=stride[2], in_ch=inchannel[2], out_ch=outchannel[2], lambd=lambd, lr=lr, gamma=gamma, epsilon=epsilon, rho=rho, device=device, weightlearning=wl, weightscaling=ws)
    fc_inputdim = int(((hinputsize2-kernel[2])/stride[2]+1)**2)*outchannel[2]
    print("New Image Dimensions after Convolution : " + str(fc_inputdim))
    
    heblayer = Hebbian_Layer(inputdim=fc_inputdim, outputdim=nbclasses, lr=lr, lamb=lambd, w_decrease=10, gamma=gamma, eps=epsilon, device=device, is_output_layer=True, output_learning=o)

    #heblayer2 = Hebbian_Layer(inputdim=hsize, outputdim=nbclasses, lr=lr, lamb=lambd, w_decrease=10, gamma=gamma, )

    mycnn.add_layer("CNNLayer1", convlayer1)
    mycnn.add_layer("CNNLayer2", convlayer2)
    mycnn.add_layer("CNNLayer3", convlayer3)
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
            mymodel.forward(inputs, oneHotEncode(labels, nclasses))
            #cn += 1
            #if cn == 10:
            #    return mymodel
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    foldername = os.getcwd() + '/SavedModels/MLP_FF_' + dataset + '_' + timestr
    os.mkdir(foldername)

    torch.save(mymodel.state_dict(), foldername + '/model')

    view_filters(mymodel, foldername)
    view_ff_weights(mymodel, foldername, dataloader)
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
            mymodel.TD_forward(inputs, oneHotEncode(labels, nclasses))
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
