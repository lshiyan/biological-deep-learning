import os 
import torch
import torch.nn as nn
import math
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import models.learning as L
from models.hyperparams import (ImageType, LearningRule, WeightScale, Inhibition, oneHotEncode,
                                InputProcessing, cnn_output_formula_2D)
from models.CNN.layers import ConvolutionHebbianLayer, ConvSoftHebbLayer, PoolingLayer, GradientClassifierLayer


class ConvolutionalNeuralNet(nn.Module):

    def __init__(self, device):
        super(ConvolutionalNeuralNet, self).__init__()
        self.layers = nn.ModuleDict()
        self.iteration = 3
        self.device = device
        

    def add_layer(self, name, layer):
        self.layers[name] = layer
    
    def set_iteration(self, i):
        self.iteration = i

    """
    Used in feedforward models
    """

    def train_conv(self, x, label=None): 
        self.train()
        with torch.no_grad():
            for layer in self.layers.values():
                if isinstance(layer, PoolingLayer):
                    x = layer(x) 
                elif isinstance(layer, ConvSoftHebbLayer):
                    target = label if layer.is_output_layer else None
                    x = layer(x, target=target)
                else:
                    break
            return x
        
    def test_conv(self, x):
        self.eval()
        for layer in self.layers.values():
            if isinstance(layer, GradientClassifierLayer):
                break
            x = layer.forward(x)
        return x


    def train_classifier(self, x, label=None):
        with torch.no_grad():
            y = self.test_conv(x)
            y = y.reshape(y.shape[0], -1)
        for layer in self.layers.values():
            if isinstance(layer, GradientClassifierLayer):
                pred = layer.forward(y, label)
        return pred
    

    def forward(self, x, label=None):
        with torch.no_grad():
            y = self.train_conv(x, label)
            y = y.reshape(y.shape[0], -1)
        for layer in self.layers.values():
            if isinstance(layer, GradientClassifierLayer):
                pred = layer.forward(y, label)
        return pred

    def forward_test(self, x):
        self.eval()
        with torch.no_grad():
            y = self.test_conv(x)
            y = y.reshape(y.shape[0], -1)
        for layer in self.layers.values():
            if isinstance(layer, GradientClassifierLayer):
                pred = layer.forward(y)
        return pred
    
    
    def set_conv_training_layers(self, layers_to_train):
        for layer in self.layers.values():
            if layer in layers_to_train:
                layer.train()
            else:
                layer.eval()



    """
    Used in topdown models
    """
    """
    def TD_inference(self, input, clamped_output=None):
        with torch.no_grad():
            layers = list(self.layers.values())
            nb_layers = len(layers)
            hlist = [None] * nb_layers
            ulist = [None] * nb_layers
            hlist[-1] = clamped_output
            for _ in range(self.iteration):
                x = input
                for idx in range(nb_layers-1):
                    h_l = hlist[idx+1]
                    u, x = layers[idx].TD_forward(x, h_l)
                    hlist[idx] = x.clone()
                    ulist[idx] = u.clone()
                lastu, prediction = layers[-1].TD_forward(hlist[-2], None)
                if clamped_output is None:
                    hlist[-1] = prediction.clone()
                ulist[-1] = lastu.clone()
            return ulist, hlist, prediction

    def initialize_TD_weights(self):
        layers = list(self.layers.values())
        nb_layers = len(layers)
        for i in range(nb_layers-2):
            layers[i].convolutionTD.weight = nn.Parameter(layers[i+1].top_down_weights(), requires_grad=False)
        layers[-2].W_TD = layers[-1].top_down_weights()

    def initialize_TD_factors(self):
        layers = list(self.layers.values())
        nb_layers = len(layers)
        for i in range(nb_layers - 2):
            layers[i].set_TD_factor(layers[i+1].fold_unfold_divisor)

    def TD_forward(self, x, clamped=None, update_weights=True):
        with torch.no_grad():
            self.initialize_TD_weights()
            ulist, hlist, prediction = self.TD_inference(x, clamped_output=clamped)
            if update_weights:
                self.update_weights(x, hlist, ulist, prediction)
            return prediction
    
    def set_training_layers(self, inp, hlist, ulist, prediction):
        layers = list(self.layers.values())
        nb_layers = len(layers)
        hlist = [inp] + hlist
        for idx in range(nb_layers-1):
            i = hlist[idx]
            u = ulist[idx]
            o = hlist[idx+1]
            layers[idx].update_weights(i, u, o, None)
        i = hlist[-2].reshape(hlist[-2].shape[0], -1)
        layers[-1].update_weights(i, ulist[-1], prediction, hlist[-1])
        #layers[-1].weight_decay()

    """
        
    def TD_forward_test(self, x):
        return self.TD_forward(x, update_weights=False)
    
    



class Hebbian_Classifier(nn.Module):
    def __init__(self, inputdim, outputdim, device, basemodel, lr):
        super(Hebbian_Classifier, self).__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.device = device
        self.basemodel = basemodel
        self.linear = nn.Linear(inputdim, outputdim, bias=False)
        self.lr = lr
        self.lamb = 1
        self.triangle = False
        self.inhib = Inhibition.Softmax ##
        self.output_learning = LearningRule.OutputContrastiveSupervised

    def inhibition(self, x):
        if self.triangle:
            m = torch.mean(x)
            x = x - m
        if self.inhib == Inhibition.RePU:
            x=torch.relu(x)
            max_ele=torch.max(x).item()
            x/=(max_ele + 1e-9)
            x=torch.pow(x, self.lamb)
        elif self.inhib == Inhibition.Softmax:
            x = nn.Softmax()(x)
        return x
    
    def classifier_update_contrastive(self, input, output, true_output):
        with torch.no_grad():
            output = output.squeeze()
            input = input.squeeze()
            true_output = true_output.squeeze()
            outer = torch.outer(true_output - output, input)
            self.linear.weight=nn.Parameter(torch.relu(self.lr * outer + self.linear.weight),
                                                 requires_grad=False)

    def classifier_update_supervised(self, input, output):
        output = output.squeeze()
        input = input.squeeze()
        outer = torch.outer(output, input)
        self.linear.weight=nn.Parameter(self.lr * outer + self.linear.weight,
                                             requires_grad=False)
        
    def update_weights_softhebb(self, input, preactivation, output):
        with torch.no_grad():
            initial_weight = self.linear.weight
            delta_w = L.update_weight_softhebb(input, preactivation, output, initial_weight)
            new_weights = initial_weight + self.lr *  delta_w
            self.linear.weight = nn.Parameter(new_weights, requires_grad=False)

    def update_weights(self, x, u, y, true_output):
        if self.output_learning == LearningRule.SoftHebb:
            self.update_weights_softhebb(x, u, true_output)
        elif self.output_learning == LearningRule.OutputContrastiveSupervised:
            self.classifier_update_contrastive(x, y, true_output)
        elif self.output_learning == LearningRule.Supervised:
            self.classifier_update_supervised(x, true_output)

    def train_conv(self, x, label):
        return self.basemodel(x, label)

    def train_classifier(self, x, label):
        self.basemodel.eval()
        with torch.no_grad():
            y = self.basemodel.forward_test(x)
            y = y.reshape(y.shape[0], -1)
        u = self.linear(y)
        #pred = self.inhibition(u)
        self.update_weights(y, u, u, true_output=label)
        return u, y 
    
    # def forward(self, x, label):
    #     self.basemodel.eval()
    #     with torch.no_grad():
    #         y = self.basemodel(x, label)
    #         y = y.reshape(y.shape[0], -1)
    #     self.optim.zero_grad()
    #     pred = self.linear(self.drop(y))
    #     loss = self.lossfn(pred, label)
    #     loss.backward()
    #     self.optim.step()
    #     return pred
    
    def forward_test(self, x):
        self.basemodel.eval()
        self.eval()
        with torch.no_grad():
            y = self.basemodel.forward_test(x)
            y = y.reshape(y.shape[0], -1)
        pred = self.linear(y)
        #pred = self.inhibition(pred)
        return pred


class Gradient_Classifier(nn.Module):
    def __init__(self, inputdim, outputdim, device, basemodel, lr):
        super(Gradient_Classifier, self).__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.device = device
        self.basemodel = basemodel
        self.linear = nn.Linear(inputdim, outputdim)
        self.drop = nn.Dropout(0.5)

        self.optim = torch.optim.Adam(self.linear.parameters(), lr=lr)
        self.lossfn = nn.CrossEntropyLoss()

    
    def train_conv(self, x, label):
        with torch.no_grad():
            self.basemodel(x, label)

    def train_classifier(self, x, label):
        self.basemodel.eval()
        with torch.no_grad():
            y = self.basemodel.forward_test(x)
            y = y.reshape(y.shape[0], -1)
        self.optim.zero_grad()
        pred = self.linear(self.drop(y))
        loss = self.lossfn(pred, label)
        loss.backward()
        self.optim.step()
        return pred
    
    def forward(self, x, label):
        self.basemodel.train()
        with torch.no_grad():
            y = self.basemodel(x, label)
            y = y.reshape(y.shape[0], -1)
        self.optim.zero_grad()
        pred = self.linear(self.drop(y))
        loss = self.lossfn(pred, label)
        loss.backward()
        self.optim.step()
        return pred
    
    def forward_test(self, x):
        self.basemodel.eval()
        self.eval()
        with torch.no_grad():
            y = self.basemodel.forward_test(x)
            y = y.reshape(y.shape[0], -1)
        pred = self.linear(self.drop(y))
        return pred
    
    def set_training_layers(self, layers_to_train):
        for layer in self.basemodel.layers.values():
            if layer in layers_to_train:
                layer.train()
            else:
                layer.eval()
        

# class Hebbian_Classifier(nn.Module):
#     def __init__(self, inputdim, outputdim, lr, lamb, device, b=1, triangle=False, output_learning=LearningRule.SoftHebb, inhibition=Inhibition.RePU):
#         super(Hebbian_Classifier, self).__init__()
#         self.inputdim = inputdim
#         self.outputdim = outputdim
#         self.lr = lr 
#         self.lamb = lamb
#         self.device = device
#         self.output_learning = output_learning
#         self.triangle = triangle
#         self.inhib = inhibition
#         self.feedforward = nn.Linear(self.inputdim, self.outputdim, bias=False)
#         self.is_output_layer = True
#         for param in self.feedforward.parameters():
#             param=torch.nn.init.uniform_(param, a=0.0, b=b)
#             param.requires_grad_(False)
    
#     def inhibition(self, x):
#         if self.triangle:
#             m = torch.mean(x)
#             x = x - m
#         if self.inhib == Inhibition.RePU:
#             x=torch.relu(x)
#             max_ele=torch.max(x).item()
#             x/=(max_ele + 1e-9)
#             x=torch.pow(x, self.lamb)
#         elif self.inhib == Inhibition.Softmax:
#             x = nn.Softmax()(x)
#         return x
    
#     def classifier_update_contrastive(self, input, output, true_output):
#         with torch.no_grad():
#             output = output.squeeze()
#             input = input.squeeze()
#             true_output = true_output.squeeze()
#             outer = torch.outer(true_output - output, input)
#             self.feedforward.weight=nn.Parameter(torch.relu(self.lr * outer + self.feedforward.weight),
#                                                  requires_grad=False)

#     def classifier_update_supervised(self, input, output):
#         output = output.squeeze()
#         input = input.squeeze()
#         outer = torch.outer(output, input)
#         self.feedforward.weight=nn.Parameter(self.lr * outer + self.feedforward.weight,
#                                              requires_grad=False)
        
#     def update_weights_softhebb(self, input, preactivation, output):
#         with torch.no_grad():
#             y = output.squeeze()
#             initial_weight = self.feedforward.weight
#             delta_w = L.update_weight_softhebb(input, preactivation, output, initial_weight)
#             new_weights = initial_weight + self.lr *  delta_w
#             self.feedforward.weight = nn.Parameter(new_weights, requires_grad=False)

#     def update_weights(self, x, u, y, true_output):
#         if self.output_learning == LearningRule.SoftHebb:
#             self.update_weights_softhebb(x, u, true_output)
#         elif self.output_learning == LearningRule.OutputContrastiveSupervised:
#             self.classifier_update_contrastive(x, y, true_output)
#         elif self.output_learning == LearningRule.Supervised:
#             self.classifier_update_supervised(x, true_output)

#     def forward(self, x, clamped=None, update_weights=True):
#         x = x.reshape(x.shape[0], -1)
#         u = self.feedforward(x)
#         y = self.inhibition(u)
#         if update_weights:
#             self.update_weights(x, u, y, true_output=clamped)
#         return u, y 

#     def TD_forward(self, x, h):
#         x = x.reshape(x.shape[0], -1)
#         u = self.feedforward(x)
#         y = self.inhibition(u)
#         return u, y

        

# class Hebbian_Layer(nn.Module):
#     def __init__(self, inputdim, outputdim, lr, lamb, rho, gamma, eps, device, eta=1, b=1, is_output_layer=False,
#                  output_learning=ClassifierLearning.Supervised, update=Learning.OrthogonalExclusive,
#                  weightscaling=WeightScale.WeightNormalization, triangle=False, inhibition = Inhibition.RePU):
#         super(Hebbian_Layer, self).__init__()
#         self.input_dim = inputdim
#         self.output_dim = outputdim
#         self.lr = lr
#         self.lamb = lamb
#         self.rho = rho
#         self.b = b
#         self.eta = eta
#         self.gamma = gamma
#         self.epsilon = eps
#         self.is_output_layer = is_output_layer
#         self.device = device
#         self.triangle = triangle

#         self.feedforward = nn.Linear(self.input_dim, self.output_dim, bias=False)
#         self.W_TD = None

#         for param in self.feedforward.parameters():
#             param=torch.nn.init.uniform_(param, a=0.0, b=self.b)
#             param.requires_grad_(False)
        
#         self.exponential_average=torch.zeros(self.output_dim).to(self.device)
#         self.o_learning = output_learning
#         self.w_update = update
#         self.weightscaling = weightscaling
#         self.inhib = inhibition

#     def unfold_weights(self):
#         return self.feedforward.weight

#     def top_down_weights(self):
#         with torch.no_grad():
#             return torch.linalg.pinv(self.unfold_weights()).T

#     def inhibition(self, x):
#         if self.triangle:
#             m = torch.mean(x)
#             x = x - m
#         x=torch.relu(x)
#         max_ele=torch.max(x).item()
#         x/=(max_ele + 1e-9)
#         x=torch.pow(x, self.lamb)
#         return x
    

#     #SoftHebb
#     def update_weights_softhebb(self, input, preactivation, output):
#         with torch.no_grad():
#             y = output.squeeze()
#             initial_weight = self.feedforward.weight
#             delta_w = L.update_weight_softhebb(input, preactivation, output, initial_weight)
#             new_weights = initial_weight + self.lr *  delta_w
#             self.feedforward.weight = nn.Parameter(new_weights, requires_grad=False)
#             self.exponential_average = torch.add(self.gamma * self.exponential_average, (1 - self.gamma) * y)

#     # Fully orthogonal
#     def update_weights_FullyOrthogonal(self, input, output):
#         with torch.no_grad():
#             y = output.squeeze()
#             initial_weight = self.feedforward.weight
#             delta_w = L.update_weights_FullyOrthogonal(input, y, initial_weight)
#             new_weights = initial_weight + self.lr *  delta_w
#             self.feedforward.weight = nn.Parameter(new_weights, requires_grad=False)
#             self.exponential_average = torch.add(self.gamma * self.exponential_average, (1 - self.gamma) * y)
    
#     # Fully Orthogonal Exclusive 
#     def update_weights_OrthogonalExclusive(self, input: torch.Tensor, output: torch.Tensor):
#         with torch.no_grad():
#             y: torch.Tensor = output.squeeze()
#             weights: torch.Tensor = self.feedforward.weight
#             delta_w = L.update_weights_OrthogonalExclusive(input, y, weights, self.device)
#             new_weights = weights + self.lr *  delta_w
#             self.feedforward.weight = nn.Parameter(new_weights, requires_grad=False)
#             self.exponential_average=torch.add(self.gamma*self.exponential_average,(1-self.gamma)*y)

#     def classifier_update_contrastive(self, input, output, true_output):
#         with torch.no_grad():
#             output = output.squeeze()
#             input = input.squeeze()
#             true_output = true_output.squeeze()
#             outer = torch.outer(true_output - output, input)
#             self.feedforward.weight=nn.Parameter(torch.relu(self.lr * outer + self.feedforward.weight),
#                                                  requires_grad=False)

#     def classifier_update_supervised(self, input, output):
#         output = output.squeeze()
#         input = input.squeeze()
#         outer = torch.outer(output, input)
#         self.feedforward.weight=nn.Parameter(self.lr * outer + self.feedforward.weight,
#                                              requires_grad=False)
    
#     def update_weights(self, input, preactivation, prediction,  true_output=None):
#         if self.is_output_layer:

#             if self.o_learning == ClassifierLearning.Contrastive:
#                 self.classifier_update_contrastive(input, prediction, true_output)
#             elif self.o_learning == ClassifierLearning.Supervised:
#                 self.classifier_update_supervised(input, true_output)
#             elif self.o_learning == ClassifierLearning.SoftHebb:
#                 self.update_weights_softhebb(input, preactivation, true_output)
#             elif self.o_learning == ClassifierLearning.Orthogonal:
#                 self.update_weights_FullyOrthogonal(input, true_output)

#         elif self.w_update == Learning.OrthogonalExclusive:
#                 self.update_weights_OrthogonalExclusive(input, prediction)
#         elif self.w_update == Learning.SoftHebb:
#             self.update_weights_softhebb(input, preactivation, prediction)
#         elif self.w_update == Learning.FullyOrthogonal:
#             self.update_weights_FullyOrthogonal(input, prediction)
#         else:
#             raise ValueError("Not Implemented")
        
#         if self.weightscaling == WeightScale.WeightNormalization:
#             self.normalize_weights()
#         elif self.weightscaling == WeightScale.WeightDecay:
#             self.weight_decay()
#         elif self.weightscaling == WeightScale.No:
#             pass
#         else:
#             raise ValueError("Not Implemented")


#     def normalize_weights(self):
#         weights = self.feedforward.weight
#         new_weights_norm = torch.norm(weights, p=2, dim=1, keepdim=True)
#         new_weights = weights / (new_weights_norm + 1e-9)
#         self.feedforward.weight=nn.Parameter(new_weights, requires_grad=False)
    
#     def weight_decay(self):
#         average=torch.mean(self.exponential_average).item()
#         A=self.exponential_average/average
#         growth_factor_positive=self.epsilon*nn.Tanh()(-self.epsilon*(A-1))+1
#         growth_factor_negative=torch.reciprocal(growth_factor_positive)
#         positive_weights=torch.where(self.feedforward.weight>0, self.feedforward.weight, 0.0)
#         negative_weights=torch.where(self.feedforward.weight<0, self.feedforward.weight, 0.0)
#         positive_weights=positive_weights*growth_factor_positive.unsqueeze(1)
#         negative_weights=negative_weights*growth_factor_negative.unsqueeze(1)
#         self.feedforward.weight=nn.Parameter(torch.add(positive_weights, negative_weights), requires_grad=False)


#     def forward(self, x, clamped=None, update_weights=True):
#         with torch.no_grad():
#             x = x.reshape(x.shape[0], -1)
#             u = self.feedforward(x)
#             y = self.inhibition(u)
#             if update_weights:
#                 self.update_weights(x, u, y, true_output=clamped)
#             return u, y
            
#     def forward_test(self, x):
#         x = x.reshape(x.shape[0], -1)
#         x = self.feedforward(x)
#         x = self.inhibition(x)
#         return x
    

#     def TD_forward(self, x, h_l):
#         if len(x.shape) > 2:
#             x = x.reshape(x.shape[0], -1)
#         if h_l is None:
#             u = self.feedforward(x)
#         else :
#             u = self.feedforward(x) + self.rho * torch.matmul(h_l, self.W_TD)
#         y = self.inhibition(u)
#         return u, y
    

def CNN_Model_from_config(inputshape, config, device, nbclasses):
    mycnn = ConvolutionalNeuralNet(device)
    lamb = config['Lambda']
    lr = config['Lr']
    beta = config['beta']
    rho = config['Rho']
    eta = config['Eta']
    is_topdown = config['Topdown']
    input_channel = inputshape[0]
    nb_conv = len(config['Convolutions'])
    l_keys = list(config['Convolutions'].keys())
    inputsize = inputshape[1:]
    is_pool = config['PoolingBlock']['Pooling']
    for layer_idx in range(len(config['Convolutions'])):
        layerconfig = config['Convolutions'][l_keys[layer_idx]]
        inhibition = Inhibition.RePU if layerconfig['inhibition'] == "REPU" else Inhibition.Softmax
        convlayer = ConvolutionHebbianLayer(input_shape=inputsize, kernel=layerconfig['kernel'], stride=layerconfig['stride'], in_ch=input_channel, out_ch=layerconfig['out_channel'], 
                                            lambd=lamb, lr=lr, rho=rho, device=device, eta=eta, padding=layerconfig['padding'], paddingmode = layerconfig['paddingmode'], triangle=layerconfig['triangle'], 
                                            whiten_input=layerconfig['whiten'], b=beta, inhibition=inhibition)
        if layerconfig['batchnorm']:
            convlayer.batchnorm = nn.BatchNorm2d(input_channel, affine=False)
        if is_pool:
            poolconfig = config["PoolingBlock"][l_keys[layer_idx]]
            convlayer.set_pooling(kernel=poolconfig['kernel'], stride=poolconfig['stride'], padding=poolconfig['padding'], type=poolconfig['Type'])
        convlayer.compute_output_shape()
        if is_topdown and layer_idx < (nb_conv-1):
            convlayer.Set_TD(inc=config['Convolutions'][l_keys[layer_idx+1]]['out_channel'], outc=layerconfig['out_channel'], 
                             kernel=config['Convolutions'][l_keys[layer_idx+1]]['kernel'], stride=config['Convolutions'][l_keys[layer_idx+1]]['stride'])
        convlayer.normalize_weights()
        mycnn.add_layer(f"CNNLayer{layer_idx+1}", convlayer)
        input_channel = layerconfig['out_channel']
        inputsize = convlayer.output_shape
        print(f"New Image Dimensions after Convolution : {((layerconfig['out_channel'],) + inputsize)}")

    fc_inputdim = convlayer.nb_tiles * config['Convolutions'][l_keys[-1]]['out_channel']
    #print("Fully connected layer input dim : " + str(fc_inputdim))

    mymodel = Gradient_Classifier(fc_inputdim, nbclasses, device, mycnn, 0.001)
    
    mymodel = mymodel.to(device)

    return mymodel



def new_CNN_Model_from_config(inputshape, config, device, nbclasses):
    mycnn = ConvolutionalNeuralNet(device)
    lamb = config['Lambda']
    lr = config['classifierLr']
    w_lr = config['w_lr']
    b_lr = config['b_lr']
    l_lr = config['l_lr']
    w_norm = config['w_norm']
    is_topdown = config['Topdown']
    input_channel = inputshape[0]
    nb_conv = len(config['Convolutions']['Layers'])
    l_keys = list(config['Convolutions']['Layers'].keys())
    inputsize = inputshape[1:] # (height, width)
    is_pool = config['PoolingBlock']["GlobalParams"]['Pooling']
    conv_stride = config['Convolutions']["GlobalParams"]['stride']
    pool_stride = config['PoolingBlock']["GlobalParams"]['stride']
    conv_padding = config['Convolutions']["GlobalParams"]['padding']
    padding_mode = config['Convolutions']["GlobalParams"]['paddingmode']
    triangle = config['Convolutions']["GlobalParams"]['triangle']
    preprocessing = InputProcessing.Whiten if config['Convolutions']["GlobalParams"]['whiten'] else InputProcessing.No 
    inhibition = Inhibition.RePU if config['Convolutions']["GlobalParams"]['inhibition'] == "REPU" else Inhibition.Softmax

    for layer_idx in range(nb_conv):
        layerconfig = config['Convolutions']["Layers"][l_keys[layer_idx]]

        convlayer = ConvSoftHebbLayer(input_shape=inputsize, kernel=layerconfig['kernel'], in_ch=input_channel, out_ch=layerconfig['out_channel'], stride=conv_stride, 
                                            padding=conv_padding, w_lr=w_lr, b_lr=b_lr, l_lr=l_lr, device=device, is_output_layer=False, initial_weight_norm=w_norm, 
                                            triangle=triangle, initial_lambda=lamb, inhibition=inhibition, learningrule=LearningRule.SoftHebb, preprocessing=preprocessing)
        
        mycnn.add_layer(f"CNNLayer{layer_idx+1}", convlayer)

        input_channel = layerconfig['out_channel']
        inputsize = convlayer.output_shape # update inputsize for next layer
        
        if is_pool:
            poolconfig = config["PoolingBlock"]["Layers"][l_keys[layer_idx]]
            poollayer = PoolingLayer(kernel=poolconfig['kernel'], stride=pool_stride, padding=poolconfig['padding'], pool_type=poolconfig['Type'])
            
            mycnn.add_layer(f"PoolLayer{layer_idx+1}", poollayer)
            
            inputsize = cnn_output_formula_2D(inputsize, poolconfig['kernel'], poolconfig['padding'], 1, pool_stride)
            

        output_shape = inputsize
        n_tiles = output_shape[0] * output_shape[1]

        if is_topdown and layer_idx < (nb_conv-1):
            convlayer.Set_TD(inc=config['Convolutions']["Layers"][l_keys[layer_idx+1]]['out_channel'], outc=layerconfig['out_channel'], 
                             kernel=config['Convolutions']["Layers"][l_keys[layer_idx+1]]['kernel'], stride=conv_stride)
        

    fc_inputdim = n_tiles * config['Convolutions']["Layers"][l_keys[-1]]['out_channel']
    #print("Fully connected layer input dim : " + str(fc_inputdim))
    
    classifier = GradientClassifierLayer(fc_inputdim, nbclasses, lr)
    mycnn.add_layer(f"ClassifierLayer", classifier)

    mycnn = mycnn.to(device)

    return mycnn



def CNNBaseline_Model(inputshape, kernels, channels, strides=None, padding=None, lambd=1, lr=0.005,
                      gamma=0.99, epsilon=0.01, rho=1e-3, eta=1e-3, b=1e-4,
                      nbclasses=10, topdown=True, device=None,
                      learningrule=LearningRule.Orthogonal,
                      weightscaling=WeightScale.WeightNormalization, outputlayerrule=LearningRule.OutputContrastiveSupervised,
                      triangle=False, whiten_input=False, inhibition=Inhibition.RePU):
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
                                            epsilon, rho, eta=eta, device=device, padding=pad, weightlearning=learningrule,
                                            weightscaling=weightscaling, triangle=triangle, whiten_input=whiten_input, b=b, inhibition=inhibition)
        if topdown and layer_id < len(kernels):
            convlayer.Set_TD(inc=channels[layer_id], outc=out_chan, kernel=kernels[layer_id], stride=strides[layer_id])
        convlayer.normalize_weights()
        mycnn.add_layer(f"CNNLayer{layer_id}", convlayer)
        layer_id += 1
        input_channels = out_chan
        input_size = convlayer.output_shape
        print(f"New Image Dimensions after Convolution : {((out_chan,) + input_size)}")


    fc_inputdim = convlayer.nb_tiles * out_chan
    print("Fully connected layer input dim : " + str(fc_inputdim))
    
    heblayer = Hebbian_Layer(inputdim=fc_inputdim, outputdim=nbclasses, lr=lr, lamb=lambd, rho=rho, eta=eta,
                             gamma=gamma, eps=epsilon, device=device, is_output_layer=True,
                             output_learning=outputlayerrule, weightscaling=weightscaling, triangle=triangle, b=b, inhibition=inhibition)

    mycnn.add_layer("HebLayer", heblayer)

    if topdown:
        mycnn.initialize_TD_factors()
    
    mycnn = mycnn.to(device)

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
