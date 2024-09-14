import os 
import torch
import torch.nn as nn
import math
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import models.learning as L
from models.hyperparams import ImageType, LearningRule, WeightScale, Inhibition, oneHotEncode





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
    def forward(self, x, clamped):
        with torch.no_grad():
            for name, layer in self.layers.items():
                true_value = clamped if layer.is_output_layer else None
                u, x = layer(x, clamped=true_value)
                assert u.isnan().any() == False, f"u has NaNs in layer {name}"
                assert (u == 0.0).all() == False, f"u has only 0s in layer {name}"
                assert x.isnan().any() == False, f"output has NaNs in layer {name}"
                assert (x == 0.0).all() == False, (f"output has only 0s in layer {name}\n"
                                                   f"u min: {torch.min(u)}\n"
                                                   f"u max: {torch.max(u)}\n"
                                                   f"W min: {torch.min(layer.unfold_weights())}\n"
                                                   f"W max: {torch.max(layer.unfold_weights())}")
            return x
    
    def forward_test(self, x):
        for layer in self.layers.values():
            _, x = layer.forward(x, update_weights=False)
        return x


    """
    Used in topdown models
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
    
    def update_weights(self, inp, hlist, ulist, prediction):
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
        
    def TD_forward_test(self, x):
        return self.TD_forward(x, update_weights=False)



class Hebbian_Classifier(nn.Module):
    def __init__(self, inputdim, outputdim, lr, lamb, device, b=1, triangle=False, output_learning=ClassifierLearning.Contrastive, inhibition=Inhibition.RePU):
        super(Hebbian_Classifier, self).__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.lr = lr 
        self.lamb = lamb
        self.device = device
        self.output_learning = output_learning
        self.triangle = triangle
        self.inhib = inhibition
        self.feedforward = nn.Linear(self.inputdim, self.outputdim, bias=False)
        self.is_output_layer = True
        for param in self.feedforward.parameters():
            param=torch.nn.init.uniform_(param, a=0.0, b=b)
            param.requires_grad_(False)
    
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
            self.feedforward.weight=nn.Parameter(torch.relu(self.lr * outer + self.feedforward.weight),
                                                 requires_grad=False)

    def classifier_update_supervised(self, input, output):
        output = output.squeeze()
        input = input.squeeze()
        outer = torch.outer(output, input)
        self.feedforward.weight=nn.Parameter(self.lr * outer + self.feedforward.weight,
                                             requires_grad=False)
        
    def update_weights_softhebb(self, input, preactivation, output):
        with torch.no_grad():
            y = output.squeeze()
            initial_weight = self.feedforward.weight
            delta_w = L.update_weight_softhebb(input, preactivation, output, initial_weight)
            new_weights = initial_weight + self.lr *  delta_w
            self.feedforward.weight = nn.Parameter(new_weights, requires_grad=False)

    def update_weights(self, x, u, y, true_output):
        if self.output_learning == ClassifierLearning.SoftHebb:
            self.update_weights_softhebb(x, u, true_output)
        elif self.output_learning == ClassifierLearning.Contrastive:
            self.classifier_update_contrastive(x, y, true_output)
        elif self.output_learning == ClassifierLearning.Supervised:
            self.classifier_update_supervised(x, true_output)

    def forward(self, x, clamped=None, update_weights=True):
        x = x.reshape(x.shape[0], -1)
        u = self.feedforward(x)
        y = self.inhibition(u)
        if update_weights:
            self.update_weights(x, u, y, true_output=clamped)
        return u, y 

    def TD_forward(self, x, h):
        x = x.reshape(x.shape[0], -1)
        u = self.feedforward(x)
        y = self.inhibition(u)
        return u, y

        

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
    


class ConvolutionHebbianLayer(nn.Module):
    def __init__(self, input_shape, kernel, stride, in_ch, out_ch, lambd, lr, rho, device,
                 eta=1, padding=0,
                 weightlearning=LearningRule.SoftHebb, weightscaling=WeightScale.WeightNormalization, triangle=False,
                 is_output_layer=False, whiten_input=False, b=1, inhibition = Inhibition.RePU):
        super(ConvolutionHebbianLayer, self).__init__()
        self.input_shape = input_shape  # (height, width)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.in_channel = in_ch
        self.out_channel = out_ch
        self.lamb = lambd
        self.lr = lr
        self.rho = rho
        self.eta = eta
        self.b=b
        self.device = device
        self.triangle = triangle
        self.whiten_input = whiten_input
        self.is_output_layer = False
        self.w_learning = weightlearning
        self.w_scaling = weightscaling
        self.inhib = inhibition

    
        self.output_shape = None
        self.nb_tiles = None

        self.convolution = nn.Conv2d(stride=self.stride, kernel_size=self.kernel, padding=self.padding,
                                     in_channels=self.in_channel, out_channels=self.out_channel, bias=False)
        self.batchnorm = nn.BatchNorm2d(self.in_channel, affine=False)
        self.pool = None
        # Top down weights
        self.convolutionTD = None # if layer above is conv
        self.top_down_factor = None # if layer above is conv
        self.W_TD = None # if layer above is FC


        self.unfold = nn.Unfold(self.kernel, dilation=1, padding=self.padding, stride=self.stride)
        self.unfold_pool = None

        self.fold = nn.Fold(output_size=input_shape, kernel_size=self.kernel, padding=self.padding, stride=self.stride)

        self.fold_unfold_divisor = self.fold(self.unfold(torch.ones(1, in_ch, input_shape[0], input_shape[1])))
        for param in self.convolution.parameters():
            param=torch.nn.init.uniform_(param, a=0, b=self.b)
            param.requires_grad_(False)

        self.exponential_average=torch.zeros(self.out_channel).to(device)


    @staticmethod
    def _generic_output_formula_(size, kernel, padding, dilation, stride):
        return int((size + 2 * padding - dilation * (kernel -1) - 1)/stride + 1)

    def _output_formula_(self, size):
        o = self._generic_output_formula_(size, kernel=self.kernel, padding=self.padding, dilation=1,
                                             stride=self.stride)
        if self.pool:
            o = self._generic_output_formula_(o, self.pool.kernel_size, self.pool.padding, 1, self.pool.stride)
        return o

    def compute_output_shape(self):
        self.output_shape = (self._output_formula_(self.input_shape[0]), self._output_formula_(self.input_shape[1])) 
        self.nb_tiles = self.output_shape[0] * self.output_shape[1]

    def Set_TD(self, inc, outc, kernel, stride):
        self.convolutionTD = nn.ConvTranspose2d(inc, outc, kernel, stride, padding=self.padding)

    def set_TD_factor(self, factor):
        self.top_down_factor = factor

    def set_pooling(self, kernel, stride, padding, type):
        if type == "Max":
            self.pool = nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding)
        else :
            self.pool = nn.AvgPool2d(kernel_size=kernel, stride=stride, padding=padding)
        self.unfold_pool = nn.Unfold(self.pool.kernel_size, 1, self.pool.padding, self.pool.stride)

    def inhibition(self, output):
        
        # output : [batch=1, outchannel, newheight, newwidth]
        if self.triangle:
            m = torch.mean(output, dim=1, keepdim=True)
            output = output - m

        output = torch.relu(output)

        max_ele, _ = torch.max(output, dim=1, keepdim=True)

        output /= (max_ele + 1e-9)

        if self.inhib == Inhibition.RePU:
        if self.inhib == Inhibition.RePU:
            output=torch.pow(output, self.lamb)
        elif self.inhib == Inhibition.Softmax:
            output = nn.Softmax()(output)

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
        initial_weight = self.convolution.weight

        # [output_channels, input_channels*kernel*kernel]
        flattened_weight = initial_weight.reshape(initial_weight.size(0), -1)

        return flattened_weight
    

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


    def top_down_weights(self):
        with torch.no_grad():
            W = self.unfold_weights()
            Winv = torch.linalg.pinv(W)
            W_TD = self.fold_weights(Winv.T)
        return W_TD
    

    def update_weights_orthogonalexclusive(self, input: torch.Tensor, output: torch.Tensor):

        weights = self.unfold_weights()

        x: torch.Tensor = input
        y: torch.Tensor = output
        
        delta_w = L.update_weights_OrthogonalExclusive(x, y, weights, self.device, self.eta)

        computed_rule: torch.Tensor = self.lr*delta_w

        avg_delta_weight = torch.mean(computed_rule, dim=0)

        new_weights = torch.add(self.unfold_weights(), avg_delta_weight)

        new_weights = self.fold_weights(new_weights)

        self.convolution.weight=nn.Parameter(torch.relu(new_weights), requires_grad=False)


    def update_weights_FullyOrthogonal(self, input, output):
        
        initial_weight = self.unfold_weights()

        delta_weight = L.update_weights_FullyOrthogonal(input, output, initial_weight, self.eta)

        delta_weight = self.lr*delta_weight

        avg_delta_weight = torch.mean(delta_weight, dim=0)

        new_weights = torch.add(initial_weight, avg_delta_weight)

        new_weights = self.fold_weights(new_weights)

        self.convolution.weight=nn.Parameter(torch.relu(new_weights), requires_grad=False)


    def update_weight_softhebb(self, input, preactivation, output):
        with torch.no_grad():
            W = self.unfold_weights()
            x = self.unfold(input)
            x = x.permute(0, 2, 1)
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
            delta_w = L.update_weight_softhebb(x,u, y, W, inhibition=self.inhib)
            new_W = W + self.lr * delta_w
            new_W = self.fold_weights(new_W)
            self.convolution.weight = nn.Parameter(new_W, requires_grad=False)


    def normalize_weights(self):

        new_weights = self.unfold_weights()

        new_weights_norm = torch.norm(new_weights, p=2, dim=1, keepdim=True) #/ np.sqrt(in_dim)
        new_weights = new_weights / (new_weights_norm + 1e-8)

        new_weights = self.fold_weights(new_weights)

        self.convolution.weight=nn.Parameter(new_weights, requires_grad=False)

    def whiten_image(self, input_image):
        batch, channel, h, w = input_image.shape
        tiles = self.unfold(input_image)
        std_tile, mean_tile = torch.std_mean(tiles, dim=-1, keepdim=True)
        whiten_tiles = (tiles - mean_tile) / (std_tile + 1e-8)
        whiten_image = self.fold(whiten_tiles) / (self.fold_unfold_divisor + 1e-8)
        assert whiten_image.isnan().any() == False, (f"NaNs in whitend input of layer {self._get_name()}\n"
                                                   f"Original image has NaNs: {input_image.isnan.any()}")
        if (whiten_image > 10000).any():
            raise Warning(f"Large values in whiten_image of layer {self._get_name()}\n"
                          f"Max value: {torch.max(whiten_image)}"
                          f"Max value of input: {torch.max(input_image)}")
        return whiten_image


    def get_flattened_input(self, x):
        convolved_input = self.unfold(self.unfold_pool(x))
        # convolved_input : [nb_tiles, input_channels, kernel, kernel]
        convolved_input = convolved_input.reshape(1, self.in_channel, self.nb_tiles, self.kernel,
                                                  self.kernel).permute(0,2,1,3,4).squeeze(0)

        # flattened_input is [nb_tiles, input_channel*kernel*kernel]
        flattened_input = convolved_input.reshape(self.nb_tiles, -1)

        return flattened_input

    def get_flattened_output(self, output):

        # flattened_output is [nb_tiles, output_channel]
        flattened_output = output.squeeze(0).permute(1,2,0).reshape(-1, self.out_channel)

        return flattened_output


    def update_weights(self, input, preactivation, prediction,  true_output=None):
        if self.w_learning == LearningRule.OrthogonalExclusive:
            self.update_weights_orthogonalexclusive(self.get_flattened_input(input), self.get_flattened_output(prediction))
        elif self.w_learning == LearningRule.FullyOrthogonal:
            self.update_weights_FullyOrthogonal(self.get_flattened_input(input),
                                                    self.get_flattened_output(prediction))
        elif self.w_learning == LearningRule.SoftHebb:
            y = prediction if true_output is None else prediction
            self.update_weight_softhebb(input, preactivation, y)
        else:
            raise ValueError("Not Implemented")

        if self.w_scaling == WeightScale.WeightNormalization:
            self.normalize_weights()

    # x is torch.Size([batch_size=1, channel, height, width])
    def forward(self, x, clamped=None, update_weights=True):

        if self.whiten_input:
            x = self.whiten_image(x)
        
        x = self.batchnorm(x)
        # output is [batch=1, output_channel, new_height, new_width]
        u = self.convolution(x)

        output = self.inhibition(u)

        if update_weights:
            self.update_weights(x, u, output,  true_output=clamped)
        
        if self.pool:
            output = self.pool(output)
        
        return u, output
    
    
    def top_down_fct(self, h_l):
        if self.convolutionTD:
            raw_td = self.convolutionTD(h_l, output_size=self.output_shape)
            return raw_td / (self.top_down_factor + 1e-9)
        else:
            h = torch.matmul(h_l, self.W_TD)
            return h.reshape(h.shape[0], self.out_channel, *self.output_shape)

    def TD_forward(self, x, h_l):
        if h_l == None:
            u = self.convolution.forward(x)
        else:
            td_component = self.top_down_fct(h_l)
            ff_component = self.convolution.forward(x)
            u = self.rho*td_component+ff_component
        y = self.inhibition(u)
        return u, y
    
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


def CNN_Model_from_config(inputshape, config, learningrule, weightscaling, outputlearning, inhibition, device, nbclasses):
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
        convlayer = ConvolutionHebbianLayer(input_shape=inputsize, kernel=layerconfig['kernel'], stride=layerconfig['stride'], in_ch=input_channel, out_ch=layerconfig['out_channel'], 
                                            lambd=lamb, lr=lr, rho=rho, device=device, eta=eta, padding=layerconfig['padding'], weightlearning=learningrule, weightscaling=weightscaling, triangle=layerconfig['triangle'], 
                                            whiten_input=layerconfig['whiten'], b=beta, inhibition=inhibition)
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
    print("Fully connected layer input dim : " + str(fc_inputdim))

    if config['Classifier']['Hebbian'] == True:
        heblayer = Hebbian_Classifier(fc_inputdim, outputdim=nbclasses, lr=lr, lamb=lamb, device=device, b=beta, triangle=config['Classifier']['triangle'], output_learning=outputlearning, inhibition=inhibition)

        mycnn.add_layer("HebLayer", heblayer)

    if is_topdown:
        mycnn.initialize_TD_factors()
    
    mycnn = mycnn.to(device)

    return mycnn


def CNNBaseline_Model(inputshape, kernels, channels, strides=None, padding=None, lambd=1, lr=0.005,
                      gamma=0.99, epsilon=0.01, rho=1e-3, eta=1e-3, b=1e-4,
                      nbclasses=10, topdown=True, device=None,
                      learningrule=LearningRule.OrthogonalExclusive,
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



def CNN_Experiment(epoch, mymodel, dataloader, testloader, dataset, nclasses, imgtype, device, traintopdown=False, testtopdown=False):

    mymodel.train()
    samples = 0
    for _ in range(epoch):
        for data in tqdm(dataloader):
            # if samples in [0, 10, 100, 1000, 10000, 20000, 30000]:
            #     print(CNN_Baseline_test(mymodel, testloader, imgtype, testtopdown))
            inputs, labels=data
            # inputs = inputs.to('mps')
            # labels = labels.to('mps')
            #print(inputs)
            inputs = inputs.to(device)
            labels = labels.to(device)
            if imgtype == ImageType.Gray:
                inputs = inputs.reshape(1,1,28,28)
            if traintopdown:
                mymodel.TD_forward(inputs, clamped=oneHotEncode(labels, nclasses, mymodel.device))
            else:
                mymodel.forward(inputs, oneHotEncode(labels, nclasses, mymodel.device))
            samples += 1
            #if cn == 10:
            #    return mymodel
    
    # timestr = time.strftime("%Y%m%d-%H%M%S")

    # if traintopdown:
    #     foldername = os.getcwd() + '/SavedModels/CNN_TD_' + dataset + '_' + timestr
    # else:
    #     foldername = os.getcwd() + '/SavedModels/CNN_FF_' + dataset + '_' + timestr

    # os.mkdir(foldername)

    # torch.save(mymodel.state_dict(), foldername + '/model')

    #view_filters(mymodel, foldername)
    #view_ff_weights(mymodel, foldername, dataloader)
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
            y = mymodel.forward_test(inputs)
            output = torch.argmax(y)
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
    if da[1][0] == 0:
        print(f"last y: {y}")
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
