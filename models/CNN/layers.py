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
from models.MLP.model import SoftHebbLayer


class ConvSoftHebbLayer(nn.Module):
    def __init__(self, input_shape, kernel, in_ch, out_ch, stride = 1, padding = 0, w_lr: float = 0.3, b_lr: float = 0.003, 
                 l_lr: float = 0.1, device = None, is_output_layer = False, initial_weight_norm: float = 0.01,
                 triangle: bool = False, initial_lambda: float = 4.0, inhibition: Inhibition = Inhibition.RePU,
                 learningrule: LearningRule = LearningRule.SoftHebb, preprocessing: InputProcessing = InputProcessing.No,
                 antihebb_factor=1):

        super(ConvSoftHebbLayer, self).__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.is_output_layer = False
        self.input_shape = input_shape  # (height, width)
        self.kernel = kernel
        self.in_channel = in_ch
        self.out_channel = out_ch
        self.padding = padding
        self.stride = stride
        self.unfold = nn.Unfold(self.kernel, dilation=1, padding=self.padding, stride=self.stride)
        self.fold = nn.Fold(output_size=input_shape, kernel_size=self.kernel, padding=self.padding, stride=self.stride)
        self.fold_unfold_divisor = (self.fold(self.unfold(torch.ones(1, in_ch, input_shape[0], input_shape[1])))).to(device)
        self.base_soft_hebb_layer = SoftHebbLayer(inputdim = in_ch * kernel ** 2 , outputdim = out_ch, w_lr= w_lr,
                                                  b_lr = b_lr, l_lr = l_lr, device=device, is_output_layer=is_output_layer, 
                                                  initial_weight_norm = initial_weight_norm, triangle= triangle,
                                                  initial_lambda = initial_lambda, inhibition = inhibition,learningrule = learningrule,
                                                  preprocessing= preprocessing, anti_hebb_factor=antihebb_factor)
        self.output_shape = cnn_output_formula_2D(input_shape, kernel, padding, 1, stride)
        self.output_tiles = self.output_shape[0] * self.output_shape[1]

    
    def forward(self, x, target=None):
        unfolded_x = self.unfold(x)
        batch_dim, in_dim, nb_tiles = unfolded_x.shape
        unfolded_x = unfolded_x.permute(0, 2, 1)
        mlp_x = unfolded_x.reshape(batch_dim * nb_tiles, in_dim)
        mlp_y = self.base_soft_hebb_layer(mlp_x, target)
        y = mlp_y.reshape(batch_dim, self.output_shape[0], self.output_shape[1], self.out_channel)
        y = y.permute(0, 3, 1, 2)
        return y



class PoolingLayer(nn.Module):
    def __init__(self, kernel, stride, padding, pool_type):
        super(PoolingLayer, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.pool_type = pool_type
    
        if pool_type == "Max":
            self.pool = nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding)
        else :
            self.pool = nn.AvgPool2d(kernel_size=kernel, stride=stride, padding=padding)

    def forward(self, x):
        return self.pool(x)
    


class GradientClassifierLayer(nn.Module):
    def __init__(self, input_shape, output_shape, lr):
        super(GradientClassifierLayer, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.lr = lr
        self.linear = nn.Linear(input_shape, output_shape)
        self.drop = nn.Dropout(0.5)
        
        self.optim = torch.optim.Adam(self.linear.parameters(), lr=self.lr)
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        if label is not None: #train
            self.optim.zero_grad()
            pred = self.linear(self.drop(x))        
            loss = self.lossfn(pred, label)
            loss.backward()
            self.optim.step()
        else: #test
            self.eval
            pred = self.linear(self.drop(x))
        return pred


class ConvolutionHebbianLayer(nn.Module):
    def __init__(self, input_shape, kernel, stride, in_ch, out_ch, lambd, lr, rho, device,
                 eta=1, padding=0, paddingmode="zeros", triangle=False, whiten_input=False, 
                 b=1, inhibition = Inhibition.RePU):
        super(ConvolutionHebbianLayer, self).__init__()
        self.input_shape = input_shape  # (height, width)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.paddingmode = paddingmode
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
        #self.is_output_layer = False
        self.w_learning = LearningRule.SoftHebb
        self.w_scaling = WeightScale.No
        self.inhib = inhibition

    
        self.output_shape = None
        self.nb_tiles = None

        self.convolution = nn.Conv2d(stride=self.stride, kernel_size=self.kernel, padding=self.padding, padding_mode=self.paddingmode,
                                     in_channels=self.in_channel, out_channels=self.out_channel, bias=False)
        self.batchnorm = None
        self.pool = None
        # Top down weights
        self.convolutionTD = None # if layer above is conv
        self.top_down_factor = None # if layer above is conv
        self.W_TD = None # if layer above is FC


        self.unfold = nn.Unfold(self.kernel, dilation=1, padding=self.padding, stride=self.stride)
        self.unfold_pool = None

        self.fold = nn.Fold(output_size=input_shape, kernel_size=self.kernel, padding=self.padding, stride=self.stride)

        self.fold_unfold_divisor = (self.fold(self.unfold(torch.ones(1, in_ch, input_shape[0], input_shape[1])))).to(device)
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
        self.convolutionTD = nn.ConvTranspose2d(inc, outc, kernel, stride, padding=self.padding, padding_mode=self.paddingmode)

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
        # if self.triangle:
        #     m = torch.mean(output, dim=1, keepdim=True)
        #     output = output - m

        if self.inhib == Inhibition.RePU:
            output = torch.relu(output)
            max_ele, _ = torch.max(output, dim=1, keepdim=True)
            output /= (max_ele + 1e-9)
            output=torch.pow(output, self.lamb)
            output = output / (output.sum(dim=1, keepdim=True) + 1e-9)
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
        if self.w_learning == LearningRule.Orthogonal:
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
        with torch.no_grad():
            if self.whiten_input:
                x = self.whiten_image(x)
            
            if self.batchnorm: 
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
