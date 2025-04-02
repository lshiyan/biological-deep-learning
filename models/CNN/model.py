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
from models.CNN.layers import ConvolutionHebbianLayer, ConvSoftHebbLayer, PoolingLayer, GradientClassifierLayer, HebbianClassifierLayer


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
            if isinstance(layer, GradientClassifierLayer) or isinstance(layer, HebbianClassifierLayer):
                break
            x = layer.forward(x)
        return x


    def train_classifier(self, x, label=None):
        with torch.no_grad():
            y = self.test_conv(x)
            y = y.reshape(y.shape[0], -1)
        for layer in self.layers.values():
            if isinstance(layer, GradientClassifierLayer) or isinstance(layer, HebbianClassifierLayer):
                pred = layer.forward(y, label)
        return pred
    

    def forward(self, x, label=None):
        with torch.no_grad():
            y = self.train_conv(x, label)
            y = y.reshape(y.shape[0], -1)
        for layer in self.layers.values():
            if isinstance(layer, GradientClassifierLayer) or isinstance(layer, HebbianClassifierLayer):
                pred = layer.forward(y, label)
        return pred

    def forward_test(self, x):
        self.eval()
        with torch.no_grad():
            y = self.test_conv(x)
            y = y.reshape(y.shape[0], -1)
        for layer in self.layers.values():
            if isinstance(layer, GradientClassifierLayer) or isinstance(layer, HebbianClassifierLayer):
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

        def TD_forward_test(self, x):
        return self.TD_forward(x, update_weights=False)
    """
    

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
        globalparams = config['Convolutions']['GlobalParams']
        inhibition = Inhibition.RePU if globalparams['inhibition'] == "REPU" else Inhibition.Softmax
        convlayer = ConvolutionHebbianLayer(input_shape=inputsize, kernel=layerconfig['kernel'], stride=globalparams['stride'], in_ch=input_channel, out_ch=layerconfig['out_channel'], 
                                            lambd=lamb, lr=lr, rho=rho, device=device, eta=eta, padding=globalparams['padding'], paddingmode = globalparams['paddingmode'], triangle=globalparams['triangle'], 
                                            whiten_input=globalparams['whiten'], b=beta, inhibition=inhibition)
        if layerconfig['batchnorm']:
            convlayer.batchnorm = nn.BatchNorm2d(input_channel, affine=False)
        if is_pool:
            poolconfig = config["PoolingBlock"][l_keys[layer_idx]]
            poolglobalparams = config["PoolingBlock"]['GlobalParams']
            convlayer.set_pooling(kernel=poolconfig['kernel'], stride=poolglobalparams['stride'], padding=poolconfig['padding'], type=poolconfig['Type'])
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
    padding_mode = config['Convolutions']["GlobalParams"]['paddingmode']
    triangle = config['Convolutions']["GlobalParams"]['triangle']
    preprocessing = InputProcessing.Whiten if config['Convolutions']["GlobalParams"]['whiten'] else InputProcessing.No 
    inhibition = Inhibition.RePU if config['Convolutions']["GlobalParams"]['inhibition'] == "REPU" else Inhibition.Softmax
    mexican_hat_sigma = config['sigma']

    for layer_idx in range(nb_conv):
        layerconfig = config['Convolutions']["Layers"][l_keys[layer_idx]]
        globallayercongif = config['Convolutions']['GlobalParams']

        convlayer = ConvSoftHebbLayer(input_shape=inputsize, kernel=layerconfig['kernel'], in_ch=input_channel, out_ch=layerconfig['out_channel'], 
                                      stride=globallayercongif['stride'], padding=globallayercongif['padding'], w_lr=w_lr, b_lr=b_lr, l_lr=l_lr, device=device, 
                                      is_output_layer=False, initial_weight_norm=w_norm, triangle=triangle, 
                                      initial_lambda=lamb, inhibition=inhibition, learningrule=LearningRule.SoftHebb, preprocessing=preprocessing,
                                      mexican_hat_sigma=mexican_hat_sigma)
        
        mycnn.add_layer(f"CNNLayer{layer_idx+1}", convlayer)

        input_channel = layerconfig['out_channel']
        inputsize = convlayer.output_shape # update inputsize for next layer
        
        if is_pool:
            poolconfig = config["PoolingBlock"]["Layers"][l_keys[layer_idx]]
            poollayer = PoolingLayer(kernel=poolconfig['kernel'], stride=poolconfig["stride"], padding=poolconfig['padding'], pool_type=poolconfig['Type'])
            
            mycnn.add_layer(f"PoolLayer{layer_idx+1}", poollayer)
            
            inputsize = cnn_output_formula_2D(inputsize, poolconfig['kernel'], poolconfig['padding'], 1, poolconfig["stride"])
            

        output_shape = inputsize
        n_tiles = output_shape[0] * output_shape[1]

        #if is_topdown and layer_idx < (nb_conv-1):
        #    convlayer.Set_TD(inc=config['Convolutions']["Layers"][l_keys[layer_idx+1]]['out_channel'], outc=layerconfig['out_channel'], 
        #                     kernel=config['Convolutions']["Layers"][l_keys[layer_idx+1]]['kernel'], stride=conv_stride)
        

    fc_inputdim = n_tiles * config['Convolutions']["Layers"][l_keys[-1]]['out_channel']
    #print("Fully connected layer input dim : " + str(fc_inputdim))
    
    classifier = HebbianClassifierLayer(fc_inputdim, nbclasses, lr) ###
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
