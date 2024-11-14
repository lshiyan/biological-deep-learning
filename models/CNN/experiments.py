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
from models.CNN.layers import ConvSoftHebbLayer, PoolingLayer



def CNN_Experiment(epoch, mymodel, dataloader, testloader, dataset, nclasses, imgtype, device, traintopdown=False, testtopdown=False, greedytrain=False):

    layers = list(mymodel.basemodel.layers.values())

    if greedytrain:
        for idx in range(len(mymodel.basemodel.layers)):
            idx += 1
            for _ in range(epoch):
                for data in tqdm(dataloader):
                    inputs, labels = data
                    labels = labels.to(device)
                    if imgtype == ImageType.Gray:
                        inputs = inputs.reshape(1,1,28,28)
                    x = inputs.to(device)
                    for r_l in range(idx):
                        if (r_l + 1) == idx:
                            _, x = layers[r_l].forward(x, oneHotEncode(labels, nclasses, mymodel.device))
                        else :
                            _, x = layers[r_l].forward(x, update_weights=False)
    else :
        mymodel.eval()
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
                mymodel.train_conv(inputs, oneHotEncode(labels, nclasses, mymodel.device))
            #if cn == 10:
            #    return mymodel

    for _ in range(1):
        for data in tqdm(dataloader):
            inputs, labels=data
            # inputs = inputs.to('mps')
            # labels = labels.to('mps')
            #print(inputs)
            inputs = inputs.to(device)
            labels = labels.to(device)
            if imgtype == ImageType.Gray:
                inputs = inputs.reshape(1,1,28,28)
            mymodel.train_classifier(inputs, oneHotEncode(labels, nclasses, mymodel.device))

    # timestr = time.strftime("%Y%m%d-%H%M%S")

    # if traintopdown:
    #     foldername = os.getcwd() + '/SavedModels/CNN_TD_' + dataset + '_' + timestr
    # else:
    #     foldername = os.getcwd() + '/SavedModels/CNN_FF_' + dataset + '_' + timestr

    # os.mkdir(foldername)

    # torch.save(mymodel.state_dict(), foldername + '/model')

    #view_filters(mymodel, foldername)
    #view_ff_weights(mymodel, foldername, dataloader)
    return mymodel.basemodel, mymodel


def new_CNN_Experiment(epoch, mymodel, dataloader, nclasses, imgtype, device, greedytrain=False):

    layers = list(mymodel.layers.values())
    mymodel.train()
    
    #lamb_values = {layer_name: [] for layer_name in mymodel.basemodel.layers.keys()}

    if greedytrain:
        for idx in range(len(mymodel.basemodel.layers)): 
            if isinstance(layers[idx], ConvSoftHebbLayer):
                mymodel.set_conv_training_layers([layers[idx]])
            else:
                continue
                # no greedy training on pooling (or classification) layers
            for _ in range(epoch):
                for data in tqdm(dataloader):
                    inputs, labels = data
                    labels = labels.to(device)
                    if imgtype == ImageType.Gray:
                        inputs = inputs.reshape(1,1,28,28)
                    x = inputs.to(device)
                    target = oneHotEncode(labels, nclasses, mymodel.device)
                    for r_l in range(idx+1): 
                        if isinstance(layers[r_l], ConvSoftHebbLayer):
                            # if you've reached layer idx, perform forward pass with targets
                            if ((r_l) == idx): 
                                x = layers[r_l].forward(x, target)
                            else : 
                                # for all layers before idx, perform forward pass without targets
                                x = layers[r_l].forward(x, target=None)
                        elif isinstance(layers[r_l], PoolingLayer):
                            x = layers[r_l].forward(x)

                    #if hasattr(layers[idx], 'lamb'):
                    #    lamb_values[mymodel.basemodel.layers[idx].items()[0]].append(layers[idx].lamb.item())


    else :

        for _ in range(epoch):
            for data in tqdm(dataloader):
                inputs, labels=data
                inputs = inputs.to(device)
                labels = labels.to(device)
                if imgtype == ImageType.Gray:
                    inputs = inputs.reshape(1,1,28,28)
                mymodel.train_conv(inputs, oneHotEncode(labels, nclasses, mymodel.device)) ###

                #for idx in range(len(mymodel.basemodel.layers)):
                #    if hasattr(layers[idx], 'lamb'):
                #       lamb_values[mymodel.basemodel.layers[idx].items()[0]].append(layers[idx].lamb.item())


    for _ in range(1):

        for data in tqdm(dataloader):
            inputs, labels=data
            inputs = inputs.to(device)
            labels = labels.to(device)
            if imgtype == ImageType.Gray:
                inputs = inputs.reshape(1,1,28,28)
            mymodel.train_classifier(inputs, oneHotEncode(labels, nclasses, mymodel.device))

    #plot_lambda(lamb_values)

    return mymodel



def plot_lambda(lamb_values):
    for layer_name, lamb_list in lamb_values.items():
        plt.figure(figsize=(10, 6))
        plt.plot(lamb_list, label=f'Lambda values for {layer_name}')
        plt.xlabel('Training Iterations')
        plt.ylabel('Lambda Value')
        plt.title(f'Tracking Lambda for {layer_name}')
        plt.legend()
        plt.grid(True)
        plt.show()


def CNN_Baseline_test(mymodel, data_loader, imgtype, topdown):
    yes = 0
    tot = 0
    da = {}
    for data in tqdm(data_loader):
        inputs, labels = data
        batch_size = labels.size(0)
        tot += batch_size
        if imgtype == ImageType.Gray:
            inputs = inputs.reshape(batch_size,1,28,28)
        if topdown:
            output = torch.argmax(mymodel.TD_forward_test(inputs), dim=1)
        else:
            y = mymodel.forward_test(inputs)
            output = torch.argmax(y, dim=1) ######

        for label, prediction in zip(labels, output):
            label = label.item()
            prediction = prediction.item()

            if label not in da:
                da[label] = (0, 0, 0)
            if prediction not in da:
                da[prediction] = (0, 0, 0)

            r, t, w = da[label]
            if prediction == label:
                yes += 1
                da[label] = (r+1, t+1, w)
            else:
                da[label] = (r, t+1, w)
                r1, t1, w1 = da[prediction]
                da[prediction] = (r1, t1, w1+1)

    accuracy = yes / tot
    return accuracy, da

