import os 
import torch
import matplotlib.pyplot as plt
import time
from tqdm.auto import tqdm
from models.hyperparams import (LearningRule, WeightScale, oneHotEncode, InputProcessing,
                                Inhibition, WeightGrowth)
from models.MLP.model import view_weights


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


def SoftMLPBaseline_Experiment(epoch, mymodel, dataloader, dataset, nclasses, device, greedytrain):
    lamb_values = {layer_name: [] for layer_name in mymodel.layers.keys()}
    
    # layer-wise training
    if greedytrain:
        for layer_name, layer in mymodel.layers.items():
            mymodel.set_training_layers([layer])
            print(f"Training layer: {layer_name}")
            for _ in range(epoch):
                for data in tqdm(dataloader):
                    inputs, labels = data
                    x = inputs.to(device)
                    target = oneHotEncode(labels, nclasses, device)
                    
                    for prev_layer_name, prev_layer in mymodel.layers.items():
                        if prev_layer_name == layer_name:
                            break
                        x = prev_layer.forward(x)
                    
                    layer.forward(x, target)

                    if hasattr(layer, 'lamb'):
                        lamb_values[layer_name].append(layer.lamb.item())
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        foldername = os.getcwd() + '/SavedModels/SoftMLP_FF_Greedy_' + dataset + '_' + timestr

    # Standard training
    else:
        mymodel.train()
        for _ in range(epoch):
            for data in tqdm(dataloader):
                inputs, labels = data
                mymodel.forward(inputs, oneHotEncode(labels, nclasses, device))

                for layer_name, layer in mymodel.layers.items():
                    if hasattr(layer, 'lamb'):
                        lamb_values[layer_name].append(layer.lamb.item())
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        foldername = os.getcwd() + '/SavedModels/SoftMLP_FF_' + dataset + '_' + timestr

    os.mkdir(foldername)
    torch.save(mymodel.state_dict(), foldername + '/model')
    view_weights(mymodel, foldername)
    plot_lambda(lamb_values)

    return mymodel


def MultiSoftMLPBaseline_Experiment(epoch, mymodel, dataloader, dataset, nclasses, device, greedytrain):
    lamb_values = {layer_name: [] for layer_name in mymodel.layers.keys()}
    
    # layer-wise training
    if greedytrain:
        for layer_name, layer in mymodel.layers.items():
            mymodel.set_training_layers([layer])
            print(f"Training layer: {layer_name}")
            for _ in range(epoch):
                for data in tqdm(dataloader):
                    inputs, labels = data
                    x = inputs.to(device)
                    target = oneHotEncode(labels, nclasses, device)
                    
                    for prev_layer_name, prev_layer in mymodel.layers.items():
                        if prev_layer_name == layer_name:
                            break
                        x = prev_layer.forward(x)
                    
                    layer.forward(x, target)

                    if hasattr(layer, 'lamb'):
                        lamb_values[layer_name].append(layer.lamb.item())
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        foldername = os.getcwd() + '/SavedModels/SoftMLP_FF_Greedy_' + dataset + '_' + timestr

    # Standard training
    else:
        mymodel.train()
        for _ in range(epoch):
            for data in tqdm(dataloader):
                inputs, labels = data
                mymodel.forward(inputs, oneHotEncode(labels, nclasses, device))

                for layer_name, layer in mymodel.layers.items():
                    if hasattr(layer, 'lamb'):
                        lamb_values[layer_name].append(layer.lamb.item())
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        foldername = os.getcwd() + '/SavedModels/SoftMLP_FF_' + dataset + '_' + timestr

    os.mkdir(foldername)
    torch.save(mymodel.state_dict(), foldername + '/model')
    #view_weights(mymodel, foldername)
    #plot_lambda(lamb_values)

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