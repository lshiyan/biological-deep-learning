import torch
import torch.nn as nn 
import torch.optim as optim
import os
import matplotlib.pyplot as plt 

from torch.utils.data import Dataset, DataLoader
from data.data_loader import MNIST_set
from models.hebbian_network import HebbianNetwork
from layers.hebbian_layer import HebbianLayer



class MLPExperiment():
    
    def __init__(self, args, input_dimension, hidden_layer_dimension, output_dimension, lamb=1, lr=0.001, num_epochs=3):
        self.model=HebbianNetwork(input_dimension, hidden_layer_dimension, output_dimension, 2)#TODO: For some reason my hebbian network is not processing batches together.
        self.args=args
        self.num_epochs=num_epochs
        self.lr=lr
    
    #Returns ADAM optimize for gradient descent.
    def optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), self.lr)
        return optimizer

    #Returns cross entropy loss function.
    def loss_function(self):
        loss_function = nn.CrossEntropyLoss()
        return loss_function
    
    #Trains the experiment 
    def train(self):
        losses=[]
        
        data_set=MNIST_set(self.args)
        data_loader=DataLoader(data_set, batch_size=1)
        
        optimizer = self.optimizer()
        loss_function = self.loss_function()
        
        self.model.train()
        for epoch in range(self.num_epochs):
            for i, data in enumerate(data_loader):
                inputs, labels=data
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss=loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
        
        
    def visualizeWeights():
        return
    
    def test(self):
        data_set=MNIST_set(self.args, 0)
        data_loader=DataLoader(data_set, batch_size=1)
        cor=0
        tot=0
        for i, data in enumerate(data_loader):
            inputs, labels=data
            outputs = torch.argmax(torch.softmax(self.model(inputs), dim=1), dim=1)
            if outputs.item()==labels.item():
                cor+=1
            tot+=1
        print("Accuracy:", cor/tot)
    
if __name__=="__main__":
    experiment=MLPExperiment(None, 784, 256, 10)
    experiment.train()
    experiment.test()
            
            
        