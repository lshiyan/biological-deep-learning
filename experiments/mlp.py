import torch
import torch.nn as nn 
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader
from data.data_loader import MNIST_set
from models.hebbian_network import HebbianNetwork


class MLPExperiment():
    
    def __init__(self, args, input_dimension, hidden_layer_dimension, output_dimension):
        self.model=nn.Linear(input_dimension, output_dimension)
        self.args=args
    
    #Returns ADAM optimize for gradient descent.
    def optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer

    #Returns cross entropy loss function.
    def loss_function(self):
        loss_function = nn.CrossEntropyLoss()
        return loss_function
    
    #Trains the experiment 
    def train(self):
        data_set=MNIST_set(self.args)
        data_loader=DataLoader(data_set, batch_size=16)
        
        optimizer = self.optimizer()
        loss_function = self.loss_function()
        
        self.model.train()
        for epoch in range(1):
            for i, data in enumerate(data_loader):
                inputs, labels=data
                print(inputs)
                print(labels)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                print(outputs)
                print(labels)
                
if __name__=="__main__":
    experiment=MLPExperiment(None, 784, 256, 10)
    experiment.train()
            
            
        