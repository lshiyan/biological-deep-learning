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
    
    def __init__(self, args, input_dimension, hidden_layer_dimension, output_dimension, 
                 lamb=1.5, heb_lr=0.01, grad_lr=0.001, num_epochs=3):
        self.model=HebbianNetwork(input_dimension, hidden_layer_dimension, 
                                  output_dimension, heb_lr=heb_lr, lamb=lamb)#TODO: For some reason my hebbian network is not processing batches together.
        self.args=args
        self.num_epochs=num_epochs
        self.grad_lr=grad_lr
    
    #Returns ADAM optimize for gradient descent.
    def optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), self.grad_lr)
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
        oud_O = None

        for epoch in range(self.num_epochs):
            print(f"Training epoch {epoch}...")
            for i, data in enumerate(data_loader):
                inputs, labels=data
                outputs = self.model(inputs, self.oneHotEncode(labels,10))
                #print((torch.softmax(outputs, dim=1, dtype=torch.float)))
                oud_O = outputs
                #break
                # if i == 60000 :
                #     self.visualizeWeights(10)
                #     break
                # """loss=loss_function(outputs, labels)
                # loss.backward()
                # optimizer.zero_grad()
                # optimizer.step()"""
            print(self.model.hebbian_layer.fc.weight)
                #
    #Given a tensor of labels, returns a one hot encoded tensor for each label.
    def oneHotEncode(self, labels, num_classes):
        one_hot_encoded = torch.zeros(len(labels), num_classes)
        one_hot_encoded.scatter_(1, labels.unsqueeze(1), 1)
    
        return one_hot_encoded
        
    #Visualizes the weights associated with the first feature detector layer.
    def visualizeWeights(self, num_choices):
        self.model.visualizeWeights(num_choices)
    
    def test(self):
        data_set=MNIST_set(self.args, 0)
        data_loader=DataLoader(data_set, batch_size=1)
        cor=0
        tot=0
        #print(self.model.classifier_layer.fc.weight)
        for i, data in enumerate(data_loader):
            if i == 1000:
                break
            inputs, labels=data
            #print((self.model(inputs, labels)))
            outputs = (torch.softmax(self.model.use_forward(inputs), dim=1, dtype=torch.float))
            predict = torch.argmax(outputs,dim=1)
            #print(predict.item() > 4)
            if labels.item() == 0:
                print(outputs)
                print(labels)
            if predict.item()==labels.item():
                cor+=1
            tot+=1
        print("Accuracy:", cor/tot)
    
if __name__=="__main__":
    experiment=MLPExperiment(None, 784, 256, 10, lamb=30, num_epochs=2)
    experiment.train()
    experiment.visualizeWeights(20)
    experiment.test()
    
            
            
        