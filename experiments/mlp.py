import torch
import torch.nn as nn 
import torch.optim as optim

from torch.utils.data import DataLoader
from data.data_loader import MNIST_set
from models.hebbian_network import HebbianNetwork
from layers.scheduler import Scheduler


class MLPExperiment():
    
    def __init__(self, args, input_dimension, hidden_layer_dimension, output_dimension, 
                 lamb=1, heb_lr=1, grad_lr=0.001, num_epochs=3, gamma=0):
        self.model=HebbianNetwork(input_dimension, hidden_layer_dimension, 
                                  output_dimension, heb_lr=heb_lr, lamb=lamb)#TODO: For some reason my hebbian network is not processing batches together.
        self.args=args
        self.num_epochs=num_epochs
        self.grad_lr=grad_lr 
        self.heb_lr=heb_lr
        self.gamma=gamma
        
    #Returns ADAM optimize for gradient descent.
    def optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), self.grad_lr)
        return optimizer
    
    #Sets the scheduler for the feature detector layer of our network.
    def set_hebbian_scheduler(self):
        scheduler=Scheduler(self.heb_lr, 1000, self.gamma)
        self.model.setScheduler(scheduler, 0)

    #Returns cross entropy loss function.
    def loss_function(self):
        loss_function = nn.CrossEntropyLoss()
        return loss_function
    
    #Trains the experiment.
    def train(self):  
        data_set=MNIST_set(self.args)
        data_loader=DataLoader(data_set, batch_size=1, shuffle=True)
        
        self.model.train()
        
        optimizer=self.optimizer()
        if self.gamma !=0 : self.set_hebbian_scheduler()
        
        for _ in range(self.num_epochs):
            for i, data in enumerate(data_loader):
                inputs, labels=data
                self.model(inputs, clamped_output=None)
                optimizer.step()

            
    #Given a tensor of labels, returns a one hot encoded tensor for each label.
    def oneHotEncode(self, labels, num_classes):
        one_hot_encoded = torch.zeros(len(labels), num_classes)
        one_hot_encoded.scatter_(1, labels.unsqueeze(1), 1)
    
        return one_hot_encoded
        
    #Visualizes the weights associated with the first feature detector layer.
    def visualizeWeights(self, num_choices, classifier=0):
        self.model.visualizeWeights(num_choices, classifier)
    
    def test(self):
        data_set=MNIST_set(self.args, 0)
        data_loader=DataLoader(data_set, batch_size=1, shuffle=True)
        cor=0
        tot=0
        for _, data in enumerate(data_loader):
            inputs, labels=data
            outputs = torch.argmax(torch.softmax(self.model(inputs, None, train=0), dim=1))
            print(outputs)
            if outputs.item()==labels.item():
                cor+=1
            tot+=1
        print("Accuracy:", cor/tot)
    
if __name__=="__main__":
    experiment=MLPExperiment(None, 784, 256 , 10, lamb=1, num_epochs=1, heb_lr=0.1)
    experiment.train()
    experiment.visualizeWeights(10, classifier=0)
    experiment.visualizeWeights(10, classifier=1)
    #experiment.test()
    
            
            
        