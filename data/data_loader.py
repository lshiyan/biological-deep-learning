import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class MNIST_set(Dataset):
    
    def __init__(self, args, train=1):
        
        """self.args=args
        
        if args.is_training:
            self.flag='train'
        else:
            self.flag='test'"""
            
        #self.root_path=args.data_root
        #self.filename=args.data_filename
        
        #self.dataframe=pd.read_csv(self.root_path+self.filename)
        
        if train:
            self.dataframe=pd.read_csv('data/mnist/mnist_train.csv', header=None)
            self.labels=torch.tensor(self.dataframe[0].values)
            self.dataframe=torch.tensor(self.dataframe.drop(self.dataframe.columns[0], axis=1).values, dtype=torch.float)
            self.dataframe/=255

        else:
            self.dataframe=pd.read_csv('data/mnist/mnist_test.csv', header=None)
            self.labels=torch.tensor(self.dataframe[0].values)
            self.dataframe=torch.tensor(self.dataframe.drop(self.dataframe.columns[0], axis=1).values, dtype=torch.float)
            self.dataframe/=255

    def normalize(self, row):
        row=row/255
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.labels[idx]
        features = self.dataframe[idx]
        
        return features, label
        
class fashion_MNIST_set(Dataset):
    
    def __init__(self, args, train=1):
        
        """self.args=args
        
        if args.is_training:
            self.flag='train'
        else:
            self.flag='test'"""
            
        #self.root_path=args.data_root
        #self.filename=args.data_filename
        
        #self.dataframe=pd.read_csv(self.root_path+self.filename)
        
        if train:
            self.dataframe=pd.read_csv('data/fashion_mnist/fashion_mnist_train.csv', header=None)
            self.labels=torch.tensor(self.dataframe[0].values)
            self.dataframe=torch.tensor(self.dataframe.drop(self.dataframe.columns[0], axis=1).values, dtype=torch.float)
            self.dataframe/=255
        
        else:
            self.dataframe=pd.read_csv('data/fashion_mnist/fashion_mnist_test.csv', header=None)
            self.labels=torch.tensor(self.dataframe[0].values)
            self.dataframe=torch.tensor(self.dataframe.drop(self.dataframe.columns[0], axis=1).values, dtype=torch.float)
            self.dataframe/=255

    def normalize(self, row):
        row=row/255
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.labels[idx]
        features = self.dataframe[idx]
        
        return features, label
    
if __name__=="__main__":
    mnist=MNIST_set(None)
    fashion_MNIST_set(None)


# TODO: modify code to implement following class and better programming

"""
Class to setup datasets (seperating images and labels) for both training and testing purposes
"""
class Image_Data_Set(Dataset):
    """
    Contructor method
    @param
        train (bool) = is the dataset in training or testing
        name (str) = name of data set
    @attr.
        flag (str) = a string indicating wether the data is used for training or testing
        data_frame (list-like object) = dataset for training/testing
        labels (torch.Tensor) = labels of dataset
    """
    def __init__(self, train=True, name='MNIST'):
        self.__name = name
        self.__flag = 'train' if train else 'test'
        self.__data_frame = None
        self.__labels = None

    """
    Set up the data frame and labels with the defined data set
    @param
        data_set (str) = string defining path to .csv file of data set
    """
    def setup_data(self, data_set):
        self.data_frame=pd.read_csv(data_set, header=None)
        self.labels=torch.tensor(self.dataframe[0].values)
        self.data_frame=torch.tensor(self.data_frame.drop(self.data_frame.columns[0], axis=1).values, dtype=torch.float)
        self.data_frame/=255
    
    # Normalize number(s) assuming max value = 255
    def normalize(self, row):
        row=row/255
    
    # Get length of data frame    
    def __len__(self):
        return len(self.dataframe)
    
    # Attritube getter functions
    def get_flag(self):
        return self.__flag
    
    def get_name(self):
        return self.__name

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.labels[idx]
        features = self.dataframe[idx]
        
        return features, label



# TODO: Write this part of the code somewhere else
#if __name__=="__main__":
    #mnist=MNIST_set(None)
    #fashion_MNIST_set(None)
    #mnist = Image_Data_Set(name="MNIST")
    #fashion_mnist = Image_Data_Set(name="Fashion_MNIST")