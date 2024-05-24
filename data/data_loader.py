import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


"""
Class to setup datasets (seperating images and labels) for both training and testing purposes
"""
class ImageDataSet(Dataset):
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
        self.data_frame = pd.read_csv(data_set, header=None)
        self.labels = torch.tensor(self.data_frame[0].values)
        self.data_frame = torch.tensor(self.data_frame.drop(self.data_frame.columns[0], axis=1).values, dtype=torch.float)
        self.data_frame /= 255
    
    # Normalize number(s) assuming max value = 255
    def normalize(self, row, max_num):
        row=row/max_num #255
    
    # Get length of data frame    
    def __len__(self):
        return len(self.data_frame)
    
    # Attritute getter functions
    def get_flag(self):
        return self.__flag
    
    def get_name(self):
        return self.__name

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.labels[idx]
        features = self.data_frame[idx]
        
        return features, label