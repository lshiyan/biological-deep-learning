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
        
            
if __name__=="__main__":
    mnist=MNIST_set(None)