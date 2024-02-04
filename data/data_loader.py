import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class MNIST_set(Dataset):
    
    def __init__(self, args):
        
        """self.args=args
        
        if args.is_training:
            self.flag='train'
        else:
            self.flag='test'"""
            
        #self.root_path=args.data_root
        #self.filename=args.data_filename
        
        #self.dataframe=pd.read_csv(self.root_path+self.filename)
        
        self.dataframe=pd.read_csv('../data/mnist/mnist_train.csv')

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = torch.tensor(self.dataframe.iloc[idx, 0], dtype=torch.long)
        features = torch.tensor(self.dataframe.iloc[idx, 1:].values, dtype=torch.float32)
        
        return features, label
        
            
if __name__=="__main__":
    mnist_csv=pd.read_csv("mnist/mnist_train.csv", header=None)