import argparse
import torch

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Biological deep learning')
    
    parser.add_argument('--test', type=str, default="testing")
    
    #Data loader
    parser.add_argument('--data_root', type=str, default="data/mnist")
    parser.add_argument('--data_filename', type=str, default="mnist_train.csv")
    
    args=parser.parse_args()