import argparse
import torch

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Biological deep learning')
    
    #Basic configurations.
    parser.add_argument('--is_training', type=int, default=1, help='status')
    
    #Data loader.
    parser.add_argument('--data_root', type=str, default="mnist/")
    parser.add_argument('--data_filename', type=str, default="mnist_train.csv")
    
    
    #Hyperparameters.
    parser.add_argument('--learning_rate', type=int, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=100)
    
    args=parser.parse_args()
    
    if args.is_training:
        print("training")