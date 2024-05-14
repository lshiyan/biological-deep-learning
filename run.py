import argparse
import torch

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Biological deep learning')
    
    # Basic configurations.
    parser.add_argument('--is_training', type=bool, default=True, help='status')
    
    # Data Factory
    parser.add_argument('--test_data', type=str, default="data/mnist/t10k-images.idx3-ubyte")
    parser.add_argument('--test_labels', type=str, default="data/mnist/t10k-labels.idx1-ubyte")
    parser.add_argument('--train_data', type=str, default="data/mnist/train-images.idx3-ubyte")
    parser.add_argument('--train_labels', type=str, default="data/mnist/train-labels.idx1-ubyte")
    
    # Data loader.
    parser.add_argument('--data_root', type=str, default="data/mnist/")
    parser.add_argument('--train_data_filename', type=str, default="mnist_train.csv")
    parser.add_argument('--test_data_filename', type=str, default="mnist_test.csv")
    
    # Hyperparameters.
    parser.add_argument('--learning_rate', type=int, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=100)
    
    args=parser.parse_args()
    
    if args.is_training:
        print("training")