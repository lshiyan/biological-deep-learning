import argparse
import os
from experiments.mlp import MLPExperiment

#Reads script files to get arguments.
def read_args_file(filename):
    with open(filename, 'r') as file:
        args = []
        for line in file:
            # Split each line on whitespace
            parts = line.strip().split()
            # Add each argument to the list
            args.extend(parts)
        return args
    
#Returns argument namespace from a given config file.
def get_args(filename):
    parser = argparse.ArgumentParser(description='Biological deep learning')
    
    #Dataset.
    parser.add_argument('--dataset', type=str, default="mnist")

    #Architecture.
    parser.add_argument('--d_input', type=int)
    parser.add_argument('--d_hidden', type=int)
    parser.add_argument('--d_output', type=int)

    #Hyperparameters.
    parser.add_argument('--lamb', type=int, default=15)
    parser.add_argument('--eps', type=float, default=0.01)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--num_epochs', type=int, default=3)
    
    arg_list=read_args_file(filename)
    
    args=parser.parse_args(arg_list)
    
    return args

if __name__=="__main__":
    args=get_args('config/mlp.config')
    experiment=MLPExperiment(args.d_input, args.d_hidden, args.d_output, lamb=args.lamb, 
                             num_epochs=args.num_epochs, heb_lr=args.learning_rate, eps=args.eps)
    
    experiment.train()