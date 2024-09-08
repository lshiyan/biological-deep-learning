import argparse
from typing import List, Optional

def parse_arguments(args_list: Optional[List] = None) -> argparse.Namespace:
    """
    FUNCTION
    Parses arguments given on command line and returns arguments
    @param
        args_list: if calling function from script and using list as command line simulation
    @return
        args: parsed arguments
    """
    # Argument parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    # Basic configurations
    parser.add_argument('--model', type=str, default="Hebb")
    parser.add_argument('--data_name', type=str, default="MNIST")
    parser.add_argument('--ext_data_name', type=str, default="E_MNIST")

    parser.add_argument('--experiment_name', type=str, default='DEFAULT')

    # Sub experiment label classes
    parser.add_argument('--sub_experiment_scope_list', type=str, default='[[0,1],[2,3],[4,5],[6,7],[8,9]]')

    # Data Factory
    parser.add_argument('--train_data', type=str, default="data/mnist/train-images.idx3-ubyte")
    parser.add_argument('--train_label', type=str, default="data/mnist/train-labels.idx1-ubyte")
    parser.add_argument('--test_data', type=str, default="data/mnist/test-images.idx3-ubyte")
    parser.add_argument('--test_label', type=str, default="data/mnist/test-labels.idx1-ubyte")
    parser.add_argument('--train_size', type=int, default=60000)
    parser.add_argument('--test_size', type=int, default=10000)
    parser.add_argument('--classes', type=int, default=10)
    
    parser.add_argument('--ext_train_data', type=str, default="data/ext_mnist/train-images.idx3-ubyte")
    parser.add_argument('--ext_train_label', type=str, default="data/ext_mnist/train-labels.idx1-ubyte")
    parser.add_argument('--ext_test_data', type=str, default="data/ext_mnist/test-images.idx3-ubyte")
    parser.add_argument('--ext_test_label', type=str, default="data/ext_mnist/test-labels.idx1-ubyte")
    parser.add_argument('--ext_train_size', type=int, default=60000)
    parser.add_argument('--ext_test_size', type=int, default=10000)
    parser.add_argument('--ext_classes', type=int, default=10)

    # CSV files generated
    parser.add_argument('--train_fname', type=str, default="data/mnist/mnist_train.csv")
    parser.add_argument('--test_fname', type=str, default="data/mnist/mnist_test.csv")
    
    parser.add_argument('--ext_train_fname', type=str, default="data/ext_mnist/ext_mnist_train.csv")
    parser.add_argument('--ext_test_fname', type=str, default="data/ext_mnist/ext_mnist_test.csv")
    
    # Bar generalization experiment specifics
    parser.add_argument('--data_matrix_size', type=int, default=4)
    parser.add_argument('--bar_data_quantity', type=int, default=3)

    # Dimension of each layer
    parser.add_argument('--input_dim', type=int, default=784)
    parser.add_argument('--heb_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=10)

    # Hebbian layer hyperparameters
    parser.add_argument('--heb_lamb', type=float, default=15)
    parser.add_argument('--heb_gam', type=float, default=0.99)
    parser.add_argument('--heb_eps', type=float, default=0.01)
    parser.add_argument('--heb_learn', type=str, default='Sanger')
    parser.add_argument('--heb_inhib', type=str, default='Relu')
    parser.add_argument('--heb_growth', type=str, default='Linear')
    parser.add_argument('--heb_bias', type=str, default='No_Bias')
    parser.add_argument('--heb_focus', type=str, default='Synapse')
    parser.add_argument('--heb_act', type=str, default='Normalized')

    # Classification layer hyperparameters
    parser.add_argument('--class_learn', type=str, default='Controlled')
    parser.add_argument('--class_growth', type=str, default='Linear')
    parser.add_argument('--class_bias', type=str, default='No_Bias')
    parser.add_argument('--class_focus', type=str, default='Synapse')
    parser.add_argument('--class_act', type=str, default='Basic')
    
    # Shared hyperparameters
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument('--sigmoid_k', type=float, default=1)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--beta", type=float, default=10e-7)
    parser.add_argument("--sigma", type=float, default=1)
    parser.add_argument("--mu", type=float, default=0)
    parser.add_argument("--init", type=str, default='uniform')
    parser.add_argument("--random_seed", type=int, default='42')

    # Experiment parameters
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--local_machine", type=bool, default=True)
    parser.add_argument("--experiment_type", type=str, default='base')

    # Parse arguments into Namespace
    args: argparse.Namespace = parser.parse_args() if args_list == None else parser.parse_args(args_list)

    return args