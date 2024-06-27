import argparse
from typing import List

def parse_arguments(args_list: List = None) -> argparse.Namespace:
    """
    FUNCTION
    Parse arguments giving on command line and return arguments
    @param
        args_list: if calling function from script and using list as command line simulation
    @return
        args: parsed arguments
    """
    # Argument parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    # Basic configurations
    parser.add_argument('--is_training', type=bool, default=True, help='status')
    parser.add_argument('--data_name', type=str, default="MNIST")

    # Data Factory
    parser.add_argument('--train_data', type=str, default="data/mnist/train-images.idx3-ubyte")
    parser.add_argument('--train_label', type=str, default="data/mnist/train-labels.idx1-ubyte")
    parser.add_argument('--test_data', type=str, default="data/mnist/t10k-images.idx3-ubyte")
    parser.add_argument('--test_label', type=str, default="data/mnist/t10k-labels.idx1-ubyte")

    # CSV files generated
    parser.add_argument('--train_filename', type=str, default="data/mnist/mnist_train.csv")
    parser.add_argument('--test_filename', type=str, default="data/mnist/mnist_test.csv")

    # Dimension of each layer
    parser.add_argument('--input_dim', type=int, default=784)
    parser.add_argument('--heb_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=10)

    # Hebbian layer hyperparameters
    parser.add_argument('--heb_lamb', type=float, default=15)
    parser.add_argument('--heb_gam', type=float, default=0.99)
    parser.add_argument('--heb_eps', type=float, default=0.01)
    parser.add_argument('--learning_rule', type=str, default='Sanger')
    parser.add_argument('--inhibition_rule', type=str, default='Relu')
    parser.add_argument('--function_type', type=str, default='Linear')

    # Classification layer hyperparameters
    parser.add_argument('--include_first', type=bool, default=True)
    
    # Shared hyperparameters
    parser.add_argument("--lr", type=float, default=0.005)

    # Experiment parameters
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--test_epochs", type=int, default=1)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--local_machine", type=bool, default=True)

    # Parse arguments into Namespace
    args: argparse.Namespace = parser.parse_args() if args_list == None else parser.parse_args(args_list)

    return args