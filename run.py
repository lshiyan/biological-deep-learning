import argparse
import os
from data.mnist_factory import convert
from data.data_loader import MNIST_set, fashion_MNIST_set
from experiments.mlp import MLPExperiment  


def convert_data(args):
    # Ensure the CSV files exist or create them using the factory
    if not os.path.exists(args.train_data_filename) or not os.path.exists(args.test_data_filename):
        print("Converting MNIST data")
        convert(args.train_data, args.train_labels, args.train_data_filename, 60000)
        convert(args.test_data, args.test_labels, args.test_data_filename, 10000)

    if not os.path.exists(args.train_data_filename_fashion) or not os.path.exists(args.test_data_filename_fashion):
        print("Converting Fashion MNIST data")
        convert(args.train_data_fashion, args.train_labels_fashion, args.train_data_filename_fashion, 60000)
        convert(args.test_data_fashion, args.test_labels_fashion, args.test_data_filename_fashion, 10000)

def load_datasets(args):
    # Load datasets
    mnist_train = MNIST_set(args, train=1)
    mnist_test = MNIST_set(args, train=0)
    fashion_mnist_train = fashion_MNIST_set(args, train=1)
    fashion_mnist_test = fashion_MNIST_set(args, train=0)

    return mnist_train, mnist_test, fashion_mnist_train, fashion_mnist_test

def run_experiment(args):
    # Initialize the experiment
    experiment = MLPExperiment(args, args.input_dimension, args.hidden_layer_dimension, args.output_dimension,
                                lamb=args.lamb, heb_lr=args.heb_lr, grad_lr=args.grad_lr, num_epochs=args.num_epochs,
                                gamma=args.gamma, eps=args.eps)

    print("Entered run_experiment function")
    
    # Run training
    experiment.train()

    # Visualization and testing
    experiment.visualizeWeights()
    accuracy = experiment.test()
    print(f"Test Accuracy: {accuracy}")


def main(args):
    convert_data(args)
    mnist_train, mnist_test, fashion_mnist_train, fashion_mnist_test = load_datasets(args)
    
    # Logic for training or testing model:
    if args.run_experiment:
        run_experiment(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Biological deep learning')
    
    # Basic configurations.
    parser.add_argument('--is_training', type=bool, default=True, help='status')
    
    # Data Factory
    parser.add_argument('--train_data', type=str, default="data/mnist/train-images.idx3-ubyte")
    parser.add_argument('--train_labels', type=str, default="data/mnist/train-labels.idx1-ubyte")
    parser.add_argument('--test_data', type=str, default="data/mnist/t10k-images.idx3-ubyte")
    parser.add_argument('--test_labels', type=str, default="data/mnist/t10k-labels.idx1-ubyte")

    parser.add_argument('--train_data_fashion', type=str, default="data/fashion_mnist/train-images.idx3-ubyte")
    parser.add_argument('--train_labels_fashion', type=str, default="data/fashion_mnist/train-labels.idx1-ubyte")
    parser.add_argument('--test_data_fashion', type=str, default="data/fashion_mnist/t10k-images.idx3-ubyte")
    parser.add_argument('--test_labels_fashion', type=str, default="data/fashion_mnist/t10k-labels.idx1-ubyte")

    # CSV files generated
    parser.add_argument('--train_data_filename', type=str, default="data/mnist/mnist_train.csv")
    parser.add_argument('--test_data_filename', type=str, default="data/mnist/mnist_test.csv")
    parser.add_argument('--train_data_filename_fashion', type=str, default="data/fashion_mnist/fashion_mnist_train.csv")
    parser.add_argument('--test_data_filename_fashion', type=str, default="data/fashion_mnist/fashion_mnist_test.csv")

    # Experiment specific configurations
    parser.add_argument('--input_dimension', type=int, default=784)
    parser.add_argument('--hidden_layer_dimension', type=int, default=64)
    parser.add_argument('--output_dimension', type=int, default=10)
    parser.add_argument('--lamb', type=float, default=5)
    parser.add_argument('--heb_lr', type=float, default=0.005)
    parser.add_argument('--grad_lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--run_experiment', type=bool, default=True, help="Run the experiment")

    
    args = parser.parse_args()
    
    main(args)
