import argparse
import os
from data.mnist_factory import convert
from data.data_loader import Image_Data_Set
from experiments.mlp import MLPExperiment  

#Test

def run_experiment(args):
    # Initialize the experiment
    experiment = MLPExperiment(args, args.input_dimension, args.hidden_layer_dimension, args.output_dimension,
                                lamb=args.lamb, heb_lr=args.heb_lr, grad_lr=args.grad_lr, num_epochs=args.num_epochs,
                                gamma=args.gamma, eps=args.eps)
    
    # Run training
    experiment.train()

    # Visualization and testing
    experiment.visualize_weights()
    accuracy = experiment.test()
    print(f"Test Accuracy: {accuracy}")

# Main function of the module
def main(args):
    # Create .csv file from the ubyte data files
    if not os.path.exists(args.train_data_filename) or not os.path.exists(args.test_data_filename):
        print(f"Converting {args.data_name} data")
        convert(args.train_data, args.train_labels, args.train_data_filename, 60000, 28)
        convert(args.test_data, args.test_labels, args.test_data_filename, 10000, 28)
    
    # Load datasets
    train_data = ImageDataSet(True, args.data_name)
    test_data = ImageDataSet(False, args.data_name)
    train_data.setup_data(args.train_data_filename)
    test_data.setup_data(args.test_data_filename)
    
    # Run experiment
    run_experiment(args)

# Run an experiment with the arguments passed
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Biological deep learning')
    
    # Basic configurations.
    parser.add_argument('--is_training', type=bool, default=True, help='status')
    parser.add_argument('--data_name', type=str, default="MNIST")
    
    # Data Factory
    parser.add_argument('--train_data', type=str, default="data/mnist/train-images.idx3-ubyte")
    parser.add_argument('--train_labels', type=str, default="data/mnist/train-labels.idx1-ubyte")
    parser.add_argument('--test_data', type=str, default="data/mnist/t10k-images.idx3-ubyte")
    parser.add_argument('--test_labels', type=str, default="data/mnist/t10k-labels.idx1-ubyte")

    # CSV files generated
    parser.add_argument('--train_data_filename', type=str, default="data/mnist/mnist_train.csv")
    parser.add_argument('--test_data_filename', type=str, default="data/mnist/mnist_test.csv")

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
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the main function for the experiment
    main(args)
