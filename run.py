import argparse
import os
from data.mnist_factory import convert
from data.data_loader import ImageDataSet
from experiments.mlp import MLPExperiment  

#Test

def run_experiment(args):
    # Initialize the experiment
    experiment = MLPExperiment(args)
    
    # Run training
    experiment.train()

    # Visualization and testing
    experiment.visualize_weights()
    accuracy = experiment.test()
    print(f"Test Accuracy: {accuracy}")

# Main function of the module
def main(args):
    # Create .csv file from the ubyte data files
    if not os.path.exists(args.train_filename) or not os.path.exists(args.test_filename):
        print(f"Converting {args.data_name} data")
        convert(args.train_data, args.train_labels, args.train_filename, 60000, 28)
        convert(args.test_data, args.test_labels, args.test_filename, 10000, 28)
    
    # Load datasets
    train_data = ImageDataSet(True, args.data_name)
    test_data = ImageDataSet(False, args.data_name)
    train_data.setup_data(args.train_filename)
    test_data.setup_data(args.test_filename)
    
    # Run experiment
    #run_experiment(args)

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
    parser.add_argument('--train_filename', type=str, default="data/mnist/mnist_train.csv")
    parser.add_argument('--test_filename', type=str, default="data/mnist/mnist_test.csv")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the main function for the experiment
    main(args)
