import argparse
import os
from data.data_loader import ImageDataSet
from experiments.mlp import MLPExperiment  

#  File to be able to run an experiment through command prompt and giving appropriate arguments

"""
Main function of the module -> preps data and runs experiment with given arguments
"""
def main(args):
    # Initialize the experiment
    experiment = MLPExperiment(args)
    
    # Run training
    experiment.train()

    # Visualization and testing
    experiment.visualize_weights()
    accuracy = experiment.test()
    print(f"Test Accuracy: {accuracy}")


"""
Gets arguments passed through command prompt and calls main() function with given arguments
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Biological deep learning')
    
    # Basic configurations.
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
    parser.add_argument('--heb_lr', type=float, default=0.001)
    parser.add_argument('--heb_lamb', type=float, default=15)
    parser.add_argument('--heb_gam', type=float, default=0.99)

    # Classification layer hyperparameters
    parser.add_argument('--cla_lr', type=float, default=0.001)
    parser.add_argument('--cla_lamb', type=float, default=1)
    parser.add_argument('--cla_gam', type=float, default=0.99)

    # Shared hyperparameters
    parser.add_argument('--eps', type=float, default=10e-5)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the main function for the experiment
    main(args)
