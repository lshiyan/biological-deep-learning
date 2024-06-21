import argparse

def parse_arguments(args_list=None):
    # Argument parser
    parser = argparse.ArgumentParser()

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
    parser.add_argument('--heb_lamb', type=float, default=15)
    parser.add_argument('--heb_gam', type=float, default=0.99)

    # Classification layer hyperparameters
    parser.add_argument('--cla_lamb', type=float, default=1)

    # Shared hyperparameters
    parser.add_argument('--eps', type=float, default=10e-5)

    # ---------------------------------------

    # The number of times to loop over the whole dataset
    parser.add_argument("--epochs", type=int, default=3)

    # Testing model performance on a test every "test-epochs" epochs
    parser.add_argument("--test_epochs", type=int, default=1)

    # A model training regularisation technique to reduce over-fitting
    parser.add_argument("--dropout", type=float, default=0.2)

    # This example demonstrates a StepLR learning rate scheduler. Different schedulers will require different hyper-parameters.
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lr_step_size", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=1)

    # ---------------------------------------
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device_id", type=str, default='cpu')
    parser.add_argument("--local_machine", type=bool, default=False)
    parser.add_argument("--save_dir", type=str, default='')
    parser.add_argument("--tboard_path", type=str, default='')

    # ---------------------------------------
    args = parser.parse_args() if args_list == None else parser.parse_args(args_list)

    return args