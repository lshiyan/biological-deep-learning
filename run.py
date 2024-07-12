import subprocess

# Create log
script_name = 'train.py'

# Simulate the command line arguments
arguments = [
            # Basic configurations  
            "--data_name=MNIST",
            "--e_data_name=E_MNIST",
            # Data Factory - MNIST
            "--mnist_train_data=data/mnist/train-images.idx3-ubyte", 
            "--mnist_train_label=data/mnist/train-labels.idx1-ubyte", 
            "--mnist_test_data=data/mnist/test-images.idx3-ubyte", 
            "--mnist_test_label=data/mnist/test-labels.idx1-ubyte",
            "--mnist_train_size=60000",
            "--mnist_test_size=10000",
            "--mnist_classes=10",
            # Data Factory - FASHION-MNIST
            "--fashion_mnist_train_data=data/fashion_mnist/train-images.idx3-ubyte", 
            "--fashion_mnist_train_label=data/fashion_mnist/train-labels.idx1-ubyte", 
            "--fashion_mnist_test_data=data/fashion_mnist/test-images.idx3-ubyte", 
            "--fashion_mnist_test_label=data/fashion_mnist/test-labels.idx1-ubyte", 
            "--fashion_mnist_train_size=60000",
            "--fashion_mnist_test_size=10000",
            "--fashion_mnist_classes=10",
            # Data Factory - E-MNIST
            "--e_mnist_train_data=data/e_mnist/train-images.idx3-ubyte", 
            "--e_mnist_train_label=data/e_mnist/train-labels.idx1-ubyte", 
            "--e_mnist_test_data=data/e_mnist/test-images.idx3-ubyte", 
            "--e_mnist_test_label=data/e_mnist/test-labels.idx1-ubyte", 
            "--e_mnist_train_size=88800",
            "--e_mnist_test_size=14800",
            "--e_mnist_classes=26",
            # CSV files generated - MNIST
            "--mnist_train_fname=data/mnist/mnist_train.csv",
            "--mnist_test_fname=data/mnist/mnist_test.csv",
            # CSV files generated - FASHION-MNIST
            "--e_mnist_train_fname=data/e_mnist/e_mnist_train.csv",
            "--e_mnist_test_fname=data/e_mnist/e_mnist_test.csv",
            # CSV files generated - E-MNIST
            "--fashion_mnist_train_fname=data/fashion_mnist/fashion_mnist_train.csv",
            "--fashion_mnist_test_fname=data/fashion_mnist/fashion_mnist_test.csv",
            # Dimension of each layer
            '--input_dim=784', 
            '--heb_dim=64', 
            '--output_dim=26',
            # Hebbian layer hyperparameters  
            '--heb_lamb=15', 
            '--heb_gam=0.99',
            '--heb_eps=0.01',
            '--sigmoid_k=1',
            '--learning_rule=sanger',
            '--inhibition_rule=relu',
            '--weight_growth=linear',
            '--weight_decay=tanh',
            '--bias_update=no_bias',
            # Classification layer hyperparameters
            '--include_first=False',
            # Shared hyperparameters
            '--lr=0.005',
            '--alpha=0',
            '--beta=1',
            '--sigma=1',
            '--mu=0',
            '--init=uniform',
            # Experiment parameters
            '--batch_size=1',
            '--epochs=1', 
            '--device=cpu',
            '--local_machine=True',
            '--experiment_type=generalization'
            ]

# Construct the command
command = ['python', script_name] + arguments

# Run the command
result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Print the output
print("Standard Output:\n", result.stdout)
print("Standard Error:\n", result.stderr) 