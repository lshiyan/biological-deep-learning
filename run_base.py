import subprocess

# Create log
script_name = 'train_base.py'

# Simulate the command line arguments
arguments = [
            # Basic configurations  
            "--data_name=MNIST",
            "--experiment_name=LR0.0001",
            # Data Factory - MNIST
            "--train_data=data/mnist/train-images.idx3-ubyte", 
            "--train_label=data/mnist/train-labels.idx1-ubyte", 
            "--test_data=data/mnist/test-images.idx3-ubyte", 
            "--test_label=data/mnist/test-labels.idx1-ubyte",
            "--train_size=60000",
            "--test_size=10000",
            "--classes=10",
            # # Data Factory - FASHION-MNIST
            # "--train_data=data/fashion_mnist/train-images.idx3-ubyte", 
            # "--train_label=data/fashion_mnist/train-labels.idx1-ubyte", 
            # "--test_data=data/fashion_mnist/test-images.idx3-ubyte", 
            # "--test_label=data/fashion_mnist/test-labels.idx1-ubyte", 
            # "--train_size=60000",
            # "--test_size=10000",
            # "--classes=10",
            # # Data Factory - E-MNIST
            # "--train_data=data/ext_mnist/train-images.idx3-ubyte", 
            # "--train_label=data/ext_mnist/train-labels.idx1-ubyte", 
            # "--test_data=data/ext_mnist/test-images.idx3-ubyte", 
            # "--test_label=data/ext_mnist/test-labels.idx1-ubyte", 
            # "--train_size=88800",
            # "--test_size=14800",
            # "--classes=26",
            # CSV files generated - MNIST
            "--train_fname=data/mnist/mnist_train.csv",
            "--test_fname=data/mnist/mnist_test.csv",
            # # CSV files generated - E-MNIST
            # "--train_fname=data/ext_mnist/ext_mnist_train.csv",
            # "--test_fname=data/ext_mnist/ext_mnist_test.csv",
            # # CSV files generated - FASHION-MNIST
            # "--train_fname=data/fashion_mnist/fashion_mnist_train.csv",
            # "--test_fname=data/fashion_mnist/fashion_mnist_test.csv",
            # Dimension of each layer
            '--input_dim=784', 
            '--heb_dim=64', 
            '--output_dim=10',
            # Hebbian layer hyperparameters  
            '--heb_lamb=16', 
            '--heb_gam=0.99',
            '--heb_eps=0.0001',
            '--heb_learn=sanger',
            '--heb_inhib=relu',
            '--heb_growth=linear',
            '--heb_focus=neuron',
            '--heb_act=normalized',
            # Classification layer hyperparameters
            '--class_learn=hebbian',
            '--class_growth=linear',
            '--class_bias=no_bias',
            '--class_focus=neuron',
            '--class_act=normalized',
            # Shared hyperparameters
            '--lr=0.0005',
            '--sigmoid_k=1',
            '--alpha=0',
            '--beta=1e-2',
            '--sigma=1',
            '--mu=0',
            '--init=uniform',
            # Experiment parameters
            '--batch_size=1',
            '--epochs=1', 
            '--device=cpu',
            # '--device=cuda:5',
            '--local_machine=True',
            '--experiment_type=base'
            ]

# Construct the command
command = ['python', script_name] + arguments

# Run the command
result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Print the output
print("Standard Output:\n", result.stdout)
print("Standard Error:\n", result.stderr) 
