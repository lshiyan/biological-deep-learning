import subprocess

# Create log
script_name = 'train_gen.py'

# Simulate the command line arguments
arguments = [
            # Basic configurations  
            "--data_name=MNIST",
            "--ext_data_name=EXT_MNIST",
            # Data Factory - MNIST
            "--train_data=data/mnist/train-images.idx3-ubyte", 
            "--train_label=data/mnist/train-labels.idx1-ubyte", 
            "--test_data=data/mnist/test-images.idx3-ubyte", 
            "--test_label=data/mnist/test-labels.idx1-ubyte",
            "--train_size=60000",
            "--test_size=10000",
            "--classes=10",
            # Data Factory - E-MNIST
            "--ext_train_data=data/ext_mnist/train-images.idx3-ubyte", 
            "--ext_train_label=data/ext_mnist/train-labels.idx1-ubyte", 
            "--ext_test_data=data/ext_mnist/test-images.idx3-ubyte", 
            "--ext_test_label=data/ext_mnist/test-labels.idx1-ubyte", 
            "--ext_train_size=88800",
            "--ext_test_size=14800",
            "--ext_classes=26",
            # CSV files generated - MNIST
            "--train_fname=data/mnist/mnist_train.csv",
            "--test_fname=data/mnist/mnist_test.csv",
            # CSV files generated - E-MNIST
            "--ext_train_fname=data/ext_mnist/ext_mnist_train.csv",
            "--ext_test_fname=data/ext_mnist/ext_mnist_test.csv",
            # Dimension of each layer
            '--input_dim=784', 
            '--heb_dim=64', 
            '--output_dim=10',
            # Hebbian layer hyperparameters  
            '--heb_lamb=15', 
            '--heb_gam=0.99',
            '--heb_eps=0.0001',
            '--sigmoid_k=1',
            '--learning_rule=sanger',
            '--inhibition_rule=relu',
            '--weight_growth=linear',
            '--weight_decay=simple',
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
            '--epochs=3', 
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