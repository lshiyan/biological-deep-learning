import subprocess

# Create log
script_name = 'train.py'

# Simulate the command line arguments
arguments = [
            # Basic configurations 
            '--is_training=True', 
            "--data_name='MNIST'",
            # Data Factory
            "--train_data=data/mnist/train-images.idx3-ubyte", 
            "--train_label=data/mnist/train-labels.idx1-ubyte", 
            "--test_data=data/mnist/t10k-images.idx3-ubyte", 
            "--test_label=data/mnist/t10k-labels.idx1-ubyte", 
            # CSV files generated
            "--train_filename=data/mnist/mnist_train.csv",
            "--test_filename=data/mnist/mnist_test.csv",
            # Dimension of each layer
            '--input_dim=784', 
            '--heb_dim=64', 
            '--output_dim=10',
            # Hebbian layer hyperparameters  
            '--heb_lamb=15', 
            '--heb_gam=0.99',
            '--heb_eps=0.01',
            '--learning_rule=sanger',
            '--inhibition_rule=relu',
            '--function_type=linear',
            # Classification layer hyperparameters
            '--include_first=True',
             # Shared hyperparameters
            '--lr=0.005',
            # Experiment parameters
            '--batch_size=1',
            '--epochs=1', 
            '--test_epochs=1',
            '--device=cpu',
            '--local_machine=True' ]

# Construct the command
command = ['python', script_name] + arguments

# Run the command
result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Print the output
print("Standard Output:\n", result.stdout)
print("Standard Error:\n", result.stderr)