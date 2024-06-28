import subprocess

# Create log
script_name = 'train.py'

# Simulate the command line arguments
arguments = ['--is_training=True', 
            "--data_name='MNIST'",
            "--train_data=data/mnist/train-images.idx3-ubyte", 
            "--train_label=data/mnist/train-labels.idx1-ubyte", 
            "--test_data=data/mnist/t10k-images.idx3-ubyte", 
            "--test_label=data/mnist/t10k-labels.idx1-ubyte", 
            "--train_filename=data/mnist/mnist_train.csv",
            "--test_filename=data/mnist/mnist_test.csv",
            '--input_dim=784', 
            '--heb_dim=64', 
            '--output_dim=10',  
            '--heb_lamb=15', 
            '--heb_gam=0.99',
            '--heb_eps=0.01', 
            '--epochs=1', 
            '--test_epochs=1', 
            '--lr=0.005',  
            '--batch_size=1',
            '--device=cpu',
            '--local_machine=True',
            '--extended_testing_mode=True',
            "--out_distribution_train_data=data/emnist/emnist-bymerge-train-images-idx3-ubyte",
            "--out_distribution_train_label=data/emnist/emnist-bymerge-train-labels-idx1-ubyte",
            "--out_distribution_train_filename=data/emnist/emnist_train.csv",
            "--out_distribution_test_data=data/emnist/emnist-bymerge-test-images-idx3-ubyte",
            "--out_distribution_test_label=data/emnist/emnist-bymerge-test-labels-idx1-ubyte",
            "--out_distribution_test_filename=data/emnist/emnist_test.csv"]

# Construct the command
command = ['python', script_name] + arguments

# Run the command
result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Print the output
print("Standard Output:\n", result.stdout)
print("Standard Error:\n", result.stderr)