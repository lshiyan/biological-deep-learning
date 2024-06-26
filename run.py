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
            '--eps=0.01', 
            '--epochs=1', 
            '--test_epochs=1', 
            '--lr=0.005',  
            '--batch_size=1',
            '--device_id=cpu',
            '--local_machine=True' ]

# Construct the command
command = ['python', script_name] + arguments

# Run the command
result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Print the output
print("Standard Output:\n", result.stdout)
print("Standard Error:\n", result.stderr)