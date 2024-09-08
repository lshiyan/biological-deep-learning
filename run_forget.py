import subprocess

# Create log
script_name = 'train_forget.py'

# Simulate the command line arguments
arguments = [

            # Basic configurations  
            "--data_name=MNIST",
            "--experiment_name=TESTING_SOFT_HEBB",
            # Dataset parameters
            "--sub_experiment_scope_list=[[0,1],[2,3],[4,5],[6,7],[8,9]]",
            # Data Factory - MNIST
            "--train_data=data/mnist/train-images.idx3-ubyte", 
            "--train_label=data/mnist/train-labels.idx1-ubyte", 
            "--test_data=data/mnist/test-images.idx3-ubyte", 
            "--test_label=data/mnist/test-labels.idx1-ubyte",
            "--train_size=60000",
            "--test_size=10000",
            "--classes=10",
            "--train_fname=data/mnist/mnist_train.csv",
            "--test_fname=data/mnist/mnist_test.csv",
            # Dimension of each layer
            '--input_dim=784', 
            '--heb_dim=64', 
            '--output_dim=10',
            # Hebbian layer hyperparameters  
            '--heb_lamb=15', 
            '--heb_gam=0.99',
            '--heb_eps=0.0001',
            '--heb_learn=orthogonal',
            '--heb_inhib=relu',
            '--heb_growth=linear',
            '--heb_focus=neuron',
            '--heb_act=normalized',
            # Classification layer hyperparameters
            '--class_learn=OUTPUT_CONTRASTIVE',
            '--class_growth=exponential',
            '--class_bias=no_bias',
            '--class_focus=neuron',
            '--class_act=normalized',
            # Shared hyperparameters
            '--lr=0.005',
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