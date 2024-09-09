import subprocess

# Create log
script_name = 'train_forget.py'

# Simulate the command line arguments
base_arguments = [
    # Model
    "--model=SGD",
    #"--model=Hebb",
    # Basic configurations  
    '--experiment_name=_FORGET_SGD_MNIST_LINEAR_',
    # Dataset parameters
    "--sub_experiment_scope_list=[[0,1],[2,3],[4,5],[6,7],[8,9]]",
    # Data Factory - MNIST
    "--data_name=MNIST",
    "--train_data=data/mnist/train-images-idx3-ubyte",
    "--train_label=data/mnist/train-labels-idx1-ubyte",
    "--test_data=data/mnist/t10k-images-idx3-ubyte",
    "--test_label=data/mnist/t10k-labels-idx1-ubyte",
    "--train_size=60000",
    "--test_size=10000",
    "--classes=10",
    "--train_fname=data/mnist/mnist_train.csv",
    "--test_fname=data/mnist/mnist_test.csv",
    # '--data_name=FASHION_MNIST',
    # '--experiment_name=_FMNIST_EXPERIMENTS_TEST',
    # '--train_data=data/fashion_mnist/train-images-idx3-ubyte',
    # '--train_label=data/fashion_mnist/train-labels-idx1-ubyte',
    # '--test_data=data/fashion_mnist/t10k-images-idx3-ubyte',
    # '--test_label=data/fashion_mnist/t10k-labels-idx1-ubyte',
    # '--train_size=60000',
    # '--test_size=10000',
    # '--classes=10',
    # '--train_fname=data/fashion_mnist/fashion-mnist_train.csv',
    # '--test_fname=data/fashion_mnist/fashion-mnist_test.csv',
    # Dimension of each layer
    '--input_dim=784', 
    '--heb_dim=64', 
    '--output_dim=10',
    # Hebbian layer hyperparameters  
    '--heb_gam=0.99',
    '--heb_eps=0.0001',
    '--heb_learn=sanger',
    '--heb_inhib=relu',
    '--heb_growth=sigmoid',
    '--heb_focus=neuron',
    '--heb_act=normalized',
    # Classification layer hyperparameters
    '--class_learn=SUPERVISED_HEBBIAN',
    '--class_growth=linear',
    '--class_bias=no_bias',
    '--class_focus=neuron',
    '--class_act=normalized',
    # Shared hyperparameters
    '--lr=0.03',
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

# List of lambda values to iterate over
lambda_values = [45]

for lamb in lambda_values:
    # Add the lambda value to the arguments
    arguments = base_arguments + [f'--heb_lamb={lamb}']
    
    # Construct the command
    command = ['python', script_name] + arguments

    # Run the command
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Print the output
    print(f"Running with lambda = {lamb}")
    print("Standard Output:\n", result.stdout)
    print("Standard Error:\n", result.stderr)