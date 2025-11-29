import subprocess
import sys

# Create log
script_name = 'train_gen.py'

# Get the current Python interpreter from the virtual environment
python_executable = sys.executable

# Simulate the command line arguments
arguments = [
            "--data_name=MNIST",
            "--ext_data_name=EXT_MNIST",
            "--experiment_name=_GEN_EXP_TEST_",
            "--train_data=data/mnist/train-images.idx3-ubyte",
            "--train_label=data/mnist/train-labels.idx1-ubyte",
            "--test_data=data/mnist/test-images.idx3-ubyte",
            "--test_label=data/mnist/test-labels.idx1-ubyte",
            "--train_size=60000",
            "--test_size=10000",
            "--classes=10",
            "--ext_train_data=data/ext_mnist/train-images.idx3-ubyte",
            "--ext_train_label=data/ext_mnist/train-labels.idx1-ubyte",
            "--ext_test_data=data/ext_mnist/test-images.idx3-ubyte",
            "--ext_test_label=data/ext_mnist/test-labels.idx1-ubyte",
            "--ext_train_size=88800",
            "--ext_test_size=14800",
            "--ext_classes=26",
            "--train_fname=data/mnist/mnist_train.csv",
            "--test_fname=data/mnist/mnist_test.csv",
            "--ext_train_fname=data/ext_mnist/ext_mnist_train.csv",
            "--ext_test_fname=data/ext_mnist/ext_mnist_test.csv",
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
            '--device=cpu',  # Use '--device=cuda:5' if using GPU
            '--local_machine=True',
            '--experiment_type=GENERALIZATION'
]

# Construct the command using the correct Python interpreter
command = [python_executable, script_name] + arguments

# Run the command
result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Print the output
print("Standard Output:\n", result.stdout)
print("Standard Error:\n", result.stderr)
