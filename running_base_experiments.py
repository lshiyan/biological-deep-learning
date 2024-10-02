import subprocess
import itertools
import logging
import sys

# Set up logging
logging.basicConfig(filename='experiment_results.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Define the base script name
script_name = 'train_base.py'


# Define specific lambda, rho, and learning rate pairs

"""
# Linear Linear
parameter_pairs = [
    (0.5, 0.03), 
    (1, 0.1),
    (2, 0.1),
    (4, 0.03),
    (8, 0.01),
    (16, 0.01),
    (32, 0.01)
]

# Exp Linear Neuron
parameter_pairs = [
    (0.5, 1), 
    (1, 1),
    (2, 1),
    (4, 1),
    (8, 1),
    (16, 1),
    (32, 0.3),
    (64, 0.1)
]


# Sigmoid Linear Neuron
parameter_pairs = [
    (0.5, 1), 
    (1, 0.3),
    (2, 0.1),
    (4, 0.1),
    (8, 0.1),
    (16, 0.003),
    (32, 0.01),
    (64, 0.001)
]

# Sigmoid Sigmoid Neuron
parameter_pairs = [
    (0.5, 0.3), 
    (1, 0.3),
    (2, 0.3),
    (4, 0.1),
    (8, 0.1),
    (16, 0.003),
    (32, 0.01),
    (64, 0.003)
]

# Sigmoid Linear Synapse
parameter_pairs = [
    (0.5, 1), 
    (1, 1),
    (2, 1),
    (4, 1),
    (8, 0.1),
    (16, 1),
    (32, 0.1),
    (64, 0.03)
]

# Sigmoid Sigmoid Synapse
parameter_pairs = [
    (0.5, 0.1), 
    (1, 1),
    (2, 0.3),
    (4, 0.1),
    (8, 0.1),
    (16, 1),
    (32, 1),
    (64, 0.3)
]
"""


# Sigmoid Sigmoid Neuron
parameter_pairs = [
    (0.5, 0.3), 
    (1, 0.3),
    (2, 0.3),
    (4, 0.1),
    (8, 0.1),
    (16, 0.003),
    (32, 0.01),
    (64, 0.003)
]

# Define the learning rates to test
# 0.1, 0.3, 0.01, 0.03, 0.001
learning_rates = [0.001]

# Define other parameters to vary
other_parameters = [
    ('sanger', 'sigmoid', 'linear', 'neuron', 'RELU', 'neuron'),
]

# Set the number of concurrent processes
max_concurrent_processes = 10

# Get the current Python interpreter from the virtual environment
python_executable = sys.executable

# Specify the GPU ID (e.g., GPU 0)
gpu_id = 0

# Process the combinations in batches
for i in range(0, len(parameter_pairs), max_concurrent_processes):
    processes = []
    batch = parameter_pairs[i:i + max_concurrent_processes]
    
    for lmbda, rho in batch:
        for lr in learning_rates:  # Loop over each learning rate
            for heb_learn, heb_growth, clas_growth, heb_focus, heb_inhib, class_focus in other_parameters:

                # Construct the complete set of arguments including the varying parameter
                arguments = [
                    '--data_name=MNIST',
                    '--experiment_name=_BASE_NEURON_SIGMOID_LINEAR_',
                    '--train_data=data/mnist/train-images.idx3-ubyte',
                    '--train_label=data/mnist/train-labels.idx1-ubyte',
                    '--test_data=data/mnist/test-images.idx3-ubyte',
                    '--test_label=data/mnist/test-labels.idx1-ubyte',
                    '--train_size=60000',
                    '--test_size=10000',
                    '--classes=10',
                    '--train_fname=data/mnist/mnist_train.csv',
                    '--test_fname=data/mnist/mnist_test.csv',
                    '--input_dim=784',
                    '--heb_dim=64',
                    '--output_dim=10',
                    '--heb_gam=0.99',
                    '--heb_eps=0.0001',
                    f'--heb_inhib={heb_inhib}',
                    f'--heb_focus={heb_focus}',
                    f'--heb_growth={heb_growth}',
                    f'--heb_learn={heb_learn}',
                    f'--heb_lamb={lmbda}',
                    f'--heb_rho={rho}',
                    '--heb_act=normalized',
                    '--class_learn=SUPERVISED_HEBBIAN',
                    f'--class_growth={clas_growth}',
                    '--class_bias=no_bias',
                    f'--class_focus={class_focus}',
                    '--class_act=normalized',
                    f'--lr={lr}',
                    '--sigmoid_k=1',
                    '--alpha=0',
                    '--beta=0.01',
                    '--sigma=1',
                    '--mu=0',
                    '--init=uniform',
                    '--batch_size=1',
                    '--epochs=10',
                    f'--device=cuda:{gpu_id}',  # Use the specified GPU or CPU
                    '--local_machine=True',
                    '--experiment_type=base'
                ]

                # Construct the command with nice
                command = ['nice', '-n', '1', python_executable, script_name] + arguments

                try:
                    # Start the process
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    processes.append(process)
                    logging.info(f"Started process with PID: {process.pid} on GPU: {gpu_id} | Lambda: {lmbda}, Rho: {rho}, LR: {lr}, Heb_Learn: {heb_learn}, Heb_Growth: {heb_growth}, Clas_Growth: {clas_growth}, Heb_Focus: {heb_focus}, Heb_Inhib: {heb_inhib}, Class_Focus: {class_focus}")
                except Exception as e:
                    logging.error(f"Failed to start process for combination: Lambda={lmbda}, Rho={rho}, LR={lr}, Heb_Learn={heb_learn}, Heb_Growth={heb_growth}, Clas_Growth={clas_growth}, Heb_Focus={heb_focus}, Heb_Inhib={heb_inhib}, Class_Focus={class_focus}. Error: {str(e)}")

    # Wait for all processes in the batch to finish before starting the next batch
    for process in processes:
        try:
            stdout, stderr = process.communicate()
            logging.info(f"Process with PID: {process.pid} completed on GPU: {gpu_id}.")
            logging.info("Standard Output:\n" + stdout)
            if stderr:
                logging.error("Standard Error:\n" + stderr)
        except Exception as e:
            logging.error(f"Failed to complete process with PID: {process.pid}. Error: {str(e)}")

logging.info("All subprocesses have completed.")