import subprocess
import logging
import sys

# Set up logging
logging.basicConfig(filename='experiment_results_gen.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Define the base script name
script_name = 'train_gen.py'


"""
# Linear Linear
parameter_pairs = [
    (0.5, 0.03, 0.003), 
    (1, 0.1, 0.003),
    (2, 0.1, 0.003),
    (4, 0.03, 0.003),
    (8, 0.01, 0.003),
    (16, 0.01, 0.003),
    (32, 0.01, 0.003),
    (64, 0.01, 0.003)
]

# Exp Exp Neuron
parameter_pairs = [
    (0.5, 1, 0.001), 
    (1, 1, 0.001),
    (2, 1, 0.03),
    (4, 1, 0.001),
    (8, 1, 0.001),
    (16, 1, 0.01),
    (32, 0.3, 0.03),
    (64, 0.1, 0.03)
]

# Exp Linear Neuron
parameter_pairs = [
    (0.5, 1, 0.001), 
    (1, 1, 0.01),
    (2, 1, 0.001),
    (4, 1, 0.001),
    (8, 1, 0.001),
    (16, 1, 0.01),
    (32, 0.3, 0.1),
    (64, 0.1, 0.1)
]


# Sigmoid Linear Neuron
parameter_pairs = [
    (0.5, 1, 0.01), 
    (1, 0.3, 0.01),
    (2, 0.1, 0.01),
    (4, 0.1, 0.01),
    (8, 0.1, 0.03),
    (16, 0.003, 0.3),
    (32, 0.01, 0.01),
    (64, 0.001, 0.1)
]

# Sigmoid Sigmoid Neuron
parameter_pairs = [
    (0.5, 0.3, 0.1), 
    (1, 0.3, 0.03),
    (2, 0.3, 0.001),
    (4, 0.1, 0.01),
    (8, 0.1, 0.1),
    (16, 0.003, 0.001),
    (32, 0.01, 0.001),
    (64, 0.003, 0.001)
]

# Sigmoid Linear Synapse
parameter_pairs = [
    (0.5, 1, 0.001), 
    (1, 1, 0.001),
    (2, 1, 0.01),
    (4, 1, 0.03),
    (8, 0.1, 0.001),
    (16, 1, 0.01),
    (32, 0.1, 0.3),
    (64, 0.03, 0.03)
]

# Sigmoid Sigmoid Synapse
parameter_pairs = [
    (0.5, 0.1, 0.001), 
    (1, 1, 0.001),
    (2, 0.3, 0.03),
    (4, 0.1, 0.1),
    (8, 0.1, 0.001),
    (16, 1, 0.1),
    (32, 1, 0.1),
    (64, 0.3, 0.3)
]
"""


# Exp Exp Neuron
parameter_pairs = [
    (8, 1, 0.001),
]

# Define other parameters to vary
other_parameters = [
    ('sanger', 'exponential', 'exponential', 'neuron', 'RELU', 'neuron'),
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
    
    for lmbda, rho, lr in batch:
        for heb_learn, heb_growth, clas_growth, heb_focus, heb_inhib, class_focus in other_parameters:

            # Construct the complete set of arguments including the varying parameters
            arguments = [
                '--data_name=MNIST',
                '--ext_data_name=EXT_MNIST',
                '--experiment_name=_GEN_NEURON_EXP_EXP_',
                '--train_data=data/mnist/train-images.idx3-ubyte',
                '--train_label=data/mnist/train-labels.idx1-ubyte',
                '--test_data=data/mnist/test-images.idx3-ubyte',
                '--test_label=data/mnist/test-labels.idx1-ubyte',
                '--train_size=60000',
                '--test_size=10000',
                '--classes=10',
                '--ext_train_data=data/ext_mnist/train-images.idx3-ubyte',
                '--ext_train_label=data/ext_mnist/train-labels.idx1-ubyte',
                '--ext_test_data=data/ext_mnist/test-images.idx3-ubyte',
                '--ext_test_label=data/ext_mnist/ext_test-labels.idx1-ubyte',
                '--ext_train_size=88800',
                '--ext_test_size=14800',
                '--ext_classes=26',
                '--train_fname=data/mnist/mnist_train.csv',
                '--test_fname=data/mnist/mnist_test.csv',
                '--ext_train_fname=data/ext_mnist/ext_mnist_train.csv',
                '--ext_test_fname=data/ext_mnist/ext_mnist_test.csv',
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
                '--beta=10e-2',
                '--sigma=1',
                '--mu=0',
                '--init=uniform',
                '--batch_size=1',
                '--epochs=1',
                f'--device=cuda:{gpu_id}',  # Use the specified GPU
                '--local_machine=True',
                '--experiment_type=generalization'
            ]

            # Construct the command with nice
            command = ['nice', '-n', '1', python_executable, script_name] + arguments

            try:
                # Start the process
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                processes.append(process)
                logging.info(f"Started process with PID: {process.pid} on GPU: {gpu_id} | Lambda: {lmbda}, LR: {lr}, Heb_Learn: {heb_learn}, Heb_Growth: {heb_growth}, Clas_Growth: {clas_growth}, Heb_Focus: {heb_focus}, Heb_Inhib: {heb_inhib}, Class_Focus: {class_focus}")
            except Exception as e:
                logging.error(f"Failed to start process for combination: Lambda={lmbda}, LR={lr}, Heb_Learn={heb_learn}, Heb_Growth={heb_growth}, Clas_Growth={clas_growth}, Heb_Focus={heb_focus}, Heb_Inhib={heb_inhib}, Class_Focus={class_focus}. Error: {str(e)}")

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