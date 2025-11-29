import subprocess
import itertools
import logging
import sys

# Set up logging
logging.basicConfig(filename='experiment_results.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Define the base script name
script_name = 'train_base.py'

# 0.5, 1, 2, 4, 8, 16, 32, 64, 128

# Define the values for the parameters you want to vary
lambda_values = [0.5, 1, 2, 4, 8, 16, 32, 64, 128]
heb_learn_values = ['orthogonal']
heb_growth_values = [ 'exponential']
clas_growth_values = ['exponential']
heb_focus_values = ['neuron']
heb_inhib_values = ['RELU']
class_focus_values = ['neuron']
learning_rate_values = [0.003]

# Generate all possible combinations of parameters
parameter_combinations = list(itertools.product(
    lambda_values,
    heb_learn_values,
    heb_growth_values,
    clas_growth_values,
    heb_focus_values,
    heb_inhib_values,
    class_focus_values,
    learning_rate_values
))

# Set the number of concurrent processes to 8
max_concurrent_processes = 10

# Get the current Python interpreter from the virtual environment
python_executable = sys.executable

# Specify the GPU ID (e.g., GPU 0)
gpu_id = 3

# Process the combinations in batches
for i in range(0, len(parameter_combinations), max_concurrent_processes):
    processes = []
    batch = parameter_combinations[i:i + max_concurrent_processes]
    
    for combination in batch:
        lmbda, heb_learn, heb_growth, clas_growth, heb_focus, heb_inhib, class_focus, lr = combination

        # Construct the complete set of arguments including the varying parameter
        arguments = [
            '--data_name=MNIST',
            '--experiment_name=_TEACHER2_BASE_NEURON_EXP_EXP_',
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
            '--heb_act=normalized',
            '--class_learn=OUTPUT_CONTRASTIVE',
            f'--class_growth={clas_growth}',
            '--class_bias=no_bias',
            f'--class_focus={class_focus}',  # Use the classification focus parameter here
            '--class_act=normalized',
            f'--lr={lr}',
            '--sigmoid_k=1',
            '--alpha=0',
            '--beta=1e-2',
            '--sigma=1',
            '--mu=0',
            '--init=uniform',
            '--batch_size=1',
            '--epochs=10',
            f'--device=cuda:{gpu_id}',
            '--local_machine=True',
            '--experiment_type=base'
        ]

        # Construct the command with nice
        command = ['nice', '-n', '4', python_executable, script_name] + arguments

        try:
            # Start the process
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            processes.append(process)
            logging.info(f"Started process with PID: {process.pid} on GPU: {gpu_id} | Lambda: {lmbda}, LR: {lr}, Heb_Learn: {heb_learn}, Heb_Growth: {heb_growth}, Class_Growth: {clas_growth}, Heb_Focus: {heb_focus}, Heb_Inhib: {heb_inhib}, Class_Focus: {class_focus}")
        except Exception as e:
            logging.error(f"Failed to start process for combination: Lambda={lmbda}, LR={lr}, Heb_Learn={heb_learn}, Heb_Growth={heb_growth}, Class_Growth={clas_growth}, Heb_Focus={heb_focus}, Heb_Inhib={heb_inhib}, Class_Focus={class_focus}. Error: {str(e)}")

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