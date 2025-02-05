import subprocess
import logging
import sys
import itertools

# Set up logging
logging.basicConfig(filename='experiment_softhebb_results.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Define the base script name
script_name = 'train_forget.py'

# Define available GPUs
available_gpus = [1]
gpu_cycle = itertools.cycle(available_gpus)

# Define batch sizes and neuron sizes to vary
batch_sizes = [16]
hidden_sizes = [1024]

# Define parameter pairs (lambda, rho, learning rate)
parameter_pairs = [(0.5, 1, 0.003)]

# Define other parameters
other_parameters = [
    ('sanger', 'sigmoid', 'sigmoid', 'neuron', 'RELU', 'neuron')
]

# Set the number of concurrent processes
max_concurrent_processes = len(available_gpus)

# Get the current Python interpreter from the virtual environment
python_executable = sys.executable

# Process combinations
processes = []

for batch_size, hsize in itertools.product(batch_sizes, hidden_sizes):
    for lmbda, rho, lr in parameter_pairs:
        for heb_learn, heb_growth, clas_growth, heb_focus, heb_inhib, class_focus in other_parameters:
            # Assign GPU in a round-robin manner
            gpu_id = next(gpu_cycle)
            
            # Construct experiment name
            exp_name = f"SOFTHEBB_BATCH{batch_size}_HSIZE{hsize}_{heb_growth.upper()}_{clas_growth.upper()}"

            # Construct the command arguments
            arguments = [
                '--data_name=MNIST',
                f'--experiment_name={exp_name}',
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
                f'--heb_dim={hsize}',
                '--output_dim=10',
                '--heb_gam=0.99',
                '--heb_eps=0.0001',
                '--sub_experiment_scope_list=[[0,1],[2,3],[4,5],[6,7],[8,9]]',
                f'--heb_inhib={heb_inhib}',
                f'--heb_focus={heb_focus}',
                f'--heb_growth={heb_growth}',
                f'--heb_learn={heb_learn}',
                f'--heb_lamb={lmbda}',
                f'--heb_rho={rho}',
                '--heb_act=normalized',
                '--class_learn=OUTPUT_CONTRASTIVE',
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
                '--w_lr=0.003',
                '--l_lr=0.003',
                '--b_lr=0.003',
                '--init=uniform',
                f'--hsize={hsize}',
                f'--batch_size={batch_size}',
                '--epochs=10',
                f'--device=cuda:{gpu_id}',
                '--local_machine=True',
                '--experiment_type=forget'
            ]

            # Construct the command
            command = ['nice', '-n', '1', python_executable, script_name] + arguments

            try:
                # Start the process
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                processes.append((process, gpu_id))
                logging.info(f"Started process with PID: {process.pid} on GPU: {gpu_id} | Exp: {exp_name}")
            except Exception as e:
                logging.error(f"Failed to start process for Exp: {exp_name}. Error: {str(e)}")
            
            # Limit concurrent processes
            if len(processes) >= max_concurrent_processes:
                # Wait for one process to complete
                finished_process, finished_gpu = processes.pop(0)
                stdout, stderr = finished_process.communicate()
                logging.info(f"Process with PID: {finished_process.pid} on GPU: {finished_gpu} completed.")
                if stderr:
                    logging.error(f"Standard Error for PID {finished_process.pid}:\n{stderr}")

# Wait for remaining processes
for process, gpu_id in processes:
    stdout, stderr = process.communicate()
    logging.info(f"Process with PID: {process.pid} on GPU: {gpu_id} completed.")
    if stderr:
        logging.error(f"Standard Error for PID {process.pid}:\n{stderr}")

logging.info("All experiments have completed.")
