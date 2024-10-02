import subprocess
import logging
import sys

# Set up logging
logging.basicConfig(filename='bar_gen_experiment_results.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Define the script name for the Bar Generalization Experiment
script_name = 'train_bar_generalization.py'

# Define specific lambda, rho, and learning rate pairs for the Bar Generalization experiment
parameter_pairs = [
    (0.5, 0.03, 0.003), 
    (1, 0.1, 0.003),
    (2, 0.1, 0.003),
    (4, 0.03, 0.003),
    (8, 0.01, 0.003),
    (16, 0.01, 0.003),
    (32, 0.01, 0.003)
]

# Define other parameters to vary
other_parameters = [
    ('sanger', 'linear', 'linear', 'neuron', 'RELU', 'neuron'),
]

# Set the number of concurrent processes
max_concurrent_processes = 10

# Get the current Python interpreter from the virtual environment
python_executable = sys.executable

# Specify the GPU ID (or set to CPU)
gpu_id = 0

# Process the combinations in batches
for i in range(0, len(parameter_pairs), max_concurrent_processes):
    processes = []
    batch = parameter_pairs[i:i + max_concurrent_processes]
    
    for lmbda, rho, lr in batch:
        for heb_learn, heb_growth, clas_growth, heb_focus, heb_inhib, class_focus in other_parameters:

            # Construct the complete set of arguments including the varying parameter
            arguments = [
                '--data_name=BAR_DATASET',
                '--experiment_name=_TESTING_BAR_GEN_EXP_',
                '--train_fname=data/bar_matrix/bar_train.csv',
                '--test_fname=data/bar_matrix/bar_test.csv',

                '--data_matrix_size=6',
                '--bar_data_quantity=5',

                '--input_dim=36',
                '--heb_dim=16',
                '--output_dim=12',

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
                '--experiment_type=bar_generalization'
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