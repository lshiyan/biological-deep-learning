import multiprocessing
from typing import Tuple, List

from experiments.colour_mnist_generalization_experiment import ColouredMnistExperiment  # Import your custom experiment class
from interfaces.experiment import Experiment
from interfaces.network import Network
from models.hebbian_network import HebbianNetwork

from utils.experiment_parser import *
from utils.experiment_comparer import *
from utils.experiment_logger import *
from utils.experiment_timer import *
from utils.experiment_stats import *

# Create log
results_log = configure_logger('Coloured MNIST Result Log', './results/coloured_mnist_results.log')

# Get arguments
ARGS = parse_arguments()

# Main Code
def main():
    train_acc_list, test_acc_list = parallel_training(ARGS, 1)
                            
    avg_test = average(test_acc_list)
    var_test = variance(test_acc_list)
    avg_train = average(train_acc_list)
    var_train = variance(train_acc_list)

    results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {ARGS.heb_lamb} || LR: {ARGS.lr} || EPS: {ARGS.heb_eps} || Dim: {ARGS.heb_dim} || Dataset: {ARGS.data_name.upper()} || Learning Rule: {ARGS.heb_learn.lower().capitalize()}/{ARGS.class_learn.lower().capitalize()} || Inhibition Type: {ARGS.heb_inhib.lower().capitalize()} || Function Type: {ARGS.heb_growth.lower().capitalize()}/{ARGS.class_growth.lower().capitalize()} || Focus: {ARGS.heb_focus.lower().capitalize()}/{ARGS.class_focus.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Test Acc: avg = {avg_test}, var = {var_test} || Train Acc: avg = {avg_train}, var = {var_train}")


# Model Training
def train_and_eval(args: Tuple) -> List[float]:
    params: argparse.Namespace
    num: int
    params, num = args
    model: Network = HebbianNetwork('Hebbian Network', params).to(params.device)
    experiment: Experiment = ColouredMnistExperiment(model, params, f'-{params.experiment_name}-{params.experiment_type.lower()}-{params.lr}--{params.heb_learn.lower()}-{params.heb_growth.lower()}-{params.heb_focus.lower()}-{params.heb_inhib.lower()}-{params.heb_lamb}---{params.class_learn.lower()}-{params.class_growth.lower()}-{params.class_focus.lower()}-{num}')
    
    accuracies: List[float] = list(experiment.run())  # Collect all returned accuracies
    
    experiment.cleanup()
    
    # Return all accuracies for later processing
    return accuracies


def parallel_training(params: argparse.Namespace, total: int) -> Tuple[List[float], List[float]]:
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=total) as pool:
        param_list = [(params, process_id) for process_id in range(total)]
        results = pool.map(train_and_eval, param_list)
    
    # Extract the first accuracy as train accuracy
    train_acc_list = [result[0] for result in results]
    
    # The second accuracy is the main test accuracy (but could be extended if more test sets are used)
    test_acc_list = [result[1] for result in results]
    
    # You could extend this to handle more test sets:
    # additional_test_acc_lists = [[result[i] for result in results] for i in range(2, len(results[0]))]

    return train_acc_list, test_acc_list


# When to run code
if __name__ == "__main__":
    main()
    print("Process Completed.")