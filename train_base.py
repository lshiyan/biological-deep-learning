import multiprocessing
from typing import Tuple
from experiments.base_experiment import BaseExperiment

from interfaces.experiment import Experiment
from interfaces.network import Network
from models.hebbian_network import HebbianNetwork
from utils.experiment_parser import *
from utils.experiment_comparer import *
from utils.experiment_logger import *
from utils.experiment_testing import eps_list, dim_list, lambda_list, lr_list, sigmoid_k_list
from utils.experiment_timer import *
from utils.experiment_stats import *

# Create log
results_log = configure_logger('Base Result Log', './results/results.log')

# Get arguments
ARGS = parse_arguments()

# Experiments setup
lambda_test = lambda_list(1)
lr_test = lr_list(0.005)
eps_test = eps_list(0.0001)
dim_test = dim_list(64)
sigmoid_k_test = sigmoid_k_list(1)


def main():
    train_acc_list, test_acc_list = parallel_training(ARGS, 5)
                            
    avg_test = round(average(test_acc_list), 4)
    var_test = round(variance(test_acc_list), 6)
    avg_train = round(average(train_acc_list), 4)
    var_train = round(variance(train_acc_list), 6)

    results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {ARGS.heb_lamb} || LR: {ARGS.lr} || EPS: {ARGS.heb_eps} || Dim: {ARGS.heb_dim} || Sigmoid K: {ARGS.sigmoid_k} || Dataset: {ARGS.data_name.upper()} || Inhibition: {ARGS.heb_inhib.lower().capitalize()} || Learning Rule: {ARGS.heb_learn.lower().capitalize()} || Function Type: {ARGS.heb_growth.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Test Acc: avg = {avg_test}, var = {var_test} || Train Acc: avg = {avg_train}, var = {var_train}")


def train_and_eval(args: Tuple) -> List[float]:
    params: argparse.Namespace
    num: int
    params, num = args
    model: Network = HebbianNetwork('Hebbian Network', params).to(params.device)
    experiment: Experiment = BaseExperiment(model, params, f'{params.experiment_type.lower()}-{params.heb_learn.lower()}-{params.heb_inhib.lower()}-{params.heb_growth.lower()}-{params.heb_bias.lower()}-{params.heb_lamb}-{params.lr}-{params.heb_eps}-{params.heb_dim}-{params.sigmoid_k}-{num}')
    accuracies: List[float] = list(experiment.run())
    experiment.cleanup()
    
    return accuracies


def parallel_training(params: argparse.Namespace, total: int) -> Tuple[List[float], List[float]]:
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=total) as pool:
        # Map the list of parameters to the function that performs training
        param_list = [
            (params, process_id)
            for process_id in range(total)
        ]
        results = pool.map(train_and_eval, param_list)
    
    # Split results into train and test accuracy lists
    train_acc_list = [round(result[0], 4) for result in results]
    test_acc_list = [round(result[1], 4) for result in results]
    
    return train_acc_list, test_acc_list


if __name__ == "__main__":
    main()