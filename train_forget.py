import multiprocessing
from typing import Tuple, List

from experiments.forget_experiment import ForgetExperiment
from interfaces.experiment import Experiment
from interfaces.network import Network
from models.hebbian_network import HebbianNetwork

from utils.experiment_parser import *
from utils.experiment_logger import *
from utils.experiment_stats import *
from models.SGD_network import SGDNetwork


# Create log
results_log = configure_logger('Forget Result Log', './results/results_forget.log')

# Get arguments
ARGS = parse_arguments()

# Main Code
def main():
    # Parallel training
    train_acc_lists, test_acc_lists = parallel_training(ARGS, 1)
    
    # Calculate and log averages and variances for each digit pair
    digit_pairs = [
        (0, 1, test_acc_lists[0], train_acc_lists[0]),
        (2, 3, test_acc_lists[1], train_acc_lists[1]),
        (4, 5, test_acc_lists[2], train_acc_lists[2]),
        (6, 7, test_acc_lists[3], train_acc_lists[3]),
        (8, 9, test_acc_lists[4], train_acc_lists[4])
    ]

    for digit_pair in digit_pairs:
        digit_1, digit_2, test_acc, train_acc = digit_pair
        avg_test = average(test_acc)
        var_test = variance(test_acc)
        avg_train = average(train_acc)
        var_train = variance(train_acc)
        results_log.info(f"Digits {digit_1} and {digit_2} || Epoch: {ARGS.epochs} || Lambda: {ARGS.heb_lamb} || Dataset: {ARGS.data_name.upper()} || Inhibition: {ARGS.heb_inhib.lower().capitalize()} || Learning Rule: {ARGS.heb_learn.lower().capitalize()} || Function Type: {ARGS.heb_growth.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Test Acc: avg = {avg_test}, var = {var_test} || Train Acc: avg = {avg_train}, var = {var_train}")


# Model Training
def train_and_eval(args: Tuple) -> List[List[float]]:
    params: argparse.Namespace
    num: int
    params, num = args
    if "sgd" == params.model.lower():
        model = SGDNetwork('SGD Network', params).to(params.device)
    elif "hebb" == params.model.lower():
        model = HebbianNetwork('Hebbian Network', params).to(params.device)
    experiment: Experiment = ForgetExperiment(model, params, f'-{params.experiment_name}-{params.experiment_type.lower()}-{params.heb_growth.lower()}-{params.heb_focus.lower()}-{params.heb_lamb}')
    accuracies = list(experiment.run())
    experiment.cleanup()

    # Return accuracies split by digit pairs
    return [
        accuracies[0:5],  # Test accuracies for digit pairs 0-1, 2-3, 4-5, 6-7, 8-9
        accuracies[5:10]  # Train accuracies for digit pairs 0-1, 2-3, 4-5, 6-7, 8-9
    ]


# Parallel Training
def parallel_training(params: argparse.Namespace, total: int) -> Tuple[List[List[float]], List[List[float]]]:
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=total) as pool:
        # Map the list of parameters to the function that performs training
        param_list = [
            (params, process_id)
            for process_id in range(total)
        ]
        results = pool.map(train_and_eval, param_list)
    
    # Split results into train and test accuracy lists for each digit pair
    test_acc_lists = [[] for _ in range(5)]
    train_acc_lists = [[] for _ in range(5)]

    for result in results:
        test_acc, train_acc = result
        for i in range(5):
            test_acc_lists[i].append(test_acc[i])
            train_acc_lists[i].append(train_acc[i])

    return train_acc_lists, test_acc_lists


# When to run code
if __name__ == "__main__":
    main()
    print("Process Completed.")