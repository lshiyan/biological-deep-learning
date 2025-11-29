import multiprocessing
from typing import Tuple

from experiments.generalization_experiment import GeneralizationExperiment
from interfaces.experiment import Experiment
from interfaces.network import Network
from models.hebbian_network import HebbianNetwork

from utils.experiment_parser import *
from utils.experiment_comparer import *
from utils.experiment_logger import *
from utils.experiment_timer import *
from utils.experiment_stats import *

# Create log
results_log = configure_logger('Base Result Log', './results/results.log')


# Get arguments
ARGS = parse_arguments()


# Main Code
def main():
    cos_test_list = []
    cos_train_list = []
    norm_test_list = []
    norm_train_list = []
    
    cos_test_ext_list = []
    cos_train_ext_list = []
    norm_test_ext_list = []
    norm_train_ext_list = []

    freeze_train_acc_list = []
    freeze_test_acc_list = []
    
    [freeze_train_acc_list, 
    freeze_test_acc_list, 
    cos_train_list, 
    cos_test_list, 
    norm_train_list, 
    norm_test_list, 
    cos_train_ext_list,
    cos_test_ext_list,
    norm_train_ext_list,
    norm_test_ext_list] = parallel_training(ARGS, 1)
    
    avg_test = average(freeze_test_acc_list)
    var_test = variance(freeze_test_acc_list)
    avg_train = average(freeze_train_acc_list)
    var_train = variance(freeze_train_acc_list)
    
    avg_test_cos = average(cos_test_list)
    var_test_cos = variance(cos_test_list)
    avg_train_cos = average(cos_train_list)
    var_train_cos = variance(cos_train_list)
    
    avg_test_cos_ext = average(cos_test_ext_list)
    var_test_cos_ext = variance(cos_test_ext_list)
    avg_train_cos_ext = average(cos_train_ext_list)
    var_train_cos_ext = variance(cos_train_ext_list)
    
    avg_test_norm = average(norm_test_list)
    var_test_norm = variance(norm_test_list)
    avg_train_norm = average(norm_train_list)
    var_train_norm = variance(norm_train_list)
    
    avg_test_norm_ext = average(norm_test_ext_list)
    var_test_norm_ext = variance(norm_test_ext_list)
    avg_train_norm_ext = average(norm_train_ext_list)
    var_train_norm_ext = variance(norm_train_ext_list)

    results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {ARGS.heb_lamb} || LR: {ARGS.lr} || EPS: {ARGS.heb_eps} || Dim: {ARGS.heb_dim} || Dataset: {ARGS.data_name.upper()} || Learning Rule: {ARGS.heb_learn.lower().capitalize()} || Function Type: {ARGS.heb_growth.lower().capitalize()} || Focus: {ARGS.heb_focus.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Freeze Test/Train Acc: avg = {avg_test}/{avg_train}, var = {var_test}/{var_train} || Cos-Sim Test/Train Acc (Dataset): avg = {avg_test_cos}/{avg_train_cos}, var = {var_test_cos}/{var_train_cos} || Cos-Sim Test/Train Acc (EXT-Dataset): avg = {avg_test_cos_ext}/{avg_train_cos_ext}, var = {var_test_cos_ext}/{var_train_cos_ext} || Norm Test/Train Acc (Dataset): avg = {avg_test_norm}/{avg_train_norm}, var = {var_test_norm}/{var_train_norm} || Norm Test/Train Acc (EXT-Dataset): avg = {avg_test_norm_ext}/{avg_train_norm_ext}, var = {var_test_norm_ext}/{var_train_norm_ext}")


# Model Training
def train_and_eval(args: Tuple) -> List[float]:
    params: argparse.Namespace
    num: int
    params, num = args
    model: Network = HebbianNetwork('Hebbian Network', params).to(params.device)
    experiment: Experiment = GeneralizationExperiment(model, params, f'-{params.experiment_name}-{params.experiment_type.lower()}-{params.lr}--{params.heb_learn.lower()}-{params.heb_growth.lower()}-{params.heb_focus.lower()}-{params.heb_inhib.lower()}-{params.heb_lamb}---{params.class_learn.lower()}-{params.class_growth.lower()}-{params.class_focus.lower()}-{num}')
    accuracies: List[float] = list(experiment.run())
    experiment.cleanup()
    
    return accuracies


# Parallel Training
def parallel_training(params: argparse.Namespace, total: int) -> List[List[float]]:
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=total) as pool:
        # Map the list of parameters to the function that performs training
        param_list = [
            (params, process_id)
            for process_id in range(total)
        ]
        results = pool.map(train_and_eval, param_list)
    
    # Split results into train and test accuracy lists
    freeze_train_acc_list = [result[0] for result in results]
    freeze_test_acc_list = [result[1] for result in results]
    
    cos_train_list = [result[2] for result in results]
    norm_train_list = [result[3] for result in results]
    cos_test_list = [result[4] for result in results]
    norm_test_list = [result[5] for result in results]

    cos_train_ext_list = [result[6] for result in results]
    norm_train_ext_list = [result[7] for result in results]
    cos_test_ext_list = [result[8] for result in results]
    norm_test_ext_list = [result[9] for result in results]
        
    return [freeze_train_acc_list, 
            freeze_test_acc_list, 
            cos_train_list, 
            cos_test_list, 
            norm_train_list, 
            norm_test_list, 
            cos_train_ext_list,
            cos_test_ext_list,
            norm_train_ext_list,
            norm_test_ext_list]


# When to run code
if __name__ == "__main__":
    main()
    print("Process Completed.")