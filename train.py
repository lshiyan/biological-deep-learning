from experiments.base_experiment import BaseExperiment

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

# Experiments setup
lambda_test = []
lambda_testing = [1]

for num in range(1, 16):
    lambda_test.append(num)
    lambda_test.append(1/num)

lambda_test.sort()

for l in lambda_testing:
    ARGS.heb_lamb = l
    test_acc_list = []
    train_acc_list = []

    
    for num in range(0, 1):
        # Base model training
        model = HebbianNetwork(ARGS)
        experiment = BaseExperiment(model, ARGS, f'{ARGS.device}-{ARGS.experiment_type.lower()}-{l}-{num}')
        test_acc, train_acc = experiment.run()
        experiment.cleanup()
        
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)
    
    # Stats for base model
    avg_test = average(test_acc_list)
    var_test = variance(test_acc_list)
    avg_train = average(train_acc_list)
    var_train = variance(train_acc_list)
    
    results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {l} || Inhibition: {ARGS.inhibition_rule.lower().capitalize()} || Learning Rule: {ARGS.learning_rule.lower().capitalize()} || Function Type: {ARGS.function_type.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Test Acc: avg = {avg_test}, var = {var_test} || Train Acc: avg = {avg_train}, var = {var_train}")
    
