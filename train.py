from experiments.cpu_experiment import CPUExperiment

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

for num in range(1,16):
    lambda_test.append(1/num)
    lambda_test.append(num)
    
lambda_test.sort()

for l in lambda_testing:
    # Set lambda parameter
    ARGS.heb_lamb = l
    
    # Get list of accuracies
    test_acc_list = []
    train_acc_list = []
    
    # Run a certain number of experiments per lambda
    for num in range(0, 30):
        # Model training/testing
        hebbian_model = HebbianNetwork(ARGS)
        hebbian_experiment = CPUExperiment(hebbian_model, ARGS, f'cpu-hebbian-{ARGS.heb_lamb}-{num}')
        hebbian_test_acc, hebbian_train_acc = hebbian_experiment.run()
        hebbian_experiment.cleanup()
        
        test_acc_list.append(hebbian_test_acc)
        train_acc_list.append(hebbian_train_acc)

    # Calculate means
    test_mean = average(test_acc_list)
    train_mean = average(train_acc_list)
    
    # Calculate variances
    test_var = variance(test_acc_list)
    train_var = variance(train_acc_list)
    
    # Calculate minimum means differences
    test_min_diff = min_diff(test_acc_list)
    train_min_diff = min_diff(train_acc_list)
    
    # Calculate minimum runs
    test_min_runs = min_sample(test_var, test_min_diff)
    train_min_runs = min_sample(train_var, train_min_diff)
    
    results_log.info(f"Learning Rule: {ARGS.learning_rule} || Inhibition: {ARGS.inhibition_rule} || Function Type: {ARGS.function_type} || Epoch: {ARGS.epochs} || Lambda: {ARGS.heb_lamb} || Test Acc: avg = {test_mean} var = {test_var} || Train Acc: avg = {train_mean} var = {train_var}")
    results_log.info(f"Minimum number of runs for test: {test_min_runs} || Minimum number of runs for train: {train_min_runs}")