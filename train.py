import sys
from experiments.base_heb_cpu import BaseHebCPU
from models.hebbian_network import HebbianNetwork
from utils.experiment_parser import *
from utils.experiment_comparer import *
from utils.experiment_logger import *
from utils.experiment_timer import *
from utils.experiment_stats import *

# Create log
results_log = configure_logger('Result Log', './results/results.log')

# Get arguments
ARGS = parse_arguments()

# Experiments setup
lambda_test = [1, 5, 10, 15]
# lambda_test = [1]

# for num in range(1,16):
#     lambda_test.append(1/num)
#     lambda_test.append(num)
    
lambda_test.sort()
print(lambda_test)

for l in lambda_test:
    ARGS.heb_lamb = l
    test_acc_list = []
    train_acc_list = []
    
    # for num in range(0, 30):
    #     model = HebbianNetwork(ARGS)
    #     experiment = BaseHebCPU(model, ARGS, f'cpu-{l}-{num}')
    #     test_acc, train_acc = experiment.run()
    #     test_acc_list.append(test_acc)
    #     train_acc_list.append(train_acc)
    #     experiment.cleanup()
    
    model = HebbianNetwork(ARGS)
    experiment = BaseHebCPU(model, ARGS, f'cpu-{l}')
    test_acc, train_acc = experiment.run()

    experiment.cleanup()
    
    # avg_test = average(test_acc_list)
    # var_test = variance(test_acc_list)
    # avg_train = average(train_acc_list)
    # var_train = variance(train_acc_list)
    
    results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {l} || Test Acc: {test_acc} || Train Acc: {train_acc}")
    # results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {l} || Test Acc: avg = {avg_test}, var = {var_test} || Train Acc: avg = {avg_train}, var = {var_train}")