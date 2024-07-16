from experiments.base_experiment import BaseExperiment

from experiments.generalization_experiment import GeneralizationExperiment
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
lambda_test = lambda_list()
lr_test = lr_list()
eps_test = eps_list()
dim_test = dim_list()
sigmoid_k_test = sigmoid_k_list(1)


for l in lambda_test:
    for lr in lr_test:
        for eps in eps_test:
            for dim in dim_test:
                for k in sigmoid_k_test:
                    ARGS.heb_lamb = l
                    ARGS.lr = lr
                    ARGS.heb_eps = eps
                    ARGS.heb_dim = dim
                    ARGS.sigmoid_k = k
    
                    test_acc_list = []
                    train_acc_list = []
                    test_acc_e_list = []
                    train_acc_e_list = []
    
                    for num in range(0, 1):
                        # Base model training
                        model = HebbianNetwork('Hebbian Network', ARGS).to(ARGS.device)
                        
                        experiment = BaseExperiment(model, ARGS, f'{ARGS.experiment_type.lower()}-{l}-{lr}-{eps}-{dim}-{k}-{num}')
                        accuracies = list(experiment.run())
                        experiment.cleanup()
                        
                        train_acc_list.append(accuracies[0])
                        test_acc_list.append(accuracies[1])
                            
                    avg_test = average(test_acc_list)
                    var_test = variance(test_acc_list)
                    avg_train = average(train_acc_list)
                    var_train = variance(train_acc_list)
                    
                    results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {l} || LR: {lr} || EPS: {eps} || Dim: {dim} || Sigmoid K: {k} || Dataset: {experiment.data_name.upper()} || Inhibition: {ARGS.inhibition_rule.lower().capitalize()} || Learning Rule: {ARGS.learning_rule.lower().capitalize()} || Function Type: {ARGS.weight_growth.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Test Acc: avg = {avg_test}, var = {var_test} || Train Acc: avg = {avg_train}, var = {var_train}")
                    
