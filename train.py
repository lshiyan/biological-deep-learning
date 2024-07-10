from experiments.base_experiment import BaseExperiment

from experiments.generalization_experiment import GeneralizationExperiment
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
lambda_testing = [15]

for num in range(1, 16):
    lambda_test.append(num)
    lambda_test.append(1/num)

lambda_test.sort()

for l in lambda_testing:
    ARGS.heb_lamb = l
    
    test_acc_list = []
    train_acc_list = []
    test_acc_e_list = []
    train_acc_e_list = []
    
    rec_cos_test_list = []
    rec_cost_train_list = []
    rec_norm_test_list = []
    rec_norm_train_list = []
    
    rec_cos_test_e_list = []
    rec_cost_train_e_list = []
    rec_norm_test_e_list = []
    rec_norm_train_e_list = []
    
    for num in range(0, 1):
        # Base model training
        model = HebbianNetwork(ARGS)
        
        if ARGS.experiment_type.lower() == 'base':
            experiment = BaseExperiment(model, ARGS, f'{ARGS.device}-{ARGS.experiment_type.lower()}-{l}-{num}')
        elif ARGS.experiment_type.lower() == 'generalization':
            experiment = GeneralizationExperiment(model, ARGS, f'{ARGS.device}-{ARGS.experiment_type.lower()}-{l}-{num}')
        
        accuracies = list(experiment.run())
        experiment.cleanup()
        
        if ARGS.experiment_type.lower() == 'base':
            train_acc_list.append(accuracies[0])
            test_acc_list.append(accuracies[1])
        
        elif ARGS.experiment_type.lower() == 'generalization':
            train_acc_e_list.append(accuracies[0])
            test_acc_e_list.append(accuracies[1])
            
            rec_cost_train_list.append(accuracies[2])
            rec_norm_train_list.append(accuracies[3])
            rec_cos_test_list.append(accuracies[4])
            rec_norm_test_list.append(accuracies[5])
            
            rec_cost_train_e_list.append(accuracies[6])
            rec_norm_train_e_list.append(accuracies[7])
            rec_cos_test_e_list.append(accuracies[8])
            rec_norm_test_e_list.append(accuracies[9])
            
    if ARGS.experiment_type.lower() == 'base':
        avg_test = average(test_acc_list)
        var_test = variance(test_acc_list)
        avg_train = average(train_acc_list)
        var_train = variance(train_acc_list)
        
        results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {l} || Dataset: {experiment.data_name.upper()} || Inhibition: {ARGS.inhibition_rule.lower().capitalize()} || Learning Rule: {ARGS.learning_rule.lower().capitalize()} || Function Type: {ARGS.weight_growth.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Test Acc: avg = {avg_test}, var = {var_test} || Train Acc: avg = {avg_train}, var = {var_train}")
        
    elif ARGS.experiment_type.lower() == 'generalization':
        
        # Freeze EMNIST
        avg_test_e = average(test_acc_e_list)
        var_test_e = variance(test_acc_e_list)
        avg_train_e = average(train_acc_e_list)
        var_train_e = variance(train_acc_e_list)
        
        results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {l} || Dataset: {experiment.e_data_name.upper()} || Inhibition: {ARGS.inhibition_rule.lower().capitalize()} || Learning Rule: {ARGS.learning_rule.lower().capitalize()} || Function Type: {ARGS.weight_growth.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Test Acc: avg = {avg_test_e}, var = {var_test_e} || Train Acc: avg = {avg_train_e}, var = {var_train_e}") # type: ignore
        
        # Cos MNIST
        avg_test_cos = average(rec_cos_test_list)
        var_test_cos = variance(rec_cos_test_list)
        avg_train_cos = average(rec_cost_train_list)
        var_train_cos = variance(rec_cost_train_list)
        
        results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {l} || Dataset: {experiment.data_name.upper()} || Inhibition: {ARGS.inhibition_rule.lower().capitalize()} || Learning Rule: {ARGS.learning_rule.lower().capitalize()} || Function Type: {ARGS.weight_growth.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Cos-Sim Test Acc: avg = {avg_test_cos}, var = {var_test_cos} || Cos-Sim Train Acc: avg = {avg_train_cos}, var = {var_train_cos}")
        
        # Cos EMNIST
        avg_test_cos_e = average(rec_cos_test_e_list)
        var_test_cos_e = variance(rec_cos_test_e_list)
        avg_train_cos_e = average(rec_cost_train_e_list)
        var_train_cos_e = variance(rec_cost_train_e_list)

        results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {l} || Dataset: {experiment.e_data_name.upper()} || Inhibition: {ARGS.inhibition_rule.lower().capitalize()} || Learning Rule: {ARGS.learning_rule.lower().capitalize()} || Function Type: {ARGS.weight_growth.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Cos-Sim Test Acc: avg = {avg_test_cos_e}, var = {var_test_cos_e} || Cos-Sim Train Acc: avg = {avg_train_cos_e}, var = {var_train_cos_e}") # type: ignore
        
        # Norm MNIST
        avg_test_norm = average(rec_norm_test_list)
        var_test_norm = variance(rec_norm_test_list)
        avg_train_norm = average(rec_norm_train_list)
        var_train_norm = variance(rec_norm_train_list)
        
        results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {l} || Dataset: {experiment.data_name.upper()} || Inhibition: {ARGS.inhibition_rule.lower().capitalize()} || Learning Rule: {ARGS.learning_rule.lower().capitalize()} || Function Type: {ARGS.weight_growth.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Norm Test Acc: avg = {avg_test_norm}, var = {var_test_norm} || Norm Train Acc: avg = {avg_train_norm}, var = {var_train_norm}")
        
        # Norm EMNIST
        avg_test_norm_e = average(rec_norm_test_list)
        var_test_norm_e = variance(rec_norm_test_list)
        avg_train_norm_e = average(rec_norm_train_e_list)
        var_train_norm_e = variance(rec_norm_train_e_list)
        
        results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {l} || Dataset: {experiment.e_data_name.upper()} || Inhibition: {ARGS.inhibition_rule.lower().capitalize()} || Learning Rule: {ARGS.learning_rule.lower().capitalize()} || Function Type: {ARGS.weight_growth.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Norm Test Acc: avg = {avg_test_norm_e}, var = {var_test_norm_e} || Norm Train Acc: avg = {avg_train_norm_e}, var = {var_train_norm_e}") # type: ignore
        
        
        
