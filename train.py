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

lr_test = []
lr_testing = [0.005]

hid_dim_test = [10, 784]
hid_dim_testing = [64]


for num in range(1, 16):
    lambda_test.append(num)
    lambda_test.append(1/num)

for num in range(1, 11):
    dim = 2 ** num
    hid_dim_test.append(dim)

for dim in hid_dim_testing:
    ARGS.heb_dim = dim
    
    test_acc_list = []
    train_acc_list = []
    test_acc_e_list = []
    train_acc_e_list = []
    
    cos_test_list = []
    cos_train_list = []
    norm_test_list = []
    norm_train_list = []
    
    cos_test_e_list = []
    cos_train_e_list = []
    norm_test_e_list = []
    norm_train_e_list = []
    
    freeze_train_acc_list = []
    freeze_test_acc_list = []
    
    for num in range(0, 1):
        # Base model training
        model = HebbianNetwork('Hebbian Network', ARGS)
        
        if ARGS.experiment_type.lower() == 'base':
            experiment = BaseExperiment(model, ARGS, f'{ARGS.device}-{ARGS.experiment_type.lower()}-{dim}-{num}')
        elif ARGS.experiment_type.lower() == 'generalization':
            experiment = GeneralizationExperiment(model, ARGS, f'{ARGS.device}-{ARGS.experiment_type.lower()}-{dim}-{num}')
        
        accuracies = list(experiment.run())
        experiment.cleanup()
        
        if ARGS.experiment_type.lower() == 'base':
            train_acc_list.append(accuracies[0])
            test_acc_list.append(accuracies[1])
        
        elif ARGS.experiment_type.lower() == 'generalization':
            freeze_train_acc_list.append(accuracies[0])
            freeze_test_acc_list.append(accuracies[1])
            
            cos_train_list.append(accuracies[2])
            norm_train_list.append(accuracies[3])
            cos_test_list.append(accuracies[4])
            norm_test_list.append(accuracies[5])
            
            cos_train_e_list.append(accuracies[6])
            norm_train_e_list.append(accuracies[7])
            cos_test_e_list.append(accuracies[8])
            norm_test_e_list.append(accuracies[9])
            
    if ARGS.experiment_type.lower() == 'base':
        avg_test = average(test_acc_list)
        var_test = variance(test_acc_list)
        avg_train = average(train_acc_list)
        var_train = variance(train_acc_list)
        
        results_log.info(f"Epoch: {ARGS.epochs} || Hebbian Dim: {dim} || Dataset: {experiment.data_name.upper()} || Inhibition: {ARGS.inhibition_rule.lower().capitalize()} || Learning Rule: {ARGS.learning_rule.lower().capitalize()} || Function Type: {ARGS.weight_growth.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Test Acc: avg = {avg_test}, var = {var_test} || Train Acc: avg = {avg_train}, var = {var_train}")
        
    elif ARGS.experiment_type.lower() == 'generalization':
        avg_test = average(freeze_train_acc_list)
        var_test = variance(freeze_train_acc_list)
        avg_train = average(freeze_train_acc_list)
        var_train = variance(freeze_train_acc_list)
        
        results_log.info(f"Epoch: {ARGS.epochs} || Hebbian Dim: {dim} || Dataset: {experiment.e_data_name.upper()} || Inhibition: {ARGS.inhibition_rule.lower().capitalize()} || Learning Rule: {ARGS.learning_rule.lower().capitalize()} || Function Type: {ARGS.weight_growth.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Test Acc: avg = {avg_test}, var = {var_test} || Train Acc: avg = {avg_train}, var = {var_train}") # type: ignore
        
        avg_test_cos = average(cos_test_list)
        var_test_cos = variance(cos_test_list)
        avg_train_cos = average(cos_train_list)
        var_train_cos = variance(cos_train_list)
        
        results_log.info(f"Epoch: {ARGS.epochs} || Hebbian Dim: {dim} || Dataset: {experiment.data_name.upper()} || Inhibition: {ARGS.inhibition_rule.lower().capitalize()} || Learning Rule: {ARGS.learning_rule.lower().capitalize()} || Function Type: {ARGS.weight_growth.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Cos-Sim Test Acc: avg = {avg_test_cos}, var = {var_test_cos} || Cos-Sim Train Acc: avg = {avg_train_cos}, var = {var_train_cos}")
        
        avg_test_cos_e = average(cos_test_e_list)
        var_test_cos_e = variance(cos_test_e_list)
        avg_train_cos_e = average(cos_train_e_list)
        var_train_cos_e = variance(cos_train_e_list)
        
        results_log.info(f"Epoch: {ARGS.epochs} || Hebbian Dim: {dim} || Dataset: {experiment.e_data_name.upper()} || Inhibition: {ARGS.inhibition_rule.lower().capitalize()} || Learning Rule: {ARGS.learning_rule.lower().capitalize()} || Function Type: {ARGS.weight_growth.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Cos-Sim Test Acc: avg = {avg_test_cos_e}, var = {var_test_cos_e} || Cos-Sim Train Acc: avg = {avg_train_cos_e}, var = {var_train_cos_e}") # type: ignore
        
        avg_test_norm = average(norm_test_list)
        var_test_norm = variance(norm_test_list)
        avg_train_norm = average(norm_train_list)
        var_train_norm = variance(norm_train_list)
        
        results_log.info(f"Epoch: {ARGS.epochs} || Hebbian Dim: {dim} || Dataset: {experiment.data_name.upper()} || Inhibition: {ARGS.inhibition_rule.lower().capitalize()} || Learning Rule: {ARGS.learning_rule.lower().capitalize()} || Function Type: {ARGS.weight_growth.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Norm Test Acc: avg = {avg_test_norm}, var = {var_test_norm} || Norm Train Acc: avg = {avg_train_norm}, var = {var_train_norm}")
        
        avg_test_norm_e = average(norm_test_e_list)
        var_test_norm_e = variance(norm_test_e_list)
        avg_train_norm_e = average(norm_train_e_list)
        var_train_norm_e = variance(norm_train_e_list)
        
        results_log.info(f"Epoch: {ARGS.epochs} || Hebbian Dim: {dim} || Dataset: {experiment.e_data_name.upper()} || Inhibition: {ARGS.inhibition_rule.lower().capitalize()} || Learning Rule: {ARGS.learning_rule.lower().capitalize()} || Function Type: {ARGS.weight_growth.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Norm Test Acc: avg = {avg_test_norm_e}, var = {var_test_norm_e} || Norm Train Acc: avg = {avg_train_norm_e}, var = {var_train_norm_e}") # type: ignore
        
        
        
