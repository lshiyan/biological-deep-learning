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
lambda_test = lambda_list(15)
lr_test = lr_list(0.005)
eps_test = eps_list(0.0001)
dim_test = dim_list(64)
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
    
                    for num in range(0, 5):
                        # Base model training
                        model = HebbianNetwork('Hebbian Network', ARGS).to(ARGS.device)
                        
                        experiment = GeneralizationExperiment(model, ARGS, f'{ARGS.experiment_type.lower()}-{ARGS.learning_rule.lower()}-{ARGS.inhibition.lower()}-{ARGS.weight_growth.lower()}-{ARGS.weight_decay.lower()}-{ARGS.bias_update.lower()}-{l}-{lr}-{eps}-{dim}-{k}-{num}')
                        accuracies = list(experiment.run())
                        experiment.cleanup()
                        
                        freeze_train_acc_list.append(accuracies[0])
                        freeze_test_acc_list.append(accuracies[1])
                        
                        cos_train_list.append(accuracies[2])
                        norm_train_list.append(accuracies[3])
                        cos_test_list.append(accuracies[4])
                        norm_test_list.append(accuracies[5])
                        
                        cos_train_ext_list.append(accuracies[6])
                        norm_train_ext_list.append(accuracies[7])
                        cos_test_ext_list.append(accuracies[8])
                        norm_test_ext_list.append(accuracies[9])
            
                        avg_test = average(freeze_train_acc_list)
                        var_test = variance(freeze_train_acc_list)
                        avg_train = average(freeze_train_acc_list)
                        var_train = variance(freeze_train_acc_list)
                        
                        results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {l} || LR: {lr} || EPS: {eps} || Dim: {dim} || Sigmoid K: {k} || Inhibition: {ARGS.inhibition_rule.lower().capitalize()} || Learning Rule: {ARGS.learning_rule.lower().capitalize()} || Function Type: {ARGS.weight_growth.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Freeze Test/Train Acc: avg = {avg_test}/{avg_train}, var = {var_test}/{var_train}")
                        
                        avg_test_cos = average(cos_test_list)
                        var_test_cos = variance(cos_test_list)
                        avg_train_cos = average(cos_train_list)
                        var_train_cos = variance(cos_train_list)
                        
                        avg_test_cos_ext = average(cos_test_ext_list)
                        var_test_cos_ext = variance(cos_test_ext_list)
                        avg_train_cos_ext = average(cos_train_ext_list)
                        var_train_cos_ext = variance(cos_train_ext_list)
                        
                        results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {l} || LR: {lr} || EPS: {eps} || Dim: {dim} || Sigmoid K: {k} || Inhibition: {ARGS.inhibition_rule.lower().capitalize()} || Learning Rule: {ARGS.learning_rule.lower().capitalize()} || Function Type: {ARGS.weight_growth.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Cos-Sim Test/Train Acc (Dataset): avg = {avg_test_cos}/{avg_train_cos}, var = {var_test_cos}/{var_train_cos} || Cos-Sim Test/Train Acc (EXT-Dataset): avg = {avg_test_cos_ext}/{avg_train_cos_ext}, var = {var_test_cos_ext}/{var_train_cos_ext}")
                        
                        avg_test_norm = average(norm_test_list)
                        var_test_norm = variance(norm_test_list)
                        avg_train_norm = average(norm_train_list)
                        var_train_norm = variance(norm_train_list)
                        
                        avg_test_norm_ext = average(norm_test_ext_list)
                        var_test_norm_ext = variance(norm_test_ext_list)
                        avg_train_norm_ext = average(norm_train_ext_list)
                        var_train_norm_ext = variance(norm_train_ext_list)
                        
                        results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {l} || LR: {lr} || EPS: {eps} || Dim: {dim} || Sigmoid K: {k} || Inhibition: {ARGS.inhibition_rule.lower().capitalize()} || Learning Rule: {ARGS.learning_rule.lower().capitalize()} || Function Type: {ARGS.weight_growth.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Norm Test/Train Acc: avg = {avg_test_norm}/{avg_train_norm}, var = {var_test_norm}/{var_train_norm} || Norm Test/Train Acc: avg = {avg_test_norm_ext}/{avg_train_norm_ext}, var = {var_test_norm_ext}/{var_train_norm_ext}")
                        
                        
        
