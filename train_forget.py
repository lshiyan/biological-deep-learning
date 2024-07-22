from experiments.forget_experiment import ForgetExperiment


from models.hebbian_network import HebbianNetwork
from utils.experiment_parser import *
from utils.experiment_comparer import *
from utils.experiment_logger import *
from utils.experiment_testing import eps_list, dim_list, lambda_list, lr_list, sigmoid_k_list
from utils.experiment_timer import *
from utils.experiment_stats import *

# Create log
results_log = configure_logger('Forget Result Log', './results/results.log')

# Get arguments
ARGS = parse_arguments()
print(ARGS.sub_experiment_scope_list)

# Experiments setup
lambda_test = [10]
lr_test = [0.005]
eps_test = [0.0001]
dim_test = [64]
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
    
                    test_acc_digit_0_1 = []
                    test_acc_digit_2_3 = []
                    test_acc_digit_4_5 = []
                    test_acc_digit_6_7 = []
                    test_acc_digit_8_9 = []

                    train_acc_digit_0_1 = []
                    train_acc_digit_2_3 = []
                    train_acc_digit_4_5 = []
                    train_acc_digit_6_7 = []
                    train_acc_digit_8_9 = []
    
                    for num in range(0, 5):
                        # Base model training
                        model = HebbianNetwork('Hebbian Network', ARGS).to(ARGS.device)
                        
                        experiment = ForgetExperiment(model, ARGS, f'{ARGS.experiment_type.lower()}-{l}-{lr}-{eps}-{dim}-{k}-{num}')
                        accuracies = list(experiment.run())
                        experiment.cleanup()
                        
                        test_acc_digit_0_1.append(accuracies[0])
                        test_acc_digit_2_3.append(accuracies[1])
                        test_acc_digit_4_5.append(accuracies[2])
                        test_acc_digit_6_7.append(accuracies[3])
                        test_acc_digit_8_9.append(accuracies[4])

                        train_acc_digit_0_1.append(accuracies[5])
                        train_acc_digit_2_3.append(accuracies[6])
                        train_acc_digit_4_5.append(accuracies[7])
                        train_acc_digit_6_7.append(accuracies[8])
                        train_acc_digit_8_9.append(accuracies[9])

                    avg_test_0_1 = average(test_acc_digit_0_1)
                    var_test_0_1 = variance(test_acc_digit_0_1)
                    avg_train_0_1 = average(test_acc_digit_0_1)
                    var_train_0_1 = variance(test_acc_digit_0_1)
                    results_log.info(f"Digits 0 and 1 || Epoch: {ARGS.epochs} || Lambda: {l} || Dataset: {experiment.data_name.upper()} || Inhibition: {ARGS.inhibition_rule.lower().capitalize()} || Learning Rule: {ARGS.learning_rule.lower().capitalize()} || Function Type: {ARGS.weight_growth.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Test Acc: avg = {avg_test_0_1}, var = {var_test_0_1} || Train Acc: avg = {avg_train_0_1}, var = {var_train_0_1}")
                    
                    avg_test_2_3 = average(test_acc_digit_2_3)
                    var_test_2_3 = variance(test_acc_digit_2_3)
                    avg_train_2_3 = average(test_acc_digit_2_3)
                    var_train_2_3 = variance(test_acc_digit_2_3)
                    results_log.info(f"Digits 2 and 3 || Epoch: {ARGS.epochs} || Lambda: {l} || Dataset: {experiment.data_name.upper()} || Inhibition: {ARGS.inhibition_rule.lower().capitalize()} || Learning Rule: {ARGS.learning_rule.lower().capitalize()} || Function Type: {ARGS.weight_growth.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Test Acc: avg = {avg_test_2_3}, var = {var_test_2_3} || Train Acc: avg = {avg_train_2_3}, var = {var_train_2_3}")
                    
                    avg_test_4_5 = average(test_acc_digit_4_5)
                    var_test_4_5 = variance(test_acc_digit_4_5)
                    avg_train_4_5 = average(test_acc_digit_4_5)
                    var_train_4_5 = variance(test_acc_digit_4_5)
                    results_log.info(f"Digits 4 and 5 || Epoch: {ARGS.epochs} || Lambda: {l} || Dataset: {experiment.data_name.upper()} || Inhibition: {ARGS.inhibition_rule.lower().capitalize()} || Learning Rule: {ARGS.learning_rule.lower().capitalize()} || Function Type: {ARGS.weight_growth.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Test Acc: avg = {avg_test_4_5}, var = {var_test_4_5} || Train Acc: avg = {avg_train_4_5}, var = {var_train_4_5}")
                    
                    avg_test_6_7 = average(test_acc_digit_6_7)
                    var_test_6_7 = variance(test_acc_digit_6_7)
                    avg_train_6_7 = average(test_acc_digit_6_7)
                    var_train_6_7 = variance(test_acc_digit_6_7)
                    results_log.info(f"Digits 6 and 7 || Epoch: {ARGS.epochs} || Lambda: {l} || Dataset: {experiment.data_name.upper()} || Inhibition: {ARGS.inhibition_rule.lower().capitalize()} || Learning Rule: {ARGS.learning_rule.lower().capitalize()} || Function Type: {ARGS.weight_growth.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Test Acc: avg = {avg_test_6_7}, var = {var_test_6_7} || Train Acc: avg = {avg_train_6_7}, var = {var_train_6_7}")
                    
                    avg_test_8_9 = average(test_acc_digit_8_9)
                    var_test_8_9 = variance(test_acc_digit_8_9)
                    avg_train_8_9 = average(test_acc_digit_8_9)
                    var_train_8_9 = variance(test_acc_digit_8_9)
                    results_log.info(f"Digits 8 and 9 || Epoch: {ARGS.epochs} || Lambda: {l} || Dataset: {experiment.data_name.upper()} || Inhibition: {ARGS.inhibition_rule.lower().capitalize()} || Learning Rule: {ARGS.learning_rule.lower().capitalize()} || Function Type: {ARGS.weight_growth.lower().capitalize()} || Experiment Type: {ARGS.experiment_type.lower().capitalize()} || Test Acc: avg = {avg_test_8_9}, var = {var_test_8_9} || Train Acc: avg = {avg_train_8_9}, var = {var_train_8_9}")
