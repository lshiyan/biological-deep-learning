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
# lambda_test = [1, 5, 10, 15]
# lambda_test = [15]

# for num in range(1,16):
#     lambda_test.append(1/num)
#     lambda_test.append(num)
    
# lambda_test.sort()

# for l in lambda_test:
#     ARGS.heb_lamb = l
#     base_test_acc_list = []
#     base_train_acc_list = []
    
#     ortho_test_acc_list = []
#     ortho_train_acc_list = []
    
#     for num in range(0, 30):
#         # Base model training
#         base_model = HebbianNetwork(ARGS)
#         base_experiment = BaseHebCPU(base_model, ARGS, f'cpu-base-{l}-{num}')
#         base_test_acc, base_train_acc = base_experiment.run()
#         base_test_acc_list.append(base_test_acc)
#         base_train_acc_list.append(base_train_acc)
#         base_experiment.cleanup()
    
#         ortho_model = OrthoHebNetwork(ARGS)
#         ortho_exp = BaseHebCPU(ortho_model, ARGS, f'cpu-ortho-{l}-{num}')
#         ortho_test_acc, ortho_train_acc = ortho_exp.run()
#         ortho_exp.cleanup()
    
#     # Stats for base model
#     base_avg_test = average(base_test_acc_list)
#     base_var_test = variance(base_test_acc_list)
#     base_avg_train = average(base_train_acc_list)
#     base_var_train = variance(base_train_acc_list)
    
#     base_results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {l} || Test Acc: avg = {base_avg_test}, var = {base_var_test} || Train Acc: avg = {base_avg_train}, var = {base_var_train}")
    
#     # Stats for ortho model
#     ortho_avg_test = average(ortho_test_acc_list)
#     ortho_var_test = variance(ortho_test_acc_list)
#     ortho_avg_train = average(ortho_train_acc_list)
#     ortho_var_train = variance(ortho_train_acc_list)
    
#     ortho_results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {l} || Test Acc: avg = {ortho_avg_test}, var = {ortho_var_test} || Train Acc: avg = {ortho_avg_train}, var = {ortho_var_train}")


# Hebbian Model Experiment
hebbian_model = HebbianNetwork(ARGS)
hebbian_experiment = CPUExperiment(hebbian_model, ARGS, f'cpu-hebbian-{ARGS.heb_lamb}')
hebbian_test_acc, hebbian_train_acc = hebbian_experiment.run()
hebbian_experiment.cleanup()

results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {ARGS.heb_lamb} || Test Acc: {hebbian_test_acc} || Train Acc: avg = {hebbian_train_acc}")