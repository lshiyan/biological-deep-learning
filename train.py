from experiments.cpu_experiment import CPUExperiment
from models.base_hebbian_network import BaseHebbianNetwork
# from models.ortho_heb_network import OrthoHebNetwork
# from models.sigmoid_heb_network import SigmoidHebNetwork

from utils.experiment_parser import *
from utils.experiment_comparer import *
from utils.experiment_logger import *
from utils.experiment_timer import *
from utils.experiment_stats import *

# Create log
base_results_log = configure_logger('Base Result Log', './results/base_results.log')
# ortho_results_log = configure_logger('Ortho Result Log', './results/ortho_results.log')
# sigmoid_results_log = configure_logger('Sigmoid Result Log', './results/sigmoid_results.log')

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


# sigmoid_model = SigmoidHebNetwork(ARGS)
# sigmoid_experiment = BaseHebCPU(sigmoid_model, ARGS, f'cpu-sigmoid-{ARGS.heb_lamb}')
# sigmoid_test_acc, sigmoid_train_acc = sigmoid_experiment.run()
# sigmoid_experiment.cleanup()

# sigmoid_results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {ARGS.heb_lamb} || Test Acc: {sigmoid_test_acc} || Train Acc: avg = {sigmoid_train_acc}")

base_model = BaseHebbianNetwork(ARGS)
base_experiment = CPUExperiment(base_model, ARGS, f'cpu-base-{ARGS.heb_lamb}')
base_test_acc, base_train_acc = base_experiment.run()
base_experiment.cleanup()

base_results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {ARGS.heb_lamb} || Test Acc: {base_test_acc} || Train Acc: avg = {base_train_acc}")