from experiments.cpu_experiment import CPUExperiment

from models.YZZ_hebbian_network import YYZHebbianNetwork
from models.base_hebbian_network import BaseHebbianNetwork
from models.hopfield_hebbian_network import HopfieldHebbianNetwork
from models.sigmoid_hebbian_network import SigmoidHebbianNetwork
from models.softmax_hebbian_network import SoftmaxHebbianNetwork

from utils.experiment_parser import *
from utils.experiment_comparer import *
from utils.experiment_logger import *
from utils.experiment_timer import *
from utils.experiment_stats import *

# Create log
base_results_log = configure_logger('Base Result Log', './results/base_results.log')
ortho_results_log = configure_logger('Ortho Result Log', './results/ortho_results.log')
sigmoid_results_log = configure_logger('Sigmoid Result Log', './results/sigmoid_results.log')
softmax_results_log = configure_logger('Softmax Result Log', './results/softmax_results.log')
hopfield_results_log = configure_logger('Hopfield Result Log', './results/hopfield_results.log')

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


# Base Model Experiment
base_model = BaseHebbianNetwork(ARGS)
base_experiment = CPUExperiment(base_model, ARGS, f'cpu-base-{ARGS.heb_lamb}')
base_test_acc, base_train_acc = base_experiment.run()
base_experiment.cleanup()

base_results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {ARGS.heb_lamb} || Test Acc: {base_test_acc} || Train Acc: avg = {base_train_acc}")


# YZZ Model Experiment
ortho_model = YYZHebbianNetwork(ARGS)
ortho_experiment = CPUExperiment(ortho_model, ARGS, f'cpu-YZZ-{ARGS.heb_lamb}')
ortho_test_acc, ortho_train_acc = ortho_experiment.run()
ortho_experiment.cleanup()

ortho_results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {ARGS.heb_lamb} || Test Acc: {ortho_test_acc} || Train Acc: avg = {ortho_train_acc}")


# Sigmoid Model Experiment
sigmoid_model = SigmoidHebbianNetwork(ARGS)
sigmoid_experiment = CPUExperiment(sigmoid_model, ARGS, f'cpu-sigmoid-{ARGS.heb_lamb}')
sigmoid_test_acc, sigmoid_train_acc = sigmoid_experiment.run()
sigmoid_experiment.cleanup()

sigmoid_results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {ARGS.heb_lamb} || Test Acc: {sigmoid_test_acc} || Train Acc: avg = {sigmoid_train_acc}")


# Softmax Model Experiment
softmax_model = SoftmaxHebbianNetwork(ARGS)
softmax_experiment = CPUExperiment(softmax_model, ARGS, f'cpu-softmax-{ARGS.heb_lamb}')
softmax_test_acc, softmax_train_acc = softmax_experiment.run()
softmax_experiment.cleanup()

softmax_results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {ARGS.heb_lamb} || Test Acc: {softmax_test_acc} || Train Acc: avg = {softmax_train_acc}")


# Hopfield Model Experiment
hopfield_model = HopfieldHebbianNetwork(ARGS)
hopfield_experiment = CPUExperiment(hopfield_model, ARGS, f'cpu-hopfield-{ARGS.heb_lamb}')
hopfield_test_acc, hopfield_train_acc = hopfield_experiment.run()
hopfield_experiment.cleanup()

hopfield_results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {ARGS.heb_lamb} || Test Acc: {hopfield_test_acc} || Train Acc: avg = {hopfield_train_acc}")
