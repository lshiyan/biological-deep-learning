from experiments.base_heb_cpu import BaseHebCPU
from models.hebbian_network import HebbianNetwork
from utils.experiment_parser import *
from utils.experiment_comparer import *
from utils.experiment_logger import *
from utils.experiment_timer import *
from utils.experiment_stats import *

# Create log
results_log = configure_logger('Result Log', './results/results.log')

# Simulate the command line arguments
args_dict = {
    '--is_training': True,
    '--data_name': 'MNIST',
    '--train_data': 'data/mnist/train-images.idx3-ubyte',
    '--train_label': 'data/mnist/train-labels.idx1-ubyte',
    '--test_data': 'data/mnist/t10k-images.idx3-ubyte',
    '--test_label': 'data/mnist/t10k-labels.idx1-ubyte',
    '--train_filename': 'data/mnist/mnist_train.csv',
    '--test_filename': 'data/mnist/mnist_test.csv',
    '--input_dim': 784,
    '--heb_dim': 64,
    '--output_dim': 10,
    '--heb_lamb': 1,
    '--heb_gam': 0.99,
    '--cla_lamb': 1,
    '--eps': 0.01,
    '--epochs': 3,
    '--test_epochs': 1,
    '--dropout': 0.2,
    '--lr': 0.005,
    '--lr_step_size': 1000,
    '--gamma': 1,
    '--batch_size': 1,
    '--device_id': 'cpu',
    '--local_machine': True
}

# Convert the dictionary to a list of arguments
args_list = []

for k, v in args_dict.items():
    args_list.append(k)
    args_list.append(str(v))
    
ARGS = parse_arguments(args_list)

lambda_test = []

for num in range(1,16):
    lambda_test.append(1/num)
    lambda_test.append(num)
    
lambda_test.sort()
print(lambda_test)

for l in lambda_test:
    ARGS.heb_lamb = l
    test_acc_list = []
    train_acc_list = []
    
    model = HebbianNetwork(ARGS)
    experiment = BaseHebCPU(model, ARGS, f'cpu-{l}')
    test_acc, train_acc = experiment.run()
    test_acc_list.append(test_acc)
    train_acc_list.append(train_acc)
    experiment.cleanup()
    
    avg_test = average(test_acc_list)
    var_test = variance(test_acc_list)
    avg_train = average(train_acc_list)
    var_train = variance(train_acc_list)
    
    results_log.info(f"Epoch: {ARGS.epochs} || Lambda: {l} || Test Acc: avg = {avg_test}, var = {var_test} || Train Acc: avg = {avg_train}, var = {var_train}")
    