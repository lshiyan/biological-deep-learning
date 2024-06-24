from experiments.base_heb_cpu import BaseHebCpu
from models.hebbian_network import HebbianNetwork
from utils.experiment_parser import *
from utils.experiment_comparer import *
from utils.experiment_logger import *
from utils.experiment_timer import *


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
    '--epochs': 1,
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

model = HebbianNetwork(ARGS)
experiment = BaseHebCpu(model, ARGS, 'tryout')
test_acc, train_acc = experiment.run()
print(f"Test Acc: {test_acc}.")
print(f"Train Acc: {train_acc}.")