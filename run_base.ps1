# Define the base script name
$scriptName = 'train_base.py'

# Define the values for the parameter you want to vary
# $lambda_values = @(0.25, 0.5, 1, 2, 4, 8, 16)
# $sig_k_values = @(0.01, 0.1, 1, 10)
# $heb_learn_values = @('sanger', 'orthogonal')
# $heb_inhib_values = @('relu', 'max', 'exp_softmax', 'wta', 'gaussian', 'norm')
# $heb_growth_values = @('linear', 'sigmoid', 'exponential')
# $heb_focus_values = @('synapse', 'neuron')
# $class_learn_values = @('hebbian', 'controlled')

$lambda_values = @(0.25, 1, 16)
$sig_k_values = @(1)
$heb_learn_values = @('orthogonal')
$heb_inhib_values = @('relu', 'max', 'exp_softmax')
$heb_growth_values = @('linear')
$heb_focus_values = @('synapse')
$class_learn_values = @('hebbian')

# Define the static part of the arguments
$staticArguments = @(
    # Basic configurations  
    "--data_name=MNIST",
    # Data Factory - MNIST
    "--train_data=data/mnist/train-images.idx3-ubyte", 
    "--train_label=data/mnist/train-labels.idx1-ubyte", 
    "--test_data=data/mnist/test-images.idx3-ubyte", 
    "--test_label=data/mnist/test-labels.idx1-ubyte",
    "--train_size=60000",
    "--test_size=10000",
    "--classes=10",
    # # Data Factory - FASHION-MNIST
    # "--train_data=data/fashion_mnist/train-images.idx3-ubyte", 
    # "--train_label=data/fashion_mnist/train-labels.idx1-ubyte", 
    # "--test_data=data/fashion_mnist/test-images.idx3-ubyte", 
    # "--test_label=data/fashion_mnist/test-labels.idx1-ubyte", 
    # "--train_size=60000",
    # "--test_size=10000",
    # "--classes=10",
    # # Data Factory - E-MNIST
    # "--train_data=data/ext_mnist/train-images.idx3-ubyte", 
    # "--train_label=data/ext_mnist/train-labels.idx1-ubyte", 
    # "--test_data=data/ext_mnist/test-images.idx3-ubyte", 
    # "--test_label=data/ext_mnist/test-labels.idx1-ubyte", 
    # "--train_size=88800",
    # "--test_size=14800",
    # "--classes=26",
    # CSV files generated - MNIST
    "--train_fname=data/mnist/mnist_train.csv",
    "--test_fname=data/mnist/mnist_test.csv",
    # # CSV files generated - E-MNIST
    # "--train_fname=data/ext_mnist/ext_mnist_train.csv",
    # "--test_fname=data/ext_mnist/ext_mnist_test.csv",
    # # CSV files generated - FASHION-MNIST
    # "--train_fname=data/fashion_mnist/fashion_mnist_train.csv",
    # "--test_fname=data/fashion_mnist/fashion_mnist_test.csv",
    # Dimension of each layer
    '--input_dim=784', 
    '--heb_dim=64', 
    '--output_dim=10',
    # Hebbian layer hyperparameters   
    '--heb_gam=0.99',
    '--heb_eps=0.0001',
    # Classification layer hyperparameters
    '--class_growth=linear',
    '--class_bias=no_bias',
    '--class_focus=synapse',
    '--class_act=basic',
    # Shared hyperparameters
    '--lr=0.005',
    '--alpha=0',
    '--beta=10e-7',
    '--sigma=1',
    '--mu=0',
    '--init=uniform',
    # Experiment parameters
    '--batch_size=1',
    '--epochs=1', 
    '--device=cpu',
    # '--device=cuda:8',
    '--local_machine=True',
    '--experiment_type=base'
)

# Iterate over each value for the parameter
foreach ($focus in $heb_focus_values) {
    foreach ($class_learn in $class_learn_values) {
        foreach ($heb_growth in $heb_growth_values) {
            foreach ($heb_inhib in $heb_inhib_values) {
                foreach ($heb_learn in $heb_learn_values) {
                    foreach ($sig_k in $sig_k_values) {
                        foreach ($lambda in $lambda_values) {
                                # Construct the complete set of arguments including the varying parameter
                                $arguments = $staticArguments + @(
                                    "--heb_focus=$focus",
                                    "--class_learn=$class_learn",
                                    "--heb_growth=$heb_growth",
                                    "--heb_inhib=$heb_inhib",
                                    "--heb_learn=$heb_learn",
                                    "--sigmoid_k=$sig_k",
                                    "--heb_lamb=$lambda"
                                )
                                
                                # Start the process with the complete set of arguments
                                $processArguments = @($scriptName) + $arguments
                                Start-Process -FilePath "python" -ArgumentList $processArguments -NoNewWindow -PassThru
                        }
                        if ($heb_growth -ne 'sigmoid') { break }
                    }
                }
            }
        }
    }
}
