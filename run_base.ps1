# Define the base script name
$scriptName = 'train_base.py'

# Define the values for the parameters you want to vary
$lambda_values = @(1, 2, 4)
$heb_learn_values = @('sanger')
$heb_growth_values = @('linear')
$clas_growth_values = @('linear')
$heb_focus_values = @('synapse')
$heb_inhib_values = @('RELU')
$learning_rate_values = @(0.01)

# Process List
$processes = @()

# Iterate over each value for the parameters
foreach ($focus in $heb_focus_values) {
    foreach ($heb_growth in $heb_growth_values) {
        foreach ($heb_learn in $heb_learn_values) {
            foreach ($lambda in $lambda_values) {
                foreach ($lr in $learning_rate_values) {
                    foreach ($clas_growth in $clas_growth_values) {
                        foreach ($heb_inhib in $heb_inhib_values) {
                            # Construct the complete set of arguments including the varying parameter
                            $arguments = @(
                            # Basic configurations  
                                "--data_name=MNIST",
                                "--experiment_name=_LR_EXPLORATION_FINAL",
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
                                "--input_dim=784",
                                "--heb_dim=64",
                                "--output_dim=10",
                                "--heb_gam=0.99",
                                "--heb_eps=0.0001",
                                "--heb_inhib=$heb_inhib",
                                "--heb_focus=$focus",
                                "--heb_growth=$heb_growth",
                                "--heb_learn=$heb_learn",
                                "--heb_lamb=$lambda",
                                "--heb_act=normalized",
                                "--class_learn=SOFT_HEBB",
                                "--class_growth=$clas_growth",
                                "--class_bias=no_bias",
                                "--class_focus=neuron",
                                "--class_act=normalized",
                                "--lr=$lr",
                                "--sigmoid_k=1",
                                "--alpha=0",
                                "--beta=1e-2",
                                "--sigma=1",
                                "--mu=0",
                                "--init=uniform",
                                "--batch_size=1",
                                "--epochs=10",
                                "--device=cpu",  # Change to "--device=cuda:4" if using GPU
                                "--local_machine=True",
                                "--experiment_type=base"
                            )
                            
                            # Start the process with the complete set of arguments
                            $allArgs = @($scriptName) + $arguments
                            $process = Start-Process -FilePath "python" -ArgumentList $allArgs -NoNewWindow -PassThru
                            $processes += $process.Id
                            Write-Output "Started process: $($process.Id)"
                        }
                    }
                }
            }
        }
    }
}

# Wait for all subprocesses to complete
foreach ($processId in $processes) {
    Start-Process -NoNewWindow -PassThru -FilePath "powershell" -ArgumentList "-Command", "Wait-Process -Id $processId"
}

# Continue with the rest of the script
Write-Output "All subprocesses have completed."
