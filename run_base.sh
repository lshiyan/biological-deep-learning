#!/bin/bash

# Define the base script name
scriptName='train_base.py'

# Define the values for the parameter you want to vary
# lambda_values=(0.25 0.5 1 2 4 8 16)
# heb_learn_values=("sanger" "orthogonal")
# heb_growth_values=("linear" "sigmoid" "exponential")
# heb_focus_values=("synapse" "neuron")

lambda_values=(0.25 1 16)
heb_learn_values=('sanger' 'orthogonal')
heb_growth_values=('linear' 'sigmoid' 'exponential')
heb_focus_values=('synapse' 'neuron')

# Process List
processes=()

# Iterate over each value for the parameter
for focus in "${heb_focus_values[@]}"; do
    for heb_growth in "${heb_growth_values[@]}"; do
        for heb_learn in "${heb_learn_values[@]}"; do
            for lambda in "${lambda_values[@]}"; do
                # Construct the complete set of arguments including the varying parameter
                arguments=(
                    "--data_name=MNIST"
                    "--train_data=data/mnist/train-images.idx3-ubyte"
                    "--train_label=data/mnist/train-labels.idx1-ubyte"
                    "--test_data=data/mnist/test-images.idx3-ubyte"
                    "--test_label=data/mnist/test-labels.idx1-ubyte"
                    "--train_size=60000"
                    "--test_size=10000"
                    "--classes=10"
                    "--train_fname=data/mnist/mnist_train.csv"
                    "--test_fname=data/mnist/mnist_test.csv"
                    "--input_dim=784"
                    "--heb_dim=64"
                    "--output_dim=10"
                    "--heb_gam=0.99"
                    "--heb_eps=0.0001"
                    "--heb_inhib=relu"
                    "--heb_focus=$focus"
                    "--heb_growth=$heb_growth"
                    "--heb_learn=$heb_learn"
                    "--heb_lamb=$lambda"
                    "--class_learn=hebbian"
                    "--class_growth=linear"
                    "--class_bias=no_bias"
                    "--class_focus=synapse"
                    "--class_act=basic"
                    "--lr=0.005"
                    "--sigmoid_k=1"
                    "--alpha=0"
                    "--beta=10e-7"
                    "--sigma=1"
                    "--mu=0"
                    "--init=uniform"
                    "--batch_size=1"
                    "--epochs=1"
                    "--device=cpu"
                    "--local_machine=True"
                    "--experiment_type=base"
                )
                
                # Start the process with the complete set of arguments
                allArgs=($scriptName "${arguments[@]}")
                python "${allArgs[@]}" &
                processes+=($!)
                echo "Started process: ${processes[-1]}"
            done
        done
    done
done

# Wait for all subprocesses to complete
for pid in "${processes[@]}"; do
    wait $pid
done

# Continue with the rest of the script
echo "All subprocesses have completed."