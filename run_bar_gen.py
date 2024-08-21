import subprocess

# Define the script name that will run the BarGeneralizationExperiment
script_name = 'train_bar_generalization.py'  # Ensure this script runs the BarGeneralizationExperiment

# Command line arguments for the experiment
arguments = [
    # Basic configurations  
    "--data_name=CUSTOM_BAR_MATRIX",  # The dataset name you're using, even if it's generated internally
    "--experiment_name=BAR_GENERALIZATION_TEST",
    
    # Dataset-specific arguments
    # If dataset paths are needed for logging or other purposes, include them here
    "--train_fname=data/bar_matrix/bar_train.csv",  # These would be placeholders or paths to save the generated datasets
    "--test_fname=data/bar_matrix/bar_test.csv",
    '--data_matrix_size=4',
    '--samples=3',
    
    # Dimension of each layer (these are typical settings for a network handling 28x28 input)
    '--input_dim=16',  # For a 28x28 matrix flattened
    '--heb_dim=64',     # Number of Hebbian layer neurons
    '--output_dim=8',  # Assuming you have an output vector corresponding to rows or columns in the bar matrix
    
    # Hebbian layer hyperparameters
    '--heb_lamb=15', 
    '--heb_gam=0.99',
    '--heb_eps=0.0001',
    '--heb_learn=orthogonal',  # Your chosen learning rule
    '--heb_inhib=relu',        # Your chosen lateral inhibition method
    '--heb_growth=linear',     # Growth method for weights
    '--heb_focus=neuron',      # Focus method
    '--heb_act=normalized',    # Activation function
    
    # Classification layer hyperparameters
    '--class_learn=OUTPUT_CONTRASTIVE',  # Classification learning rule
    '--class_growth=exponential',        # Growth method for the classification layer
    '--class_bias=no_bias',              # Whether to use bias in the classification layer
    '--class_focus=neuron',              # Focus method for classification
    '--class_act=normalized',            # Activation function for classification
    
    # Shared hyperparameters
    '--lr=0.005',  # Learning rate
    '--sigmoid_k=1',  # Sigmoid constant
    '--alpha=0',  # Lower bound for parameter initialization
    '--beta=1e-2',  # Upper bound for parameter initialization
    '--sigma=1',  # Variance for normal initialization
    '--mu=0',  # Mean for normal initialization
    '--init=uniform',  # Initialization method
    
    # Experiment parameters
    '--batch_size=1',  # Batch size
    '--epochs=10',      # Number of epochs
    '--device=cpu',  # GPU device (adjust if you have multiple GPUs)
    '--local_machine=True',  # Flag indicating this is run on a local machine
    '--experiment_type=bar_generalization'  # Custom experiment type for this specific experiment
]

# Construct the command to run the experiment
command = ['python', script_name] + arguments

# Run the command using subprocess
result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Output the results
print("Standard Output:\n", result.stdout)
print("Standard Error:\n", result.stderr)