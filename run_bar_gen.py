import subprocess

# Define the script name that will run the BarGeneralizationExperiment
script_name = 'train_bar_generalization.py'  

# Command line arguments for the experiment
arguments = [
    # Basic configurations  
    "--data_name=BAR_DATASET",  
    "--experiment_name=_FINALIZING_BAR_GEN_EXP_",
    
    # Dataset-specific arguments
    "--train_fname=data/bar_matrix/bar_train.csv",  
    "--test_fname=data/bar_matrix/bar_test.csv",
    '--data_matrix_size=4',
    '--bar_data_quantity=5',    # Total number of samples will be 8 raised to the power of 'bar_data_quantity'
    
    # Dimension of each layer (these are typical settings for a network handling 28x28 input)
    '--input_dim=16',  # total number of pixels in input image
    '--heb_dim=16',     
    '--output_dim=8',  
    
    # Hebbian layer hyperparameters
    '--heb_lamb=15', 
    '--heb_gam=0.99',
    '--heb_eps=0.0001',
    '--heb_learn=orthogonal',  
    '--heb_inhib=relu',        
    '--heb_growth=linear',     
    '--heb_focus=neuron',     
    '--heb_act=normalized',   
    
    # Classification layer hyperparameters
    '--class_learn=SUPERVISED_HEBBIAN',  
    '--class_growth=exponential',        
    '--class_bias=no_bias',              
    '--class_focus=neuron',             
    '--class_act=normalized',            
    
    # Shared hyperparameters
    '--lr=0.005',  
    '--sigmoid_k=1',  
    '--alpha=0', 
    '--beta=1e-2',  
    '--sigma=1',  
    '--mu=0',  
    '--init=uniform', 
    '--random_seed=42',
    
    # Experiment parameters
    '--batch_size=1',  
    '--epochs=10',      
    '--device=cpu',  
    '--local_machine=True',  
    '--experiment_type=bar_generalization'  
]

# Construct the command to run the experiment
command = ['python', script_name] + arguments

# Run the command using subprocess
result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Output the results
print("Standard Output:\n", result.stdout)
print("Standard Error:\n", result.stderr)