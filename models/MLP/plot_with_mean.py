import pandas as pd
import matplotlib.pyplot as plt
import os

# Read the data
data = pd.read_csv("MLP_mexican_hat_tests/three_hidden_layer.csv", usecols=range(12))

# Create output directory
output_folder = "plots"
os.makedirs(output_folder, exist_ok=True)

parameters = ['num_layers']

for param in parameters:
    plt.figure(figsize=(10, 6))
    
    # Set grid style
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate mean and std for each parameter value
    grouped_data = data.groupby(param)['test_accuracy'].agg(['mean', 'std']).reset_index()
    
    # Plot with error bars
    plt.errorbar(grouped_data[param], grouped_data['mean'], 
                yerr=grouped_data['std'], 
                fmt='o-',  # Circle markers connected by lines
                capsize=5,  # Length of error bar caps
                capthick=1,  # Thickness of caps
                elinewidth=1,  # Width of error bars
                markersize=8,  # Size of markers
                color='#2196F3',  # Blue color
                ecolor='#757575',  # Gray error bars
                markeredgecolor='white',  # White edge around markers
                markeredgewidth=1,
                label='Mean Accuracy')
    
    # Customize plot appearance
    plt.title(f"Test Accuracy vs Number of Layers in MLP", fontsize=12, pad=15)
    plt.xlabel(param.capitalize(), fontsize=10)
    plt.ylabel("Test Accuracy", fontsize=10)
    
    # Set background color and edge color
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')  # Light gray background
    plt.gcf().set_facecolor('white')
    
    # Add legend
    plt.legend()
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{output_folder}/Test_Accuracy_vs_num_layers.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

print(f"Plots saved to the '{output_folder}' directory.")
