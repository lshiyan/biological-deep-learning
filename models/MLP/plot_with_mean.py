import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

data = pd.read_csv("MLP_hyper_search/multi_layer_results.csv", usecols=range(11))

sns.set(style="whitegrid")

output_folder = "plots"
os.makedirs(output_folder, exist_ok=True)

parameters = ['num_layers']

for param in parameters:
    plt.figure(figsize=(10, 6))
    
    grouped_data = data.groupby(param)['test_accuracy'].agg(['mean', 'std']).reset_index()
    
    if data[param].dtype == 'object' or data[param].dtype == 'bool':
        sns.pointplot(data=grouped_data, x=param, y='mean', join=False, capsize=0.1, errwidth=1, yerr=grouped_data['std'])
    else:
        plt.errorbar(grouped_data[param], grouped_data['mean'], yerr=grouped_data['std'], fmt='o', capsize=5)
    
    plt.title(f"Test Accuracy vs {param.capitalize()} (with Std Dev)")
    plt.xlabel(param.capitalize())
    plt.ylabel("Test Accuracy")

    plot_filename = f"{output_folder}/Test_Accuracy_vs_{param}_with_std_dev.png"
    plt.savefig(plot_filename)
    plt.close()

print(f"Plots saved to the '{output_folder}' directory.")
