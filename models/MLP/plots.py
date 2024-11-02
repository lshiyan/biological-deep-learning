import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

data = pd.read_csv("MLP_hyper_search/results_triangle.csv", usecols=range(9))

sns.set(style="whitegrid")

output_folder = "plots"
os.makedirs(output_folder, exist_ok=True)

parameters = ['triangle']

for param in parameters:
    plt.figure(figsize=(10, 6))
    
    if data[param].dtype == 'object' or data[param].dtype == 'bool':
        sns.boxplot(data=data, x=param, y='test_accuracy')
    else:
        sns.scatterplot(data=data, x=param, y='test_accuracy')
    
    plt.title(f"Test Accuracy vs {param.capitalize()}")
    plt.xlabel(param.capitalize())
    plt.ylabel("Test Accuracy")

    plot_filename = f"{output_folder}/Test_Accuracy_vs_{param}.png"
    plt.savefig(plot_filename)
    plt.close()  

print(f"Plots saved to the '{output_folder}' directory.")


