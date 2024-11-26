import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data
data = pd.read_csv("data.csv")

# Set style for seaborn
sns.set(style="whitegrid")

# List of all parameter columns
parameters = ['hsize', 'lambda', 'w_lr', 'b_lr', 'l_lr', 'triangle', 'white', 'func']

# Plot accuracy against each parameter in separate plots
for param in parameters:
    plt.figure(figsize=(10, 6))
    
    # Choose plot type based on parameter type
    if data[param].dtype == 'object' or data[param].dtype == 'bool':
        sns.boxplot(data=data, x=param, y='test_accuracy')
    else:
        sns.scatterplot(data=data, x=param, y='test_accuracy')
    
    plt.title(f"Test Accuracy vs {param.capitalize()}")
    plt.xlabel(param.capitalize())
    plt.ylabel("Test Accuracy")
    plt.show()
