import pandas as pd
import matplotlib.pyplot as plt
import os

csv_file = "MLP_mexican_hat_tests/three_hidden_layer.csv"
data = pd.read_csv(csv_file, usecols=range(12))

df_grouped = data.groupby(['num_layers'])['test_accuracy'].mean().reset_index()

pivot_table = df_grouped.pivot(columns='num_layers', values='test_accuracy')

output_folder = "plots"
os.makedirs(output_folder, exist_ok=True)

plt.figure(figsize=(10, 6))
for hsize in pivot_table.columns:
    plt.plot(pivot_table.index, pivot_table[hsize], label=f'hsize={hsize}')

plt.title('Test Accuracy vs. Num Layers')
plt.xlabel('Number of Layers')
plt.ylabel('Test Accuracy')
plt.legend(title='num_layers')
plt.grid(True)

# Save the plot
plot_filename = os.path.join(output_folder, "Test_Accuracy_vs_mexican_factor_3layers.png")
plt.savefig(plot_filename)
plt.close()

print(f"Plot saved to '{plot_filename}'")
