import numpy as np
import torch
from torch.utils.data import TensorDataset
from typing import Tuple
import matplotlib.pyplot as plt


class DataSetupLayer:
    
    @staticmethod
    def generate_bar_matrix(
        n: int, 
        min_horizontal: int = 1, 
        max_horizontal: int = -1, 
        min_vertical: int = 1, 
        max_vertical: int = -1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a single n x n matrix with random vertical and horizontal white bars.
        
        Args:
            n (int): The size of the matrix.
            min_horizontal (int): Minimum number of horizontal bars (rows) that should be white.
            max_horizontal (int): Maximum number of horizontal bars (rows) that should be white.
            min_vertical (int): Minimum number of vertical bars (columns) that should be white.
            max_vertical (int): Maximum number of vertical bars (columns) that should be white.
            
        Returns:
            Tuple containing:
            - matrix (np.ndarray): The generated n x n matrix.
            - col_labels (np.ndarray): n-sized array representing which columns have white bars.
            - row_labels (np.ndarray): n-sized array representing which rows have white bars.
        """
        # Set default values if max values are not provided
        if max_horizontal == -1:
            max_horizontal = n
        if max_vertical == -1:
            max_vertical = n
        
        # Ensure the max values are within valid range
        max_horizontal = min(max_horizontal, n)
        max_vertical = min(max_vertical, n)
        
        # Initialize the matrix with all black (0)
        matrix = np.zeros((n, n), dtype=np.float32)
        
        # Randomly choose the number of rows and columns to set to white
        num_horizontal_bars = np.random.randint(min_horizontal, max_horizontal + 1)
        num_vertical_bars = np.random.randint(min_vertical, max_vertical + 1)
        
        # Randomly select which rows and columns will have white bars
        row_indices = np.random.choice(n, size=num_horizontal_bars, replace=False)
        col_indices = np.random.choice(n, size=num_vertical_bars, replace=False)
        
        # Create label arrays
        col_labels = np.zeros(n, dtype=np.int32)
        row_labels = np.zeros(n, dtype=np.int32)
        
        # Set the selected columns and rows to white (1)
        for i in col_indices:
            matrix[:, i] = 1  # Set the entire column to white
            col_labels[i] = 1
        
        for i in row_indices:
            matrix[i, :] = 1  # Set the entire row to white
            row_labels[i] = 1
        
        return matrix, col_labels, row_labels

    @staticmethod
    def generate_dataset(
        num_samples: int, 
        n: int, 
        min_horizontal: int = 1, 
        max_horizontal: int = -1, 
        min_vertical: int = 1, 
        max_vertical: int = -1
    ) -> TensorDataset:
        """
        Generate a dataset of n x n matrices with random white bars and corresponding labels.
        
        Args:
            num_samples (int): Number of samples to generate.
            n (int): Size of each n x n matrix.
            min_horizontal (int): Minimum number of horizontal bars (rows) that should be white.
            max_horizontal (int): Maximum number of horizontal bars (rows) that should be white.
            min_vertical (int): Minimum number of vertical bars (columns) that should be white.
            max_vertical (int): Maximum number of vertical bars (columns) that should be white.
            
        Returns:
            TensorDataset: Dataset containing the matrices and their corresponding labels.
        """
        matrices = []
        col_labels = []
        row_labels = []
        
        for _ in range(num_samples):
            matrix, col_label, row_label = DataSetupLayer.generate_bar_matrix(
                n, 
                min_horizontal=min_horizontal, 
                max_horizontal=max_horizontal, 
                min_vertical=min_vertical, 
                max_vertical=max_vertical
            )
            matrices.append(matrix)
            col_labels.append(col_label)
            row_labels.append(row_label)
        
        # Convert lists to tensors
        matrices_tensor = torch.tensor(matrices).unsqueeze(1)  # Add channel dimension for PyTorch
        col_labels_tensor = torch.tensor(col_labels)
        row_labels_tensor = torch.tensor(row_labels)
        
        return TensorDataset(matrices_tensor, col_labels_tensor, row_labels_tensor)

    @staticmethod
    def save_dataset(dataset: TensorDataset, filename: str) -> None:
        """
        Save the generated dataset to a file.
        
        Args:
            dataset (TensorDataset): The dataset to save.
            filename (str): The file path to save the dataset.
        """
        torch.save(dataset, filename)
        print(f"Dataset saved to {filename}")

    @staticmethod
    def load_dataset(filename: str) -> TensorDataset:
        """
        Load a previously saved dataset from a file.
        
        Args:
            filename (str): Path to the saved dataset file.
            
        Returns:
            TensorDataset: The loaded dataset.
        """
        return torch.load(filename)
    
def visualize_samples(dataset: TensorDataset, num_samples: int = 5) -> None:
    """
    Visualize samples from the dataset.
    
    Args:
        dataset (TensorDataset): The dataset to visualize.
        num_samples (int): Number of samples to visualize.
    """
    # Get the tensors from the dataset
    matrices, col_labels, row_labels = dataset.tensors
    
    # Plot the specified number of samples
    plt.figure(figsize=(10, 2 * num_samples))
    for i in range(min(num_samples, len(matrices))):
        matrix = matrices[i].squeeze(0).numpy()  # Remove channel dimension and convert to numpy array
        col_label = col_labels[i].numpy()
        row_label = row_labels[i].numpy()
        
        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(matrix, cmap='gray')
        plt.title(f"Sample {i+1} - Col Labels: {col_label}, Row Labels: {row_label}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    n = 28  # Example matrix size (28x28)
    num_samples = 1000  # Number of samples in the dataset
    min_horizontal = 2  # Minimum number of horizontal bars
    max_horizontal = 5  # Maximum number of horizontal bars
    min_vertical = 3  # Minimum number of vertical bars
    max_vertical = 4  # Maximum number of vertical bars
    output_file = 'bar_matrix_dataset.pt'  # Output file name

    # Generate the dataset
    dataset = DataSetupLayer.generate_dataset(
        num_samples, 
        n, 
        min_horizontal=min_horizontal, 
        max_horizontal=max_horizontal, 
        min_vertical=min_vertical, 
        max_vertical=max_vertical
    )

    # Save the dataset to a file
    DataSetupLayer.save_dataset(dataset, output_file)

    # Load the dataset to verify it was saved correctly
    loaded_dataset = DataSetupLayer.load_dataset(output_file)
    print(f"Loaded dataset contains {len(loaded_dataset)} samples.")

    # Visualize some samples from the dataset
    visualize_samples(loaded_dataset, num_samples=5)

if __name__ == "__main__":
    main()