import numpy as np
import torch
from torch.utils.data import TensorDataset
import random
from typing import Tuple, List
import matplotlib.pyplot as plt

class CustomBarMatrixDataset:

    @staticmethod
    def generate_bar_matrix(n: int, horizontal_indices: List[int], vertical_indices: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a single n x n matrix with specified horizontal and vertical white bars.
        
        Args:
            n (int): The size of the matrix.
            horizontal_indices (List[int]): Indices of horizontal bars (rows) that should be white.
            vertical_indices (List[int]): Indices of vertical bars (columns) that should be white.
            
        Returns:
            Tuple containing:
            - matrix (np.ndarray): The generated n x n matrix.
            - col_labels (np.ndarray): n-sized array representing which columns have white bars.
            - row_labels (np.ndarray): n-sized array representing which rows have white bars.
        """
        matrix = np.zeros((n, n), dtype=np.float32)
        col_labels = np.zeros(n, dtype=np.int32)
        row_labels = np.zeros(n, dtype=np.int32)

        # Set the selected columns and rows to white (1)
        for i in vertical_indices:
            matrix[:, i] = 1
            col_labels[i] = 1
        for i in horizontal_indices:
            matrix[i, :] = 1
            row_labels[i] = 1

        return matrix, col_labels, row_labels

    @staticmethod
    def generate_training_set(n: int, k: int, forbidden_combinations: List[Tuple[int, int]]) -> TensorDataset:
        """
        Generate the training set with the specified rules.
        
        Args:
            n (int): The size of the matrix.
            k (int): The exponent to determine the number of training examples (8^k).
            forbidden_combinations (List[Tuple[int, int]]): List of forbidden (horizontal, vertical) bar index pairs.
            
        Returns:
            TensorDataset: The generated training dataset.
        """
        num_samples = 8 ** k
        matrices = []
        col_labels = []
        row_labels = []

        for _ in range(num_samples):
            while True:
                horizontal_bars = random.choice([0, 1, 2])
                if horizontal_bars == 0:
                    vertical_bars = 2
                elif horizontal_bars == 1:
                    vertical_bars = random.choice([1, 2])
                else:
                    vertical_bars = random.choice([0, 1, 2])

                horizontal_indices = random.sample(range(n), horizontal_bars)
                vertical_indices = random.sample(range(n), vertical_bars)

                # Check if any forbidden combination is violated
                violation = any((h, v) in forbidden_combinations for h in horizontal_indices for v in vertical_indices)
                
                if not violation:
                    break

            matrix, col_label, row_label = CustomBarMatrixDataset.generate_bar_matrix(n, horizontal_indices, vertical_indices)
            matrices.append(matrix)
            col_labels.append(col_label)
            row_labels.append(row_label)

        matrices_tensor = torch.tensor(matrices).unsqueeze(1)
        col_labels_tensor = torch.tensor(col_labels)
        row_labels_tensor = torch.tensor(row_labels)

        return TensorDataset(matrices_tensor, col_labels_tensor, row_labels_tensor)
    
    @staticmethod
    def generate_test_set_one(n: int) -> TensorDataset:
        """
        Generate the first test set (single bars only).
        
        Args:
            n (int): The size of the matrix.
            
        Returns:
            TensorDataset: The generated test dataset.
        """
        matrices = []
        col_labels = []
        row_labels = []
        
        for i in range(n):
            # Single horizontal bar
            matrix, col_label, row_label = CustomBarMatrixDataset.generate_bar_matrix(n, [i], [])
            matrices.append(matrix)
            col_labels.append(col_label)
            row_labels.append(row_label)
            
            # Single vertical bar
            matrix, col_label, row_label = CustomBarMatrixDataset.generate_bar_matrix(n, [], [i])
            matrices.append(matrix)
            col_labels.append(col_label)
            row_labels.append(row_label)
        
        matrices_tensor = torch.tensor(matrices).unsqueeze(1)
        col_labels_tensor = torch.tensor(col_labels)
        row_labels_tensor = torch.tensor(row_labels)
        
        return TensorDataset(matrices_tensor, col_labels_tensor, row_labels_tensor)


    @staticmethod
    def generate_test_set_two(n: int, forbidden_combinations: List[Tuple[int, int]]) -> TensorDataset:
        """
        Generate the second test set (forbidden combinations).
        
        Args:
            n (int): The size of the matrix.
            forbidden_combinations (List[Tuple[int, int]]): List of forbidden (horizontal, vertical) bar index pairs.
            
        Returns:
            TensorDataset: The generated test dataset.
        """
        matrices = []
        col_labels = []
        row_labels = []

        for h, v in forbidden_combinations:
            matrix, col_label, row_label = CustomBarMatrixDataset.generate_bar_matrix(n, [h], [v])
            matrices.append(matrix)
            col_labels.append(col_label)
            row_labels.append(row_label)

        matrices_tensor = torch.tensor(matrices).unsqueeze(1)
        col_labels_tensor = torch.tensor(col_labels)
        row_labels_tensor = torch.tensor(row_labels)

        return TensorDataset(matrices_tensor, col_labels_tensor, row_labels_tensor)


    @staticmethod
    def generate_test_set_three(n: int) -> TensorDataset:
        """
        Generate the third test set (3+ bars per side).
        
        Args:
            n (int): The size of the matrix.
            
        Returns:
            TensorDataset: The generated test dataset.
        """
        matrices = []
        col_labels = []
        row_labels = []
        
        # Generate matrices with 3+ bars per side
        for horizontal_bars in range(3, n + 1):
            for vertical_bars in range(3, n + 1):
                horizontal_indices = random.sample(range(n), horizontal_bars)
                vertical_indices = random.sample(range(n), vertical_bars)
                
                matrix, col_label, row_label = CustomBarMatrixDataset.generate_bar_matrix(n, horizontal_indices, vertical_indices)
                matrices.append(matrix)
                col_labels.append(col_label)
                row_labels.append(row_label)
        
        matrices_tensor = torch.tensor(matrices).unsqueeze(1)
        col_labels_tensor = torch.tensor(col_labels)
        row_labels_tensor = torch.tensor(row_labels)
        
        return TensorDataset(matrices_tensor, col_labels_tensor, row_labels_tensor)




    @staticmethod
    def save_dataset(dataset: TensorDataset, filename: str) -> None:
        torch.save(dataset, filename)
        print(f"Dataset saved to {filename}")

    @staticmethod
    def load_dataset(filename: str) -> TensorDataset:
        return torch.load(filename)

    @staticmethod
    def visualize_samples(dataset: TensorDataset, num_samples: int = 5) -> None:
        matrices, col_labels, row_labels = dataset.tensors
        plt.figure(figsize=(10, 2 * num_samples))
        for i in range(min(num_samples, len(matrices))):
            matrix = matrices[i].squeeze(0).numpy()
            col_label = col_labels[i].numpy()
            row_label = row_labels[i].numpy()
            plt.subplot(num_samples, 1, i + 1)
            plt.imshow(matrix, cmap='gray')
            plt.title(f"Sample {i+1} - Col Labels: {col_label}, Row Labels: {row_label}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

def main():
    n = 28
    k = 3
    random.seed(42)
    np.random.seed(42)
    
    forbidden_combinations = [(0, 1), (2, 3), (3,3), (4,4)]  # Example of forbidden combinations
    
    # Generate the training set
    training_set = CustomBarMatrixDataset.generate_training_set(n, k, forbidden_combinations)
    CustomBarMatrixDataset.save_dataset(training_set, 'training_set.pt')
    CustomBarMatrixDataset.visualize_samples(training_set, num_samples=5)
    
    # Generate test set one (single bars only)
    test_set_one = CustomBarMatrixDataset.generate_test_set_one(n)
    CustomBarMatrixDataset.save_dataset(test_set_one, 'test_set_one.pt')
    CustomBarMatrixDataset.visualize_samples(test_set_one, num_samples=5)
    
    # Generate test set two (forbidden combinations)
    test_set_two = CustomBarMatrixDataset.generate_test_set_two(n, forbidden_combinations)
    CustomBarMatrixDataset.save_dataset(test_set_two, 'test_set_two.pt')
    CustomBarMatrixDataset.visualize_samples(test_set_two, num_samples=5)
    
    # Generate test set three (3+ bars per side)
    test_set_three = CustomBarMatrixDataset.generate_test_set_three(n)
    CustomBarMatrixDataset.save_dataset(test_set_three, 'test_set_three.pt')
    CustomBarMatrixDataset.visualize_samples(test_set_three, num_samples=5)

if __name__ == "__main__":
    main()