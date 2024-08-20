import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
from layers.input_layer import InputLayer
from typing import IO, List
import numpy as np
from typing import Tuple



from utils.experiment_constants import DataSets

class DataSetupLayer(InputLayer):
    """
    CLASS
    Defines how the input dataset will be processed before feeding it to the base network
    @instance attr.
        NetworkLayer ATTR.
            * Not used for this layer *
            name (LayerNames): name of layer
        InputLayer ATTR.
        OWN ATTR.
    """
    def __init__(self) -> None:
        """
        CONSTRUCTOR METHOD
        @param
            None
        @return
            None
        """
        super().__init__()

    
    @staticmethod
    def setup_data(data: str, label: str, filename: str, size: int, dataset: DataSets) -> TensorDataset:
        """
        STATIC METHOD
        Function to setup requested dataset
        @param
            data: data filename
            label: label filename
            filename: data (img + label) filename
            size: number of data
        @return
            tensor dataset containing (data, label)
        """
        # Converting to .csv file if needed
        if not os.path.exists(filename):
            DataSetupLayer.convert(data, label, filename, size, 28)
         
        # Setup dataset   
        data_frame: pd.DataFrame = pd.read_csv(filename, header=None, on_bad_lines='skip')
        labels: torch.Tensor = torch.tensor(data_frame[0].values) if dataset != DataSets.E_MNIST else torch.tensor(data_frame[0].values) - 1
        data_tensor: torch.Tensor = torch.tensor(data_frame.drop(data_frame.columns[0], axis=1).values, dtype=torch.float)
        if dataset == DataSets.E_MNIST: data_tensor.T
        data_tensor /= 255
        
        return TensorDataset(data_tensor, labels)
    
    
    @staticmethod
    def filter_data_loader(data_loader: DataLoader, filter: dict[int, int]):
        """
        Function to filter dataset to include only the specified classes
        @param
            tensor_dataset: original TensorDataset containing EMNIST data and labels
            selected_classes: list of letter classes to include
        @return
            filtered TensorDataset containing only data for the specified letter classes
        """
        data_kept: List[torch.Tensor] = []
        labels_kept: List[float] = []

        # Loop through each label in the dataset
        for data, label in data_loader:
            if label.item() in filter.keys():
                labels_kept.append(filter[int(label.item())])
                data_kept.append(data)

        filtered_data = torch.stack(data_kept)
        filtered_labels = torch.tensor(labels_kept)
        filtered_dataset = TensorDataset(filtered_data, filtered_labels)
        filtered_data_loader = DataLoader(filtered_dataset, batch_size=data_loader.batch_size, shuffle=True)
        
        return filtered_data_loader

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
        if max_vertical  == -1:
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


    @staticmethod
    def convert(img_file_path: str, 
                label_file_path: str, 
                out_file_path: str, 
                data_size: int, 
                img_size: int) -> None:
        """
        STATIC METHOD
        Convert .ubyte files into a .csv file for ease of use
        @param
            img_file: path to image files
            label_file: path to file with labels
            out_file: path to .cvs file that merges data and label
            data_size: number of data inputs
            img_size: size of image
        @return
            None
        """    
        # Get absolute path of all the necessary files (img, label, out)
        project_root: str = os.getcwd()
        img_file: str = os.path.join(project_root, img_file_path)
        label_file: str = os.path.join(project_root, label_file_path)
        out_file: str = os.path.join(project_root, out_file_path)

        # Open all necessary files
        imgs: IO = open(img_file, "rb")
        out: IO = open(out_file, "w")
        labels: IO = open(label_file, "rb")
        
        # Skip header bytes
        imgs.read(16)
        labels.read(8)
        
        # Create a 2D list of images where each image is a 1D list where the first element is the label
        img_area: int = img_size ** 2
        images: List[List[int]] = []

        for _ in range(data_size):
            image: List[int] = [int.from_bytes(labels.read(1), byteorder='big')]
            for _ in range(img_area):
                image.append(int.from_bytes(imgs.read(1), byteorder='big'))
            images.append(image)

        # Convert each image from 1D list to a comma-seperated str and write it into out file
        for image in images:
            out.write(",".join(str(pix) for pix in image) + "\n")
        
        # Close files
        imgs.close()
        out.close()
        labels.close()

def main():
        # Generate a dataset
        n = 28  # Matrix size
        num_samples = 1000  # Number of samples to generate
        dataset = DataSetupLayer.generate_dataset(num_samples, n, min_horizontal=2, max_horizontal=5, min_vertical=3, max_vertical=4)

        # Save the dataset
        DataSetupLayer.save_dataset(dataset, 'bar_matrix_dataset.pt')

        # Load the dataset
        loaded_dataset = DataSetupLayer.load_dataset('bar_matrix_dataset.pt')

if __name__ == "__main__":
    main()