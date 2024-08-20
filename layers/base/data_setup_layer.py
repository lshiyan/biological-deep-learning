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
    def generate_bar_matrix(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a single n x n matrix with random vertical and horizontal white bars.
        
        Args:
            n (int): The size of the matrix.
            
        Returns:
            Tuple containing:
            - matrix (np.ndarray): The generated n x n matrix.
            - col_labels (np.ndarray): n-sized array representing which columns have white bars.
            - row_labels (np.ndarray): n-sized array representing which rows have white bars.
        """
        # Initialize the matrix with all black (0)
        matrix = np.zeros((n, n), dtype=np.float32)
        
        # Randomly choose which columns and rows will have white bars
        col_labels = np.random.choice([0, 1], size=(n,))
        row_labels = np.random.choice([0, 1], size=(n,))
        
        # Set the selected columns and rows to white (1)
        for i in range(n):
            if col_labels[i] == 1:
                matrix[:, i] = 1  # Set the entire column to white
            if row_labels[i] == 1:
                matrix[i, :] = 1  # Set the entire row to white
        
        return matrix, col_labels, row_labels

    @staticmethod
    def setup_bar_matrix_data(n: int, num_samples: int, output_filename: str) -> TensorDataset:
        """
        Generate and save a dataset of n x n matrices with random white bars and corresponding labels.
        
        Args:
            n (int): Size of each n x n matrix.
            num_samples (int): Number of samples to generate.
            output_filename (str): Filename to save the generated dataset.
            
        Returns:
            TensorDataset: Dataset containing the matrices and their corresponding labels.
        """
        matrices = []
        col_labels = []
        row_labels = []
        
        for _ in range(num_samples):
            matrix, col_label, row_label = DataSetupLayer.generate_bar_matrix(n)
            matrices.append(matrix)
            col_labels.append(col_label)
            row_labels.append(row_label)
        
        # Convert lists to tensors
        matrices_tensor = torch.tensor(matrices).unsqueeze(1)  # Add channel dimension for PyTorch
        col_labels_tensor = torch.tensor(col_labels)
        row_labels_tensor = torch.tensor(row_labels)
        
        # Optionally save the dataset for future use
        torch.save((matrices_tensor, col_labels_tensor, row_labels_tensor), output_filename)
        print(f"Bar matrix dataset saved to {output_filename}")
        
        return TensorDataset(matrices_tensor, col_labels_tensor, row_labels_tensor)

    @staticmethod
    def load_bar_matrix_data(filename: str) -> TensorDataset:
        """
        Load a previously saved bar matrix dataset from a file.
        
        Args:
            filename (str): Path to the saved dataset file.
            
        Returns:
            TensorDataset: The loaded dataset.
        """
        matrices_tensor, col_labels_tensor, row_labels_tensor = torch.load(filename)
        return TensorDataset(matrices_tensor, col_labels_tensor, row_labels_tensor)




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