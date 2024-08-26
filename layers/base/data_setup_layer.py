import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
from layers.input_layer import InputLayer
from typing import IO, List
import numpy as np
from typing import Tuple
import random



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

    # Function to apply color to MNIST digit
    @staticmethod
    def apply_color(image, color):
        """
        STATIC METHOD
        Function applies color to MNIST
        @param
            image: the 28x28 MNIST image
            color: dictionary containing RGB values
        @return
            np.ndarray: Colored image as a 3-channel numpy array
        """
        brightness = image.numpy().squeeze()  # Get brightness from MNIST grayscale image
        colored_image = np.zeros((28, 28, 3))  # Create an empty 3-channel image (RGB)
        
        for i in range(3):  # Apply color based on brightness
            colored_image[:, :, i] = brightness * color['rgb'][i] / 255.0

        return colored_image

    @staticmethod
    def setup_colored_mnist(data: str, label: str, filename: str, size: int, dataset: DataSets) -> TensorDataset:
        """
        STATIC METHOD
        Function to setup colored MNIST dataset
        @param
            data: data filename
            label: label filename
            filename: data (img + label) filename
            size: number of data
            dataset: dataset type (e.g., DataSets.MNIST)
        @return
            TensorDataset containing (colored data, label)
        """
        # Color definitions in RGB
        colors = {
            'green': {'rgb': np.array([178, 217, 178]), 'cmyk': np.array([0.18, 0.0, 0.18, 0.15])},
            'beige': {'rgb': np.array([247, 234, 190]), 'cmyk': np.array([0.0, 0.05, 0.23, 0.03])},
            'red': {'rgb': np.array([250, 112, 100]), 'cmyk': np.array([0.0, 0.55, 0.60, 0.02])},
            'blue': {'rgb': np.array([31, 90, 255]), 'cmyk': np.array([0.88, 0.65, 0.0, 0.0])},
            'purple': {'rgb': np.array([175, 151, 194]), 'cmyk': np.array([0.10, 0.22, 0.0, 0.24])}
        }

        # Ensure the MNIST dataset is ready
        if not os.path.exists(filename):
            DataSetupLayer.convert(data, label, filename, size, 28)
         
        # Load the dataset
        data_frame: pd.DataFrame = pd.read_csv(filename, header=None, on_bad_lines='skip')
        labels: torch.Tensor = torch.tensor(data_frame[0].values)
        data_tensor: torch.Tensor = torch.tensor(data_frame.drop(data_frame.columns[0], axis=1).values, dtype=torch.float).reshape(-1, 28, 28)
        data_tensor /= 255
        
        # Apply colorization
        colored_data = []
        for i in range(data_tensor.shape[0]):
            color_name = np.random.choice(list(colors.keys()))  # Randomly choose a color
            colored_image = DataSetupLayer.apply_color(data_tensor[i], colors[color_name])
            colored_data.append(torch.tensor(colored_image, dtype=torch.float32).permute(2, 0, 1))  # Convert to tensor and permute to (C, H, W)
        
        colored_data_tensor = torch.stack(colored_data)  # Stack all colored images into a tensor
        
        return TensorDataset(colored_data_tensor, labels)
    
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



#####################################################################################################################
#                                        Bar Generalization Dataset Section
#####################################################################################################################

    @staticmethod
    def generate_bar_matrix(n: int, 
                            horizontal_indices: List[int], 
                            vertical_indices: List[int]
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    def generate_training_set(n: int, 
                              k: int, 
                              forbidden_combinations: List[Tuple[int, int]]
                              ) -> TensorDataset:
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

            matrix, col_label, row_label = DataSetupLayer.generate_bar_matrix(n, horizontal_indices, vertical_indices)
            matrices.append(matrix)
            col_labels.append(col_label)
            row_labels.append(row_label)

        # Assuming `matrices` and `col_labels` are lists of numpy arrays
        matrices_np = np.array(matrices)  # Convert the list of numpy arrays to a single numpy array
        matrices_tensor = torch.tensor(matrices_np).unsqueeze(1)  # Then convert to a PyTorch tensor

        col_labels_np = np.array(col_labels)  # Convert the list of numpy arrays to a single numpy array
        col_labels_tensor = torch.tensor(col_labels_np)  # Then convert to a PyTorch tensor

        row_labels_np = np.array(row_labels)  # Assuming you also have `row_labels` that need conversion
        row_labels_tensor = torch.tensor(row_labels_np)  # Convert to a PyTorch tensor

        return TensorDataset(matrices_tensor, col_labels_tensor, row_labels_tensor)



    @staticmethod
    def generate_test_set_one(
        n: int, 
        k: int
        ) -> TensorDataset:
        """
        Generate the first test set (single bars only).
        
        Args:
            n (int): The size of the matrix.
            k (int): Determines the number of samples to generate.
            
        Returns:
            TensorDataset: The generated test dataset.
        """
        num_samples = 8 ** k
        matrices = []
        col_labels = []
        row_labels = []

        while len(matrices) < num_samples:
            for i in range(n):
                if len(matrices) >= num_samples:
                    break
                # Single horizontal bar
                matrix, col_label, row_label = DataSetupLayer.generate_bar_matrix(n, [i], [])
                matrices.append(matrix)
                col_labels.append(col_label)
                row_labels.append(row_label)
                
                if len(matrices) >= num_samples:
                    break
                # Single vertical bar
                matrix, col_label, row_label = DataSetupLayer.generate_bar_matrix(n, [], [i])
                matrices.append(matrix)
                col_labels.append(col_label)
                row_labels.append(row_label)

        # Assuming `matrices` and `col_labels` are lists of numpy arrays
        matrices_np = np.array(matrices)  # Convert the list of numpy arrays to a single numpy array
        matrices_tensor = torch.tensor(matrices_np).unsqueeze(1)  # Then convert to a PyTorch tensor

        col_labels_np = np.array(col_labels)  # Convert the list of numpy arrays to a single numpy array
        col_labels_tensor = torch.tensor(col_labels_np)  # Then convert to a PyTorch tensor

        row_labels_np = np.array(row_labels)  # Assuming you also have `row_labels` that need conversion
        row_labels_tensor = torch.tensor(row_labels_np)  # Convert to a PyTorch tensor


        return TensorDataset(matrices_tensor, col_labels_tensor, row_labels_tensor)



    @staticmethod
    def generate_test_set_two(
        n: int, 
        k: int,
        forbidden_combinations: List[Tuple[int, int]]
        ) -> TensorDataset:
        """
        Generate the second test set (forbidden combinations).
        
        Args:
            n (int): The size of the matrix.
            forbidden_combinations (List[Tuple[int, int]]): List of forbidden (horizontal, vertical) bar index pairs.
            k (int): Determines the number of samples to generate.
            
        Returns:
            TensorDataset: The generated test dataset.
        """
        num_samples = 8 ** k
        matrices = []
        col_labels = []
        row_labels = []

        while len(matrices) < num_samples:
            for h, v in forbidden_combinations:
                if len(matrices) >= num_samples:
                    break
                matrix, col_label, row_label = DataSetupLayer.generate_bar_matrix(n, [h], [v])
                matrices.append(matrix)
                col_labels.append(col_label)
                row_labels.append(row_label)

        # Assuming `matrices` and `col_labels` are lists of numpy arrays
        matrices_np = np.array(matrices)  # Convert the list of numpy arrays to a single numpy array
        matrices_tensor = torch.tensor(matrices_np).unsqueeze(1)  # Then convert to a PyTorch tensor

        col_labels_np = np.array(col_labels)  # Convert the list of numpy arrays to a single numpy array
        col_labels_tensor = torch.tensor(col_labels_np)  # Then convert to a PyTorch tensor

        row_labels_np = np.array(row_labels)  # Assuming you also have `row_labels` that need conversion
        row_labels_tensor = torch.tensor(row_labels_np)  # Convert to a PyTorch tensor

        return TensorDataset(matrices_tensor, col_labels_tensor, row_labels_tensor)


    
    @staticmethod
    def generate_test_set_three(
        n: int, 
        k: int, 
        max_bars: int
        ) -> TensorDataset:
        """
        Generate the third test set with up to the maximum specified number of bars.
        
        Args:
            n (int): The size of the matrix.
            k (int): Determines the number of samples to generate.
            max_bars (int): The maximum number of bars in either direction.
            
        Returns:
            TensorDataset: The generated test dataset.
        """
        num_samples = 8 ** k
        matrices = []
        col_labels = []
        row_labels = []

        while len(matrices) < num_samples:
            # Randomly decide if the max_bars will be for horizontal or vertical
            if random.choice([True, False]):
                # max_bars for horizontal, random for vertical
                horizontal_bars = max_bars
                vertical_bars = random.randint(0, max_bars - 1)
            else:
                # max_bars for vertical, random for horizontal
                horizontal_bars = random.randint(0, max_bars - 1)
                vertical_bars = max_bars

            # Ensure we generate unique samples without exceeding the sample count
            if len(matrices) >= num_samples:
                break

            # Generate random indices for the bars
            horizontal_indices = random.sample(range(n), horizontal_bars)
            vertical_indices = random.sample(range(n), vertical_bars)

            # Generate the matrix and corresponding labels
            matrix, col_label, row_label = DataSetupLayer.generate_bar_matrix(n, horizontal_indices, vertical_indices)
            matrices.append(matrix)
            col_labels.append(col_label)
            row_labels.append(row_label)
            

        # Assuming `matrices` and `col_labels` are lists of numpy arrays
        matrices_np = np.array(matrices)  # Convert the list of numpy arrays to a single numpy array
        matrices_tensor = torch.tensor(matrices_np).unsqueeze(1)  # Then convert to a PyTorch tensor

        col_labels_np = np.array(col_labels)  # Convert the list of numpy arrays to a single numpy array
        col_labels_tensor = torch.tensor(col_labels_np)  # Then convert to a PyTorch tensor

        row_labels_np = np.array(row_labels)  # Assuming you also have `row_labels` that need conversion
        row_labels_tensor = torch.tensor(row_labels_np)  # Convert to a PyTorch tensor
        
        return TensorDataset(matrices_tensor, col_labels_tensor, row_labels_tensor)



