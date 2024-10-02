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
        
        # Loop through three color channels (RGB)
        for i in range(3):  
            colored_image[:, :, i] = brightness * color['rgb'][i] / 255.0   
            # 1: Retrieve RGB value corresponding to current channel
            # 2: Color each pixel of the grayscale image according to its brightness and the chosen RGB color values

        return colored_image

    @staticmethod
    def setup_colored_mnist(data: str, label: str, filename: str, size: int, dataset: DataSets, color_map: dict, test_color_map: dict = None) -> TensorDataset:
        """
        STATIC METHOD
        Function to setup colored MNIST dataset with specified colors for each digit.
        @param
            data: data filename
            label: label filename
            filename: data (img + label) filename
            size: number of data
            dataset: dataset type (e.g., DataSets.MNIST)
            color_map: dictionary mapping each digit to a list of colors for training
            test_color_map: dictionary mapping each digit to a list of colors for testing (optional)
        @return
            TensorDataset containing (colored data, label)
        """
        # Ensure the MNIST dataset is ready
        if not os.path.exists(filename):
            DataSetupLayer.convert(data, label, filename, size, 28)
        
        # Load the MNIST dataset
        data_frame: pd.DataFrame = pd.read_csv(filename, header=None, on_bad_lines='skip')
        labels: torch.Tensor = torch.tensor(data_frame[0].values)
        data_tensor: torch.Tensor = torch.tensor(data_frame.drop(data_frame.columns[0], axis=1).values, dtype=torch.float).reshape(-1, 28, 28)
        data_tensor /= 255
        
        # Apply colorization
        colored_data = []
        colored_labels = []

        for i in range(data_tensor.shape[0]):
            digit = labels[i].item()  # Get the digit label
            colors = color_map[digit]  # Get the list of colors for this digit
            
            # Apply each color in the list to create multiple colored copies of the digit
            for color in colors:
                rgb_values = color['rgb']  # Get the RGB values
                colored_image = DataSetupLayer.apply_color(data_tensor[i], {'rgb': rgb_values})
                colored_data.append(torch.tensor(colored_image, dtype=torch.float32).permute(2, 0, 1))
                colored_labels.append(digit)  # Append the label for each colored copy
        
        colored_data_tensor = torch.stack(colored_data)  # Stack all colored images into a tensor
        colored_labels_tensor = torch.tensor(colored_labels, dtype=torch.long)  # Convert labels to tensor
        
        return TensorDataset(colored_data_tensor, colored_labels_tensor)
    
    #@staticmethod
    #def setup_data(data: str, label: str, filename: str, size: int, dataset: DataSets) -> TensorDataset:
    #    """
    #    STATIC METHOD
    #    Function to setup requested dataset
    #    @param
    #        data: data filename
    #        label: label filename
    #        filename: data (img + label) filename
    #        size: number of data
    #    @return
    #        tensor dataset containing (data, label)
    #    """
    #    # Converting to .csv file if needed
    #    if not os.path.exists(filename):
    #        DataSetupLayer.convert(data, label, filename, size, 28)
    #     
    #    # Setup dataset   
    #    data_frame: pd.DataFrame = pd.read_csv(filename, header=None, on_bad_lines='skip')
    #    labels: torch.Tensor = torch.tensor(data_frame[0].values) if dataset != DataSets.E_MNIST else torch.tensor(data_frame[0].values) - 1
    #    data_tensor: torch.Tensor = torch.tensor(data_frame.drop(data_frame.columns[0], axis=1).values, dtype=torch.float)
    #    if dataset == DataSets.E_MNIST: data_tensor.T
    #    data_tensor /= 255
    #    
    #    return TensorDataset(data_tensor, labels)
    
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
        
        # Read CSV, ensure no headers and proper data types
        data_frame = pd.read_csv(filename, header=None, on_bad_lines='skip', dtype=str)  # Read as string to check for non-numeric data
        data_frame = data_frame.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, setting errors as NaN
        
        # Drop rows with any NaN values
        if data_frame.isnull().values.any():
            print("Warning: Non-numeric data found and removed.")
            data_frame = data_frame.dropna()
        
        # Convert labels to a tensor
        labels = torch.tensor(data_frame[0].values, dtype=torch.long) if dataset != DataSets.E_MNIST else torch.tensor(data_frame[0].values, dtype=torch.long) - 1
        
        # Convert data to a tensor
        data_tensor = torch.tensor(data_frame.drop(data_frame.columns[0], axis=1).values, dtype=torch.float32)
        
        # Transpose data tensor if required and normalize
        if dataset == DataSets.E_MNIST:
            data_tensor = data_tensor.T
        data_tensor /= 255.0

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

        # Set the selected columns and rows to white (1) in a vectorized way
        matrix[:, vertical_indices] = 1
        col_labels[vertical_indices] = 1

        matrix[horizontal_indices, :] = 1
        row_labels[horizontal_indices] = 1

        return matrix, col_labels, row_labels

    @staticmethod
    def generate_training_set(n: int, 
                              k: int, 
                              forbidden_combinations: List[Tuple[int, int]], 
                              base_path: str = 'data/bar_matrix'
                              ) -> TensorDataset:
        """
        Generate the training set with the specified rules or load it if it exists.
        
        Args:
            n (int): The size of the matrix.
            k (int): The exponent to determine the number of training examples (8^k).
            forbidden_combinations (List[Tuple[int, int]]): List of forbidden (horizontal, vertical) bar index pairs.
            base_path (str): Base path to save or load the dataset.
            
        Returns:
            TensorDataset: The generated or loaded training dataset.
        """
        # Dynamically construct the save path based on matrix size and quantity
        save_path = os.path.join(base_path, f'matrix_size_{n}/k_{k}/training_set.pt')

        # Check if the dataset already exists
        if os.path.exists(save_path):
            print(f"Loading existing training dataset from {save_path}")
            return torch.load(save_path)
        
        print(f"Training dataset not found. Generating a new dataset and saving it to {save_path}")
        
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

        matrices_tensor = torch.tensor(np.array(matrices)).unsqueeze(1)
        col_labels_tensor = torch.tensor(np.array(col_labels))
        row_labels_tensor = torch.tensor(np.array(row_labels))

        dataset = TensorDataset(matrices_tensor, col_labels_tensor, row_labels_tensor)
        
        # Save the generated dataset
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(dataset, save_path)
        print(f"Training dataset saved to {save_path}")

        return dataset

    @staticmethod
    def generate_test_set_one(
        n: int, 
        k: int,
        base_path: str = 'data/bar_matrix'
        ) -> TensorDataset:
        """
        Generate the first test set (single bars only) or load it if it exists.
        
        Args:
            n (int): The size of the matrix.
            k (int): Determines the number of samples to generate.
            base_path (str): Base path to save or load the dataset.
            
        Returns:
            TensorDataset: The generated or loaded test dataset.
        """
        # Dynamically construct the save path based on matrix size and quantity
        save_path = os.path.join(base_path, f'matrix_size_{n}/k_{k}/test_set_one.pt')

        # Check if the dataset already exists
        if os.path.exists(save_path):
            print(f"Loading existing test set one from {save_path}")
            return torch.load(save_path)
        
        print(f"Test set one not found. Generating a new dataset and saving it to {save_path}")

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

        matrices_tensor = torch.tensor(np.array(matrices)).unsqueeze(1)
        col_labels_tensor = torch.tensor(np.array(col_labels))
        row_labels_tensor = torch.tensor(np.array(row_labels))

        dataset = TensorDataset(matrices_tensor, col_labels_tensor, row_labels_tensor)
        
        # Save the generated dataset
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(dataset, save_path)
        print(f"Test set one saved to {save_path}")

        return dataset

    @staticmethod
    def generate_test_set_two(
        n: int, 
        k: int,
        forbidden_combinations: List[Tuple[int, int]],
        base_path: str = 'data/bar_matrix'
        ) -> TensorDataset:
        """
        Generate the second test set (forbidden combinations) or load it if it exists.
        
        Args:
            n (int): The size of the matrix.
            k (int): Determines the number of samples to generate.
            forbidden_combinations (List[Tuple[int, int]]): List of forbidden (horizontal, vertical) bar index pairs.
            base_path (str): Base path to save or load the dataset.
            
        Returns:
            TensorDataset: The generated or loaded test dataset.
        """
        # Dynamically construct the save path based on matrix size and quantity
        save_path = os.path.join(base_path, f'matrix_size_{n}/k_{k}/test_set_two.pt')

        # Check if the dataset already exists
        if os.path.exists(save_path):
            print(f"Loading existing test set two from {save_path}")
            return torch.load(save_path)
        
        print(f"Test set two not found. Generating a new dataset and saving it to {save_path}")

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

        matrices_tensor = torch.tensor(np.array(matrices)).unsqueeze(1)
        col_labels_tensor = torch.tensor(np.array(col_labels))
        row_labels_tensor = torch.tensor(np.array(row_labels))

        dataset = TensorDataset(matrices_tensor, col_labels_tensor, row_labels_tensor)
        
        # Save the generated dataset
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(dataset, save_path)
        print(f"Test set two saved to {save_path}")

        return dataset
    
    @staticmethod
    def generate_test_set_three(
        n: int, 
        k: int, 
        max_bars: int,
        base_path: str = 'data/bar_matrix'
        ) -> TensorDataset:
        """
        Generate the third test set with up to the maximum specified number of bars or load it if it exists.
        
        Args:
            n (int): The size of the matrix.
            k (int): Determines the number of samples to generate.
            max_bars (int): The maximum number of bars in either direction.
            base_path (str): Base path to save or load the dataset.
            
        Returns:
            TensorDataset: The generated or loaded test dataset.
        """
        # Dynamically construct the save path based on matrix size and quantity
        save_path = os.path.join(base_path, f'matrix_size_{n}/k_{k}/test_set_three.pt')

        # Check if the dataset already exists
        if os.path.exists(save_path):
            print(f"Loading existing test set three from {save_path}")
            return torch.load(save_path)
        
        print(f"Test set three not found. Generating a new dataset and saving it to {save_path}")

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
        
        matrices_tensor = torch.tensor(np.array(matrices)).unsqueeze(1)
        col_labels_tensor = torch.tensor(np.array(col_labels))
        row_labels_tensor = torch.tensor(np.array(row_labels))

        dataset = TensorDataset(matrices_tensor, col_labels_tensor, row_labels_tensor)
        
        # Save the generated dataset
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(dataset, save_path)
        print(f"Test set three saved to {save_path}")

        return dataset