import torch
from torch.utils.data import TensorDataset
import os
import pandas as pd
from layers.layer import NetworkLayer
from typing import IO, List


class InputLayer (NetworkLayer):
    """
    Class defining how the input dataset will be processed before feeding it to the network
    
    @instance attr.
        PARENT ATTR.
            * Not used for this layer *
        OWN ATTR.
            train_data (str) = train data filename (.ubyte)
            train_label (str) = train label filename (.ubyte)
            train_filename (str) = train data (img + label) filename (.csv)
            test_data (str) = test data filename (.ubyte)
            test_label (str) = test label filename (.ubyte)
            test_filename (str) = test data (img + label) filename (.csv)
    """

    def __init__(self, train_data: str, train_label: str, train_filename: str, test_data: str, test_label: str, test_filename: str) -> None:
        """
        Constructor method
        @param
            train_data: train data filename (.ubyte)
            train_label: train label filename (.ubyte)
            train_filename: train data (img + label) filename (.csv)
            test_data: test data filename (.ubyte)
            test_label: test label filename (.ubyte)
            test_filename: test data (img + label) filename (.csv)
        @return
            None
        """
        super().__init__(0, 0, 0)
        self.train_data: str = train_data
        self.train_label: str = train_label
        self.train_filename: str = train_filename
        self.test_data: str = test_data
        self.test_label: str = test_label
        self.test_filename: str = test_filename


    def setup_train_data(self) -> TensorDataset:
        """
        Function to setup the training dataset
        @param
            None
        @return
            tensor dataset containing (data, label)
        """
        if not os.path.exists(self.train_filename):
            InputLayer.convert(self.train_data, self.train_label, self.train_filename, 60000, 28)
        
        data_frame: pd.DataFrame = pd.read_csv(self.train_filename, header=None, on_bad_lines='skip')
        labels: torch.Tensor = torch.tensor(data_frame[0].values)
        data_tensor: torch.Tensor = torch.tensor(data_frame.drop(data_frame.columns[0], axis=1).values, dtype=torch.float)
        data_tensor /= 255
        
        return TensorDataset(data_tensor, labels)


    def setup_test_data(self) -> TensorDataset:
        """
        Function to setup the testing dataset
        @param
            None
        @return
            tensor dataset containing (data, label)
        """
        if not os.path.exists(self.test_filename):
            InputLayer.convert(self.test_data, self.test_label, self.test_filename, 10000, 28)
            
        data_frame: pd.DataFrame = pd.read_csv(self.test_filename, header=None, on_bad_lines='skip')
        labels: torch.Tensor = torch.tensor(data_frame[0].values)
        data_tensor: torch.Tensor = torch.tensor(data_frame.drop(data_frame.columns[0], axis=1).values, dtype=torch.float)
        data_tensor /= 255
        
        return TensorDataset(data_tensor, labels)


    @classmethod
    def convert(cls, img_file: str, label_file: str, out_file: str, data_size: int, img_size: int):
        """
        CLASS METHOD
        Convert .ubyte files into a .csv file for ease of use
        @param
            img_file: path to image files
            label_file: path to file with labels
            out_file: path to .cvs file that merges data and label
            data_size: number of data inputs
            img_size: size of image
        @return
        """    
        # Get absolute path of all the necessary files (img, label, out)
        project_root: str = os.getcwd()
        img_file: str = os.path.join(project_root, img_file)
        label_file: str = os.path.join(project_root, label_file)
        out_file: str = os.path.join(project_root, out_file)

        # Open all necessary files
        imgs: IO = open(img_file, "rb")
        out: IO = open(out_file, "w")
        labels: IO = open(label_file, "rb")
        
        # Skip header bytes
        imgs.read(16)
        labels.read(8)
        
        # Create a 2D list of images where each image is a 1D list where the first element is the label
        img_size = img_size**2
        images: List[List[int]] = []

        for _ in range(data_size):
            image: List[int] = [int.from_bytes(labels.read(1), byteorder='big')]
            for _ in range(img_size):
                image.append(int.from_bytes(imgs.read(1), byteorder='big'))
            images.append(image)

        # Convert each image from 1D list to a comma-seperated str and write it into out file
        for image in images:
            out.write(",".join(str(pix) for pix in image) + "\n")
        
        # Close files
        imgs.close()
        out.close()
        labels.close()