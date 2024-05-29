import torch
from torch.utils.data import TensorDataset
import os
import pandas as pd
from layers.layer import NetworkLayer


"""
Class defining how the input dataset will be processed before feeding it to the network
"""
class InputLayer (NetworkLayer):
    """
    Constructor method InputLayer
    @param
        train_data (str) = train data filename (.ubyte)
        train_label (str) = train label filename (.ubyte)
        train_filename (str) = train data (img + label) filename (.csv)
        test_data (str) = test data filename (.ubyte)
        test_label (str) = test label filename (.ubyte)
        test_filename (str) = test data (img + label) filename (.csv)
    @attr.
        train_data (str) = train data filename (.ubyte)
        train_label (str) = train label filename (.ubyte)
        train_filename (str) = train data (img + label) filename (.csv)
        test_data (str) = test data filename (.ubyte)
        test_label (str) = test label filename (.ubyte)
        test_filename (str) = test data (img + label) filename (.csv)
    """
    def __init__(self, train_data, train_label, train_filename, test_data, test_label, test_filename):
        self.train_data = train_data
        self.train_label = train_label
        self.train_filename = train_filename
        self.test_data = test_data
        self.test_label = test_label
        self.test_filename = test_filename

    """
    Function to setup the taining dataset
    @param
    @return
        data_frame (torch.Tensor) = tensor containing dataset to train
    """
    def setup_train_data(self):
        InputLayer.convert(self.train_data, self.train_label, self.train_filename, 60000, 28)
        
        data_frame = pd.read_csv(self.train_filename, header=None)
        labels = torch.tensor(data_frame[0].values)
        data_frame = torch.tensor(data_frame.drop(data_frame.columns[0], axis=1).values, dtype=torch.float)
        data_frame /= 255
        return TensorDataset(data_frame, labels)


    """
    Function to setup the testing dataset
    @param
    @return
        data_frame (torch.Tensor) = tensor containing dataset to test
    """
    def setup_test_data(self):
        InputLayer.convert(self.test_data, self.test_label, self.test_filename, 10000, 28)
        data_frame = pd.read_csv(self.test_filename, header=None)
        labels = torch.tensor(data_frame[0].values)
        data_frame = torch.tensor(data_frame.drop(data_frame.columns[0], axis=1).values, dtype=torch.float)
        data_frame /= 255
        return TensorDataset(data_frame, labels)


    def __str__(self):
        return "The input processing layer of the network."


    """
    Class method to convert .ubyte files into a .csv file for ease of use
    """
    @classmethod
    def convert(cls, img_file, label_file, out_file, data_size, img_size):
        # Get absolute path of all the necessary files (img, label, out)
        project_root = os.getcwd()
        img_file = os.path.join(project_root, img_file)
        label_file = os.path.join(project_root, label_file)
        out_file = os.path.join(project_root, out_file)

        # Open all necessary files
        imgs = open(img_file, "rb")
        out = open(out_file, "w")
        labels = open(label_file, "rb")
        
        # Skip start of file because???
        # TODO: why skip bytes?
        imgs.read(16)
        labels.read(8)
        
        # Create a 2D list of images where each image is a 1D list where the first element is the label
        img_size = img_size**2
        images = []

        for i in range(data_size):
            image = [ord(labels.read(1))]
            for j in range(img_size):
                image.append(ord(imgs.read(1)))
            images.append(image)

        # Convert each image from 1D list to a comma seperated str and write it into out file
        for image in images:
            out.write(",".join(str(pix) for pix in image)+"\n")
        
        # Close files
        imgs.close()
        out.close()
        labels.close()