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
        PARENT ATTR.
            * Not used for this layer *
        OWN ATTR.
            train_data (str) = train data filename (.ubyte)
            train_label (str) = train label filename (.ubyte)
            train_filename (str) = train data (img + label) filename (.csv)
            test_data (str) = test data filename (.ubyte)
            test_label (str) = test label filename (.ubyte)
            test_filename (str) = test data (img + label) filename (.csv)
    @return
        ___ (layers.InputLayer) = returns instance of InputLayer
    """
    def __init__(self, train_data, train_label, train_filename, test_data, test_label, test_filename, device_id):
        super().__init__(0, 0, device_id) # WHY WAS THE DEVICE ID SET TO 0???? 
        self.train_data = train_data
        self.train_label = train_label
        self.train_filename = train_filename
        self.test_data = test_data
        self.test_label = test_label
        self.test_filename = test_filename


    # TODO 1: CAN REDUCE THE TWO FUNCTIONS INTO 1, BY PROVIDING A HYPERPARAM INDICATING TRAINING OR NOT

    # Static method to log bad lines
    @staticmethod
    def log_bad_lines(line, line_number):
        log_dir = os.path.dirname(os.path.abspath(__file__))
        log_file_path = os.path.join(log_dir, "bad_lines.log")
        with open(log_file_path, "a") as log_file:                  # Open log file in append mode
            log_file.write(f"Line {line_number}: {line}\n")         # Write bad line and its line number



    """
    Function to setup the dataset
    @param
    @return
        data_frame (torch.Tensor) = tensor containing dataset
    """
    def setup_data(self, dataset_type='train'):
        if dataset_type == 'train':
            filename = self.train_filename
            data_file = self.train_data
            label_file = self.train_label
            data_size = 60000
        else:
            filename = self.test_filename
            data_file = self.test_data
            label_file = self.test_label
            data_size = 10000

        if not os.path.exists(filename):
            InputLayer.convert(data_file, label_file, filename, data_size, 28)
        
        data_frame = pd.read_csv(filename, header=None, on_bad_lines=InputLayer.log_bad_lines if dataset_type == 'train' else 'skip', engine='python')
        labels = torch.tensor(data_frame[0].values)
        data_frame = torch.tensor(data_frame.drop(data_frame.columns[0], axis=1).values, dtype=torch.float)
        data_frame /= 255
        return TensorDataset(data_frame, labels)



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
        
        # Skip header bytes
        imgs.read(16)
        labels.read(8)
        
        # Create a 2D list of images where each image is a 1D list where the first element is the label
        img_size = img_size**2
        images = []

        for i in range(data_size):
            image = [int.from_bytes(labels.read(1), byteorder='big')]
            for j in range(img_size):
                image.append(int.from_bytes(imgs.read(1), byteorder='big'))
            images.append(image)

        # Convert each image from 1D list to a comma-seperated str and write it into out file
        for image in images:
            out.write(",".join(str(pix) for pix in image) + "\n")
        
        # Close files
        imgs.close()
        out.close()
        labels.close()


    # List of all methods from layers.NetworkLayer that are not needed for this layer
    # TODO: find a better way to implement the logic of having an input processing layer that still extends the layer.NetworkLayer interface
    def create_id_tensors(self): pass
    def set_scheduler(self): pass
    def visualize_weights(self, path, num, use): pass
    def update_weights(self): pass
    def update_bias(self): pass
    def forward(self): pass
    def active_weights(self): pass
    def _train_forward(self, x, clamped_output): pass
    def _eval_forward(self, x, clamped_output): pass