from interfaces.layer import NetworkLayer
from torch.utils.data import TensorDataset


class InputLayer(NetworkLayer):
    """
    INTERFACE
    Input layer in ANN -> All input layers should implement this class
    
    @instance attr.
        PARENT ATTR.
            * Not used for this layer *
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
        super().__init__(0, 0, 0)


    @staticmethod
    def setup_data(data: str, label: str, filename: str, data_type: str) -> TensorDataset:
        """
        METHOD
        Function to setup requested dataset
        @param
            data: data filename
            label: label filename
            filename: data (img + label) filename
            data_type: which dataset to setup (train/test)
        @return
            tensor dataset containing (data, label)
        """
        raise NotImplementedError("This method has yet to be implemented.")
        

    @staticmethod
    def convert(img_file: str, 
                label_file: str, 
                out_file: str, 
                data_size: int) -> None:
        """
        CLASS METHOD
        Convert data file into more usable format
        @param
            img_file: path to image files
            label_file: path to file with labels
            out_file: path to more usable format file
            data_size: number of data inputs
        @return
            None
        """    
        raise NotImplementedError("This method has yet to be implemented.")
        

    # List of all methods from layers.NetworkLayer that are not needed for this layer
    # TODO: find a better way to implement the logic of having an input processing layer that still extends the layer.NetworkLayer interface
    def create_id_tensors(self): pass
    def update_weights(self): pass
    def update_bias(self): pass
    def forward(self, input, clamped_output): pass
    def _train_forward(self, x, clamped_output=None): pass
    def _eval_forward(self, x): pass
    def visualize_weights(self, path, num, use, name): pass
    def active_weights(self): pass