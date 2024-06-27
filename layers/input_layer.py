from interfaces.layer import NetworkLayer


class InputLayer(NetworkLayer):
    """
    INTERFACE
    Input layer in ANN -> All input layers should implement this class
    
    @instance attr.
        PARENT ATTR.
            * Not used for this layer *
        OWN ATTR.
            train_data (str) = train data filename
            train_label (str) = train label filename
            train_filename (str) = train data (img + label) filename
            test_data (str) = test data filename
            test_label (str) = test label filename
            test_filename (str) = test data (img + label) filename
    """
    def __init__(self, train_data: str, 
                 train_label: str, 
                 train_filename: str, 
                 test_data: str, 
                 test_label: str, 
                 test_filename: str) -> None:
        """
        Constructor method
        @param
            train_data: train data filename
            train_label: train label filename
            train_filename: train data (img + label) filename
            test_data: test data filename
            test_label: test label filename
            test_filename: test data (img + label) filename
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


    def setup_data(self, data_type: str):
        """
        METHOD
        Function to setup requested dataset
        @param
            data_type: which dataset to setup (train/test)
        @return
            tensor dataset containing (data, label)
        """
        raise NotImplementedError("This method has yet to be implemented.")
        

    @classmethod
    def convert(cls, img_file: str, 
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