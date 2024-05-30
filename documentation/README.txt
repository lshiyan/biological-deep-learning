The followng document contains a detailed explanation of the code's functionality and usage.

Structure Explanation
    data: folder containing all the data required for the training/testing of the models
    experiments: folder containing all the code to make custom experiments
    layers:folder containing all the code for the layers used within the models
    models: folder containing all the diferent models created
    results: folder ssh keys
    ssh

Code Explanation (BaseHebbianExperiment)
    Running an experiment
        - experiment = BaseHebbianExperiment(args) -> experiemnt.train() -> experiment.test()
        - Create an instance of an experiment, passing it the parameters as needed, then train and test it using the data
        - Our experiment uses the ADAM optimizer and the cross entropy loss function
    How Model Works

    Hebbian Layer Functionality

    Classification Layer Functionality



Code Implementation
    Create New Experiment
        - Extend experiment.Experiment
        - Override __intit(args), optimizer(), loss_function(), train(), test(), print_exponential_averages()*
        - Methods:
            - Constructor: creates a new instance of the class
            - optimizer(): defines optimzer to be used during training
            - loss_function(): defines loss function of the model
            - set_scheduler(): defines scheduler for each layer **CURRENTLY NOT IN USE**
            - train(): defines how the model will be trained
            - test(): defines how the model will be tested and the accurac of the model
            - print_exponential_averages(): prints exponential averages
            - active_weights(beta): returns number of weights above a certain level
            - visualize_weights(): vizualizes the features learned
            - *Class method* one_hot_encode(labels, num_classes): encodes the labels into a tensor
    Create New Network

    Create New layer







