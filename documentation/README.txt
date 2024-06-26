The followng document contains a detailed explanation of the code's functionality and usage.

Structure Explanation
    data: folder containing all the data required for the training/testing of the models
    experiments: folder containing all the code to make custom experiments
    layers:folder containing all the code for the layers used within the models
    models: folder containing all the diferent models created
    results: folder containing the results of the experiments???
    ssh: folder ssh keys

Code Explanation (BaseHebbianExperiment)
    Running an experiment
        - experiment = BaseHebbianExperiment(args) -> experiemnt.train() -> experiment.test()
        - Create an instance of an experiment, passing it the parameters as needed, then train and test it using the data
        - Our experiment uses the ADAM optimizer and the cross entropy loss function
    
    
    How Model Works
        - Our model has 3 layers 
            => Input layer that process the input into a .csv file that is easier to use
            => Hebbian Layer/ Hidden Layer that uses hebbian learning to learn features
            => Classification layer that determines which label should be associated with the input
        - Data is first proceesed through the input layer, then it is given to the hebbian layer where multiple neurons wil learn features associated with the input data, and lastly, the classification will classify the learned features
        - The moddel has a list of (name, layer) for each of its layers and all the parameters for each layer as attributes



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
        - Extend models.Network
        - Override forward(x, clamped_output)
        - Methods:
            - Constructor: creates a new instance of the class
            - add_layer(name, layer):
            - get_layer(name): get the layer with name
            - named_layer(): get all (name, layer) tuples
            - layers(): get a list of all layers
            - set_scheduler(): set the scheduler for each layer **CURRENTLY NOT IN USE**
            - visualize_weights(): vizualizes the features learned for each layer
            - active_weights(beta): returns number of weights above a certain level for each layer
            - forward(): defines how data moves within the network


    Create New layer
        - Extend layers.NetworkLayer
        - Override set_scheduler(), visualize_weights(), update_weights(input, output, clamped_output), update_bias(output)
        - Methods:
            - Constructor:
            - create_id_tensors(): create an ID tensor
            - set_scheduler(): defines a scheduler and sets it forthis layer **CURRENTLY NOT IN USE**
            - visualize_weights(): visualizes the weights of a given layer
            - update_weights(input, output, clamped_output): method defining how the layer learns
            - update_bias(output): define how bias is updated **CURRENTLY NOT IN USE**
            - forward(x, clamped_output): defines how data flows through this layer
            - active_weights(beta): returns nunmber of weights that are considered active



***** TESTING PURPOSES *****
- notebook.ipynb can and should be used to test small parts of code and look at runtime/imediate feedback
- cmd python run.py for use on cpu, isc train base_hebbian_learning.isc for SC



#################### WHAT NEEDSIMPROVEMENT ####################
- Check the TODOs and NOTEs within the code
- For all functions, need to have error catching



