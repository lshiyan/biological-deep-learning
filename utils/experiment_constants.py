from enum import Enum

class LearningRules(Enum):
    """
    ENUM CLASS
    Defines learning rules constants
    @instance attr.
    """
    HEBBIAN_LEARNING_RULE = 'HEBBIAN'
    SANGER_LEARNING_RULE = 'SANGER'
    FULLY_ORTHOGONAL_LEARNING_RULE = 'ORTHOGONAL'
    
    
class LateralInhibitions(Enum):
    """
    ENUM CLASS
    Defines lateral inhibition constants
    @instance attr.
    """
    RELU_INHIBITION = 'RELU'
    SOFTMAX_INHIBITION = 'SOFTMAX'
    HOPFIELD_INHIBITION = 'HOPFIELD'
    WTA_INHIBITION = 'WTA'
    GAUSSIAN_INHIBITION = 'GAUSSIAN'
    NORM_INHIBITION = 'NORM'


class WeightGrowth(Enum):
    """
    ENUM CLASS
    Defines function types constants
    @instance attr.
    """
    LINEAR = 'LINEAR'
    SIGMOID = 'SIGMOID'
    

class LayerNames(Enum):
    """
    ENUM CLASS
    Defines names of layers of modules
    @instance attr.
    """
    INPUT = 'INPUT'
    HIDDEN = 'HIDDEN'
    OUTPUT = 'OUTPUT'
    

class ExperimentTypes(Enum):
    """
    ENUM CLASS
    Defines types of experiment to be ran
    @instance attr.
    """
    BASE = 'BASE'
    GENERALIZATION = 'GENERALIZATION'
    FORGET = 'FORGET'
    
    
class Purposes(Enum):
    """
    ENUM CLASS
    Defines purposes for loggin/file creation purposes
    @instance attr.
    """
    TRAIN_ACCURACY = 'TRAIN'
    TEST_ACCURACY = 'TEST'
    
    
class ExperimentPhases(Enum):
    """
    ENUM CLASS
    Defines name of dataset used
    @instance attr.
    """
    BASE = 'BASE'
    RECONSTRUCTION = 'RECONSTRUCTION'
    FREEZING_WEIGHTS = 'FREEZING_WEIGHTS'
    FORGET = 'FORGET'


class ParamInit(Enum):
    """
    ENUM CLASS
    Defines types of fc parameter instantiation
    @instance attr.
    """
    UNIFORM = 'UNIFORM'
    NORMAL = 'NORMAL'
    
    
class WeightDecay(Enum):
    TANH = 'TANH'
    SIMPLE = 'SIMPlE'
    NO_DECAY = 'NO_DECAY'
    
    
class BiasUpdate(Enum):
    HEBBIAN = 'HEBBIAN'
    SIMPLE = 'SIMPLE'
    NO_BIAS = 'NO_BIAS'
    
     
class DataSets(Enum):
    MNIST = 'MNIST'
    E_MNIST = 'E_MNIST'
    FASHION_MNIST = 'FASHION_MNIST'