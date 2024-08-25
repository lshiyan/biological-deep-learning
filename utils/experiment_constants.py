from enum import Enum

class LearningRules(Enum):
    """
    ENUM CLASS
    Defines learning rules constants
    @instance attr.
    """
    HEBBIAN_LEARNING_RULE = 'HEBBIAN'
    SUPERVISED_HEBBIAN_LEARNING_RULE = 'SUPERVISED_HEBBIAN'
    SANGER_LEARNING_RULE = 'SANGER'
    FULLY_ORTHOGONAL_LEARNING_RULE = 'ORTHOGONAL'
    SOFT_HEBB_LEARNING_RULE = 'SOFT_HEBB'
    OUTPUT_CONTRASTIVE_LEARNING_RULE = 'OUTPUT_CONTRASTIVE'

    
    
class LateralInhibitions(Enum):
    """
    ENUM CLASS
    Defines lateral inhibition constants
    @instance attr.
    """
    RELU_INHIBITION = 'RELU'
    MAX_INHIBITION = 'MAX'
    EXP_SOFTMAX_INHIBITION = 'EXP_SOFTMAX'
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
    EXPONENTIAL = 'EXPONENTIAL'
    

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
    BAR_GENERALIZATION = 'BAR_GENERALIZATION'
    COLOURED_MNIST = 'COLOURED_MNIST'
    
    
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
    COLOURED_MNIST = 'COLOURED_MNIST'
    BAR_DATASET = 'BAR_DATASET'
    
    
class ActivationMethods(Enum):
    BASIC = 'BASIC'
    NORMALIZED = 'NORMALIZED'
    
    
class Focus(Enum):
    SYNASPSE = 'SYNAPSE'
    NEURON = 'NEURON'