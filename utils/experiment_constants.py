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


class FunctionTypes(Enum):
    """
    ENUM CLASS
    Defines function types constants
    @instance attr.
    """
    LINEAR = 'LINEAR'
    SIGMOID = 'SIGMOID'
    
