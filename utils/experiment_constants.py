from enum import Enum, auto

class LearningRules(Enum):
    """
    ENUM CLASS
    Defines learning rules constants
    @instance attr.
    """
    HEBBIAN_LEARNING_RULE = auto()
    SANGER_LEARNING_RULE = auto()
    FULLY_ORTHOGONAL_LEARNING_RULE = auto()
    
    
class LateralInhibitions(Enum):
    """
    ENUM CLASS
    Defines lateral inhibition constants
    @instance attr.
    """
    RELU_INHIBITION = auto()
    SOFTMAX_INHIBITION = auto()
    HOPFIELD_INHIBITION = auto()


class FunctionTypes(Enum):
    """
    ENUM CLASS
    Defines function types constants
    @instance attr.
    """
    LINEAR = auto()
    SIGMOID = auto()
    
    