from enum import Enum
import torch

class ImageType(Enum):
    Gray = 1
    RGB = 2


class LearningRule(Enum):
    Supervised = 1
    OutputContrastiveSupervised = 2
    Orthogonal = 3
    FullyOrthogonal = 4
    SoftHebb = 5
    SoftHebbOutputContrastive = 6

class WeightGrowth(Enum):
    Default = 1
    Linear = 2
    Sigmoidal = 3
    Exponential = 4

class WeightScale(Enum):
    WeightDecay = 1
    WeightNormalization = 2
    No = 3


class Inhibition(Enum):
    Softmax = 1
    RePU = 2

class InputProcessing(Enum):
    No = None
    Whiten = 1

def oneHotEncode(labels, num_classes, device):
    one_hot_encoded = torch.zeros(len(labels), num_classes).to(device)
    one_hot_encoded.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot_encoded


def cnn_output_formula_1D(size, kernel, padding, dilation, stride):
    return int((size + 2 * padding - dilation * (kernel -1) - 1)/stride + 1)

def cnn_output_formula_2D(shape, kernel, padding, dilation, stride):
    h = cnn_output_formula_1D(shape[0],  kernel, padding, dilation, stride)
    w = cnn_output_formula_1D(shape[1],  kernel, padding, dilation, stride)
    return h, w
