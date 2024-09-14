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


class WeightScale(Enum):
    WeightDecay = 1
    WeightNormalization = 2
    No = 3


class Inhibition(Enum):
    Softmax = 1
    RePU = 2


def oneHotEncode(labels, num_classes, device):
    one_hot_encoded = torch.zeros(len(labels), num_classes).to(device)
    one_hot_encoded.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot_encoded
