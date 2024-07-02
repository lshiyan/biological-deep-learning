import torch
from torch.utils.data import DataLoader, TensorDataset


def compare_dataloaders(loader1:DataLoader, loader2:DataLoader) -> bool:
    """
    FUNCTION
    Compares 2 dataloaders
    @param
        loader1: a DataLoader
        loader2: a DataLoader
    @return
        True/False
    """
    for batch1, batch2 in zip(loader1, loader2):
        input1, label1 = batch1
        label1 = label1.item()
        input2, label2 = batch2
        label2 = label2.item()
        
        if label1 != label2: return False
        if not torch.equal(input1, input2): return False
    
    return True


def compare_datasets(dataset1: TensorDataset, dataset2: TensorDataset) -> bool:
    """
    Method to compares 2 datasets
    @param
        dataset1 = a TensorDataset
        dataset2 = a TensorDataset
    @return
        True/False
    """
    if len(dataset1) != len(dataset2): return False

    for tensor1, tensor2 in zip(dataset1.tensors, dataset2.tensors):
        if not torch.equal(tensor1, tensor2): return False
    return True