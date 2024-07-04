import math
from typing import List

def average(numbers: List[float]) -> float:
    """
    FUNCTION
    Returns the average of a list of numbers
    @param
        numbers: list of numbers
    @return
        average of list
    """
    return sum(numbers) / len(numbers) if numbers else 0


def variance(numbers: List[float], sample: bool = True) -> float:
    """
    FUNCTION
    Returns variance of list of numbers
    @param
        numbers: list of numbers
        sample: if sample is within data points
    @return
        variance of list
    """
    if len(numbers) in [0, 1]: return 0
    
    mean: float = average(numbers)
    squared_diffs: List[float] = [(x - mean) ** 2 for x in numbers]
    
    if sample:
        return sum(squared_diffs) / (len(numbers) - 1)
    else:
        return sum(squared_diffs) / len(numbers)
    

def min_diff(means: List[float]) -> float:
    """
    FUNCTION
    Returns smallest difference between means
    @param
        means: list of means
    @return
        smallest_diff: smallest difference
    """
    sorted_means: List[float] = sorted(means)
    
    smallest_diff: float = float('inf')
    prev: float = sorted_means[0]
    
    for mean in sorted_means:
        diff: float = 0
        
        diff = mean - prev
        
        if 0 < diff < smallest_diff:
            smallest_diff = diff
        
        prev = mean
    
    return smallest_diff


def min_sample(var: float, mean_diff: float) -> int:
    """
    FUNCTION
    Returns minimum number of samples for a given experiment
    @param
        var: variance gotten from previous try
        mean_diff: smallest difference between means
    @return
        minimum number of samples
    """
    
    return math.ceil((var ** 2) / (mean_diff ** 2)) if mean_diff != 0 else 0