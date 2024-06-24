from typing import List

def average(numbers: List[float]) -> float:
    """
    Returns the average of a list of numbers

    @param
        numbers: list of numbers
    @return
        average of list
    """
    return sum(numbers) / len(numbers) if numbers else 0


def variance(numbers: List[float], sample: bool = True) -> float:
    """
    Returns variance of list of numbers

    @param
        numbers: list of numbers
        sample: if sample is within data points
    @return
        variance of list
    """
    if len(numbers) == 0: return 0
    
    mean: float = average(numbers)
    squared_diffs: float = [(x - mean) ** 2 for x in numbers]
    
    if sample:
        return sum(squared_diffs) / (len(numbers) - 1)
    else:
        return sum(squared_diffs) / len(numbers)