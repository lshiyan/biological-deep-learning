from typing import List, Optional


def lambda_list(num: Optional[float] = None) -> List[float]:
    test_list: List[float] = []
    
    if num:
        test_list.append(num)
    else:
        for n in range(1, 20):
            test_list.append(n)
            test_list.append(1/n)
    test_list.sort()
    return test_list
    
    
def lr_list(num: Optional[float] = None) -> List[float]:
    test_list: List[float] = []
    
    if num:
        test_list.append(num)
    else:
        for n in range(2, 5):
            test_list.append(10 ** -n)
            test_list.append(5 * (10 ** -n))
    test_list.sort()
    return test_list
    
    
def dim_list(num: Optional[float] = None) -> List[float]:
    test_list: List[float] = []
    
    if num:
        test_list.append(num)
    else:
        test_list.append(10)
        test_list.append(784)
        for n in range(4, 10):
            test_list.append(2 ** n)
    test_list.sort()
    return test_list
    
    
def eps_list(num: Optional[float] = None) -> List[float]:
    test_list: List[float] = []
    
    if num:
        test_list.append(num)
    else:
        for n in range(2, 7):
            test_list.append(10 ** -n)
    test_list.sort()
    return test_list
    
    
def sigmoid_k_list(num: Optional[float] = None) -> List[float]:
    test_list: List[float] = []
    
    if num:
        test_list.append(num)
    else:
        for n in range(1, 11):
            test_list.append(0.1 * n)
    test_list.sort()    
    return test_list