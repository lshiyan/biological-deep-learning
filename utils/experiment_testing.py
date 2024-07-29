from typing import List


def lambda_list(*args: float) -> List[float]:
    test_list: List[float] = []
    
    if args:
        for num in args: test_list.append(num)
    else:
        for n in range(-4, 4):
            lamb = 2 ** n
            test_list.append(lamb)
    test_list.sort()
    return test_list
    
    
def lr_list(*args: float) -> List[float]:
    test_list: List[float] = []
    
    if args:
        for num in args: test_list.append(num)
    else:
        for n in range(2, 5):
            test_list.append(10 ** -n)
            test_list.append(5 * (10 ** -n))
    test_list.sort()
    return test_list
    
    
def dim_list(*args: float) -> List[float]:
    test_list: List[float] = []
    
    if args:
        for num in args: test_list.append(num)
    else:
        test_list.append(10)
        test_list.append(784)
        for n in range(4, 10):
            test_list.append(2 ** n)
    test_list.sort()
    return test_list
    
    
def eps_list(*args: float) -> List[float]:
    test_list: List[float] = []
    
    if args:
        for num in args: test_list.append(num)
    else:
        for n in range(2, 7):
            test_list.append(10 ** -n)
    test_list.sort()
    return test_list
    
    
def sigmoid_k_list(*args: float) -> List[float]:
    test_list: List[float] = []
    
    if args:
        for num in args: test_list.append(num)
    else:
        for n in range(1, 11):
            test_list.append(0.1 * n)
    test_list.sort()    
    return test_list