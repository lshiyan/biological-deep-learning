import argparse
import torch

if __name__=="main":
    parser = argparse.ArgumentParser(description='Biological deep learning')
    
    parser.add_argument('--test', type=str, default="test")
    
    print('test')