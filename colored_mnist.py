import os

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils

def color_grayscale_arr(arr, color_choice):
    """Converts grayscale image to cyan, magenta, yellow, or black."""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])

    if color_choice == 'cyan':
        arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype), arr, arr], axis=2)
    elif color_choice == 'magenta':
        arr = np.concatenate([arr, np.zeros((h, w, 1), dtype=dtype), arr], axis=2)
    elif color_choice == 'yellow':
        arr = np.concatenate([arr, arr, np.zeros((h, w, 1), dtype=dtype)], axis=2)
    else:  # black
        arr = np.concatenate([arr, arr, arr], axis=2)  # black is simply grayscale repeated across channels
    
    return arr


class ColoredMNIST(datasets.VisionDataset):
  """
  Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf
  Args:
    root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
    env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
    transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.
  """
  def __init__(self, root='./data', env='train1', transform=None, target_transform=None):
    super(ColoredMNIST, self).__init__(root, transform=transform,
                                target_transform=target_transform)

    self.prepare_colored_mnist()
    if env in ['train1', 'train2', 'test']:
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')
    elif env == 'all_train':
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', 'train1.pt')) + \
                               torch.load(os.path.join(self.root, 'ColoredMNIST', 'train2.pt'))
    else:
      raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, test, and all_train')

  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data_label_tuples[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.data_label_tuples)
  
  def prepare_colored_mnist(self):
    colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
    if os.path.exists(os.path.join(colored_mnist_dir, 'train1.pt')) \
            and os.path.exists(os.path.join(colored_mnist_dir, 'train2.pt')) \
            and os.path.exists(os.path.join(colored_mnist_dir, 'test.pt')):
        print('Colored MNIST dataset already exists')
        return

    print('Preparing Colored MNIST')
    train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

    train1_set = []
    train2_set = []
    test_set = []
    color_map = {0: 'cyan', 1: 'magenta', 2: 'yellow', 3: 'black'}
    
    for idx, (im, label) in enumerate(train_mnist):
        if idx % 10000 == 0:
            print(f'Converting image {idx}/{len(train_mnist)}')
        im_array = np.array(im)

        # Assign a binary label y to the image based on the digit
        binary_label = 0 if label < 5 else 1

        # Flip label with 25% probability
        if np.random.uniform() < 0.25:
            binary_label = binary_label ^ 1

        # Assign a color based on the binary label
        if binary_label == 0:
            color_choice = color_map[idx % 4]
        else:
            color_choice = color_map[(idx % 4) ^ 1]

        print(f'Image {idx}: Label {binary_label}, Color {color_choice}')  # Debugging line

        colored_arr = color_grayscale_arr(im_array, color_choice)

        if idx < 20000:
            train1_set.append((Image.fromarray(colored_arr), binary_label))
        elif idx < 40000:
            train2_set.append((Image.fromarray(colored_arr), binary_label))
        else:
            test_set.append((Image.fromarray(colored_arr), binary_label))

    os.makedirs(colored_mnist_dir, exist_ok=True)
    torch.save(train1_set, os.path.join(colored_mnist_dir, 'train1.pt'))
    torch.save(train2_set, os.path.join(colored_mnist_dir, 'train2.pt'))
    torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))



def plot_dataset_digits(dataset):
  fig = plt.figure(figsize=(13, 8))
  columns = 6
  rows = 3
  # ax enables access to manipulate each of subplots
  ax = []

  for i in range(columns * rows):
    img, label = dataset[i]
    # create subplot and append to ax
    ax.append(fig.add_subplot(rows, columns, i + 1))
    ax[-1].set_title("Label: " + str(label))  # set title
    plt.imshow(img)

  plt.show()  # finally, render the plot

# Then use the modified class
train1_set = ColoredMNIST(root='./data', env='train1')
plot_dataset_digits(train1_set)