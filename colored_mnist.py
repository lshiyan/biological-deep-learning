import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Color definitions in RGB and CMYK
colors = {
    'green': {
        'rgb': np.array([178, 217, 178]),
        'cmyk': np.array([0.18, 0.0, 0.18, 0.15])
    },
    'beige': {
        'rgb': np.array([247, 234, 190]),
        'cmyk': np.array([0.0, 0.05, 0.23, 0.03])
    },
    'red': {
        'rgb': np.array([250, 112, 100]),
        'cmyk': np.array([0.0, 0.55, 0.60, 0.02])
    },
    'blue': {
        'rgb': np.array([31, 90, 255]),
        'cmyk': np.array([0.88, 0.65, 0.0, 0.0])
    },
    'purple': {
        'rgb': np.array([175, 151, 194]),
        'cmyk': np.array([0.10, 0.22, 0.0, 0.24])
    }
}

# Load the MNIST dataset
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(mnist_data, batch_size=1, shuffle=True)

# Function to apply color to MNIST digit
def apply_color(image, color):
    brightness = image.numpy().squeeze()  # Get brightness from MNIST grayscale image
    colored_image = np.zeros((28, 28, 3))  # Create an empty 3-channel image (RGB)
    
    for i in range(3):  # Apply color based on brightness
        colored_image[:, :, i] = brightness * color['rgb'][i] / 255.0

    return colored_image

# Apply each color to a sample digit and display it
for color_name, color_values in colors.items():
    data_iter = iter(data_loader)
    sample_image, label = next(data_iter)
    
    # Apply color to the MNIST image
    colored_image = apply_color(sample_image, color_values)
    
    # Display the original and colored image
    plt.figure(figsize=(6, 3))
    
    # Original grayscale image
    plt.subplot(1, 2, 1)
    plt.title(f"Original MNIST {label.item()}")
    plt.imshow(sample_image.squeeze(), cmap='gray')
    plt.axis('off')
    
    # Colored image
    plt.subplot(1, 2, 2)
    plt.title(f"Colored with {color_name.capitalize()}")
    plt.imshow(colored_image)
    plt.axis('off')
    
    plt.show()