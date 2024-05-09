import os

"""
Converts image file and labels into .csv file
@param 
    img_file (str) = relative path to image file
    label_file (str) = relative path to label file
    out_file (str) = relative path to out file
    data_size (int) = number of data inputs to read
"""

def convert(img_file, label_file, out_file, data_size):
    # Get absolute path of all the necessary files
    project_root = os.getcwd()
    img_file = os.path.join(project_root, img_file)
    label_file = os.path.join(project_root, label_file)
    out_file = os.path.join(project_root, out_file)

    # Open all necessary files
    imgs = open(img_file, "rb")
    out = open(out_file, "w")
    labels = open(label_file, "rb")
    
    # Skip start of file because???
    # TODO: why skip bytes?
    imgs.read(16)
    labels.read(8)
    
    # Create a 2D list of images where each image is a 1D list where the first element is the label
    img_size = 28*28
    images = []

    for i in range(data_size):
        image = [ord(labels.read(1))]
        for j in range(img_size):
            image.append(ord(imgs.read(1)))
        images.append(image)

    # Convert each image from 1D list to a comma seperated str and write it into out file
    for image in images:
        out.write(",".join(str(pix) for pix in image)+"\n")
    
    # Close files
    imgs.close()
    out.close()
    labels.close()

# TODO: These lines should be written somewhere else
convert("data/mnist/train-images.idx3-ubyte", "data/mnist/train-labels.idx1-ubyte",
        "data/mnist/mnist_train.csv", 60000)
convert("data/mnist/t10k-images.idx3-ubyte", "data/mnist/t10k-labels.idx1-ubyte",
        "data/mnist/mnist_test.csv", 10000)

convert("data/fashion_mnist/train-images.idx3-ubyte", "data/fashion_mnist/train-labels.idx1-ubyte",
        "data/fashion_mnist/fashion-mnist_train.csv", 60000)
convert("data/fashion_mnist/t10k-images.idx3-ubyte", "data/fashion_mnist/t10k-labels.idx1-ubyte",
        "data/fashion_mnist/fashion-mnist_test.csv", 10000)