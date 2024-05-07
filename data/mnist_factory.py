import os

def convert(imgf, labelf, outf, n):
    project_root = os.getcwd() 
    imgf = os.path.join(project_root, imgf)
    labelf = os.path.join(project_root, labelf)
    outf = os.path.join(project_root, outf)

    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")
    
    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

convert("data/mnist/train-images.idx3-ubyte", "data/mnist/train-labels.idx1-ubyte",
        "data/mnist/mnist_train.csv", 60000)
convert("data/mnist/t10k-images.idx3-ubyte", "data/mnist/t10k-labels.idx1-ubyte",
        "data/mnist/mnist_test.csv", 10000)

convert("data/fashion_mnist/train-images-idx3-ubyte", "data/fashion_mnist/train-labels-idx1-ubyte",
        "data/fashion_mnist/fashion-mnist_train.csv", 60000)
convert("data/fashion_mnist/t10k-images-idx3-ubyte", "data/fashion_mnist/t10k-labels-idx1-ubyte",
        "data/fashion_mnist/fashion-mnist_test.csv", 10000)