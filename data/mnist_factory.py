import os

"""
Converts image file and labels into .csv file
@param 
    img_file (str) = relative path to image file
    label_file (str) = relative path to label file
    out_file (str) = relative path to out file
    data_size (int) = number of data inputs to read
"""
def convert(img_file, label_file, out_file, data_size, img_size):
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
    img_size = img_size**2
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