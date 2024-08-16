from torchvision import datasets, transforms

transform_MNIST = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))
        ])
train_dataset_f = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_MNIST)
test_dataset_f = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_MNIST)

transform_EMNIST = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.T),
            transforms.Lambda(lambda x : x.reshape(-1))
        ])
train_dataset_e = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform_EMNIST)
test_dataset_e = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform_EMNIST)

