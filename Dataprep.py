from torchvision import datasets, transforms

transform_MNIST = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))
        ])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_MNIST)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_MNIST)

transform_EMNIST = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.T),
            transforms.Lambda(lambda x : x.reshape(-1))
        ])
train_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform_EMNIST)
test_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform_EMNIST)



