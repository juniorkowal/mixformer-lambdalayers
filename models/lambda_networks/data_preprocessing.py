import torchvision
from torch.utils.data import DataLoader
import os
import torch
import matplotlib.pyplot as plt

def data_preprocessing(args):
    '''
    Preprocessing the data and creating the dataloaders for training and testing
    Args:
        args: user input

    Returns:
        train_loader: iterator containing training data
        test_loader: iterator containing test data
        input_lst: first training data batch

    '''
    # Define training, testing and plotting transformations
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    plot_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ToTensor()])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    if args.dataset == 'CIFAR100':
        dataset_class = torchvision.datasets.CIFAR100
        root_train = ".\\CIFAR_100_train"
        root_test = ".\\CIFAR_100_test"
    else:
        dataset_class = torchvision.datasets.CIFAR10
        root_train = ".\\CIFAR_10_train"
        root_test = ".\\CIFAR_10_test"

    # Preparing the dataset
    download_train = not os.path.exists(root_train)
    download_test = not os.path.exists(root_test)

    train_dataset = dataset_class(root=root_train, train=True, download=download_train, transform=train_transform)
    plot_dataset = dataset_class(root=root_train, train=True, download=False, transform=plot_transform)
    test_dataset = dataset_class(root=root_test, train=False, download=download_test, transform=test_transform)

    generator = torch.Generator().manual_seed(42)
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2], generator=generator)

    # Create train and test data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.b_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.b_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.b_size, shuffle=True)


    # Obtain one data training batch
    # input_lst, label = next(iter(plot_loader))

    # Plot a random group of images from the training set
    # plt.figure()
    # for i in range(args.n_rows_plot*args.n_col_plot):
    #     plt.subplot(args.n_rows_plot, args.n_col_plot, i+1)
    #     plt.imshow(input_lst[i].permute(1, 2, 0))
    # plt.show()

    return train_loader, valid_loader, test_loader
