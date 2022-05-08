import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def read_dataset(dataset):
    batch_size = 50

    if dataset == 'cifar10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='../../data', train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='../../data', train=False,
                                               download=True, transform=transform)
        idx_label = {0: 'airplane',
                     8: 'ship',
                     1: 'car',
                     9: 'truck',
                     2: 'bird',
                     6: 'frog',
                     3: 'cat',
                     5: 'dog',
                     4: 'deer',
                     7: 'horse',
                     }

    elif dataset == 'mnist':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.MNIST(root='../../data', train=True,
                                              download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='../../data', train=False,
                                             download=True, transform=transform)
        idx_label = {0: '0',
                     1: '1',
                     2: '2',
                     3: '3',
                     4: '4',
                     5: '5',
                     6: '6',
                     7: '7',
                     8: '8',
                     9: '9',
                     }

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    # part_te = torch.utils.data.random_split(testset, [1000, 9000])[0]
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    return trainloader, testloader, idx_label


def extract_feat(net, layer, dataloader, device, prune_type=None):
    """
    Ref: https://discuss.pytorch.org/t/how-can-i-extract-intermediate-layer-output-from-loaded-cnn-model/77301/3
    Extract features from a specific hidden layer.
    Args:
        net: network model, loaded from torch.load()
        layer: name of the layer, usually we use 'fc2'
        dataloader: pytorch dataloader from read_dataset function
        device: either 'cpu' or 'cuda'
        prune_type: either 'node' or 'weight' or None
    Returns: features (matrix with shape num_datapoint x num_dimension) and labels
    """

    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    net._modules[layer].register_forward_hook(get_features('feat'))

    feats, labels = [], []
    # loop through batches
    with torch.no_grad():
        for images, label in dataloader:
            outputs = net(images.to(device))
            feats.append(features['feat'].detach().cpu().numpy())
            labels.append(label.numpy())

    feats = np.concatenate(feats)
    labels = np.concatenate(labels)

    # remove the diemsions in the masks
    if prune_type == 'node':
        weight = net._modules[layer].weight.detach().cpu().numpy()
        node_mask = np.sum(np.abs(weight), axis=1) != 0
        feats = feats[:, node_mask]

    # # remove zero columns
    # zero_cols = np.isclose(np.std(feats, axis=0), np.zeros(feats.shape[1]))
    # feats = np.delete(feats, zero_cols, axis=1)

    return feats, labels


def get_apoz(feats, percent=True):
    apoz_stat = feats.shape[0] - np.count_nonzero(feats, axis=0)
    if percent:
        apoz_stat = apoz_stat / feats.shape[0]
    return apoz_stat

