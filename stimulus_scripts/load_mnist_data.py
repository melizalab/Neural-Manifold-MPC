# Import Libraries
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset,Subset

def load_MNIST_data(data_path='stimuli/MNIST',download=False):
    # Transform images to 28x28 images with pixel values between 0 and 1
    transform = transforms.Compose([
                transforms.Resize((28,28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

    # Classic MNIST train and testing sets
    mnist_train = datasets.MNIST(data_path, train=True, download=download, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=download, transform=transform)

    # Combine to make a full set
    mnist_full = ConcatDataset([mnist_train, mnist_test])
    full_len = len(mnist_full)

    '''
    Half of the data is reserved for constructing the latent representation
    of the visual stimuli (stimuli_mnist), the second half is reserved for
    training and validation of the SNN on classification. This prevents data
    leakage from confounding results.
    '''
    split_size = full_len // 2
    stimuli_mnist = Subset(mnist_full, range(split_size))
    snn_mnist = Subset(mnist_full,  range(split_size, full_len))

    return stimuli_mnist,snn_mnist


def train_val_test_split(dataset, train_ratio, val_ratio, test_ratio,seed=None):
    # Ensure the ratios sum to 1
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1!"

    # Get the size of the dataset and create a list of indices
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    if seed is not None:
        np.random.seed(seed)
        np.random.shuffle(indices)

    # Calculate split sizes
    train_end = int(train_ratio * dataset_size)
    val_end = train_end + int(val_ratio * dataset_size)

    # Create subsets using indices
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    return train_subset, val_subset, test_subset
