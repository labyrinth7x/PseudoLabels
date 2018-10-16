import torchvision as tv
import numpy as np


def get_dataset(args, transform_train, transform_val):
    # prepare dataset
    mnist_train_val = tv.datasets.MNIST(args.train_root, train=True, download=args.download)

    # get train_unlabeled/train_labeled/val dataset
    train_labeled_indexes, train_unlabeled_indexes, val_indexes = train_val_split(mnist_train_val.train_labels, 1000, args.labeled_num)

    train_labeled = MnistTrainLabeled(args, train_labeled_indexes, train=True, transform=transform_train)
    train_unlabeled = MnistTrainLabeled(args, train_unlabeled_indexes, train=True, transform=transform_train)
    val = MnistVal(args, val_indexes, train=True, transform=transform_val)

    return train_labeled, train_unlabeled, val

def train_val_split(train_val, val_num, labeled_num):
    train_val = np.array(train_val)
    train_labeled_indexes = []
    train_unlabeled_indexes = []
    val_indexes = []

    for id in range(10):
        indexes = np.where(train_val == id)
        np.random.shuffle(indexes)
        val_indexes.extend(indexes[:val_num])
        train_labeled_indexes(indexes[val_num:val_num+labeled_num])
        train_unlabeled_indexes(indexes[val_num+labeled_num])
    np.random.shuffle(train_labeled_indexes)
    np.random.shuffle(train_unlabeled_indexes)
    np.random.shuffle(val_indexes)

    return train_labeled_indexes, train_unlabeled_indexes, val_indexes


class MnistTrainLabeled(tv.datasets.MNIST):
    def __init__(self, args, train_indexes, train=True, transform=None, target_transform=None, download=False):
        super(MnistTrainLabeled, self).__init__(args.train_root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.args = args
        self.train_data = self.train_data[train_indexes]
        self.train_labels = self.train_labels[train_indexes]


class MnistTrainUnlabeled(tv.datasets.MNIST):
    def __init__(self, args, train_indexes, train=True, transform=None, target_transform=None, download=False):
        super(MnistTrainLabeled, self).__init__(args.train_root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.args= args
        self.train_data = self.train_data[train_indexes]
        self.train_labels = self.train_labels[train_indexes]


class MnistVal(tv.datasets.MNIST):
    def __init__(self, args, val_indexes, train=True, transform=None, target_transform=None, download=False):
        super(MnistVal, self).__init__(args.train_root, train=train, transform = transform, target_transform=target_transform, download=download)
        self.args = args
        self.train_data = self.train_data[val_indexes]
        self.train_labels = self.train_labels[val_indexes]
