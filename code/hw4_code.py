"""
HW4 code — copy your implementations from Homework 4 here.

If you had trouble with HW4, come to a TA and we'll give you the solution.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

import hyperparameters as hp


BANNER_ID = 000000000 # <- must match student.py
torch.manual_seed(BANNER_ID)
np.random.seed(BANNER_ID)


# ========================================================================
#  SceneDataset — loads the 15-scenes dataset
#
class SceneDataset:
    """Load the 15-scenes dataset using ImageFolder (given, do not modify).

    Arguments:
        data_dir   -- path to dataset (must contain train/, val/, test/)
        batch_size -- batch size for DataLoaders
        image_size -- resize images to this square size

    After construction, provides:
        .train_loader  -- DataLoader for training set (shuffled)
        .val_loader    -- DataLoader for validation set
        .test_loader   -- DataLoader for test set
        .classes       -- list of class name strings
        .num_classes   -- number of classes
    """

    def __init__(self, data_dir, batch_size=hp.ENDTOEND_BATCH_SIZE, image_size=hp.ENDTOEND_IMAGE_SIZE):
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform)
        val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform)
        test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform)

        nw = 0 if os.name == 'nt' else 4
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=nw)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=nw)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=nw)
        self.classes = train_set.classes
        self.num_classes = len(self.classes)


# ========================================================================
#  Training loop
#
def train_loop(model, train_loader, optimizer, loss, epochs,
               device, val_loader=None, tasklabel="", on_epoch_end=None):
    """Train a model and optionally evaluate on a validation set each epoch.

    Arguments:
        model:          nn.Module to train
        train_loader:   DataLoader for training data
        optimizer:      torch.optim optimizer
        loss:           loss function (e.g., nn.CrossEntropyLoss())
        epochs:         number of training epochs
        device:         torch.device passed from main.py
        val_loader:     optional DataLoader for validation
        tasklabel:      string prefix for print output
        on_epoch_end:   optional callback, called as on_epoch_end(epoch, model)

    Returns:
        List of training accuracies     (float, one per epoch).
        List of validation accuracies   (float, one per epoch); empty if val_loader is None.
    """
    # TODO: paste your HW4 implementation
    raise NotImplementedError


# ========================================================================
#  CropRotationDataset — generates random rotated crops
#
class CropRotationDataset(Dataset):
    """Create a dataset of random rotated crops from images.

    Arguments:
        device     -- torch device for GPU-accelerated augmentation
        data_dir   -- path to a directory of images (with or without class subfolders)
        num_crops  -- total number of crops to generate per epoch
        crop_size  -- spatial size of each crop
        rotation   -- if True (default), apply random rotation and return rotation label
        batch_size -- batch size for the DataLoader

    After construction, provides:
        .train_loader  -- DataLoader for this dataset (shuffled)
        .classes       -- list of class name strings
        .num_classes   -- number of classes
    """

    def __init__(self, device, data_dir, num_crops=hp.ROTATION_NUM_CROPS,
                 crop_size=hp.ROTATION_CROP_SIZE, rotation=True,
                 batch_size=hp.ROTATION_BATCH_SIZE):
        # TODO: paste your HW4 implementation
        raise NotImplementedError

    def __len__(self):
        # TODO: paste your HW4 implementation
        raise NotImplementedError

    def __getitem__(self, idx):
        # TODO: paste your HW4 implementation
        raise NotImplementedError
