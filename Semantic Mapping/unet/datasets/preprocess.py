import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import itertools

import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


# extract the label training images
def load_data_labelID(path_label, subfolder, transform, batch_size, shuffle=False):
    # create the label dataset
    dataset = datasets.ImageFolder(path_label, transform)
    index = dataset.class_to_idx[subfolder]
    n = 0
    m = 0
    for i in range(dataset.__len__()):
        if index != dataset.imgs[n][1]:
            del dataset.imgs[n]
            n = n - 1
        else:
            if m % 3 != 2:
                del dataset.imgs[n]
                n = n - 1
            m = m + 1
        n = n + 1
    len_dataset = dataset.__len__()
    Dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16, pin_memory=True)


    return (len_dataset, Dataloader)


# extract the label training images
def load_data_label(path_label, subfolder, transform, batch_size, shuffle=False):
    # create the label dataset
    dataset = datasets.ImageFolder(path_label, transform)
    index = dataset.class_to_idx[subfolder]
    n = 0
    m = 0
    for i in range(dataset.__len__()):
        if index != dataset.imgs[n][1]:
            del dataset.imgs[n]
            n = n - 1
        else:
            if m % 3 != 0:
                del dataset.imgs[n]
                n = n - 1
            m = m + 1
        n = n + 1
    len_dataset = dataset.__len__()
    Dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16, pin_memory=True)


    return (len_dataset, Dataloader)


# extract raw training images
def load_data_raw(path, subfolder, transform, batch_size, shuffle=False):
    dataset = datasets.ImageFolder(path, transform)
    index = dataset.class_to_idx[subfolder]
    n = 0
    for i in range(dataset.__len__()):
        if index != dataset.imgs[n][1]:
            del dataset.imgs[n]
            n = n - 1
        n = n + 1
    len_dataset = dataset.__len__()
    Dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16, pin_memory=True)


    return (len_dataset, Dataloader)


# extract combined training images
def load_data_train_combined(path_label, path_raw, subfolder, transform, batch_size, shuffle=False):
    # create the label dataset
    dataset_label = datasets.ImageFolder(path_label, transform)
    index_label = dataset_label.class_to_idx[subfolder]
    n_label = 0
    m_label = 0
    for i in range(dataset_label.__len__()):
        if index_label != dataset_label.imgs[n_label][1]:
            del dataset_label.imgs[n_label]
            n_label = n_label - 1
        else:
            if m_label % 3 != 0:
                del dataset_label.imgs[n_label]
                n_label = n_label - 1
            m_label = m_label + 1
        n_label = n_label + 1
    len_dataset_label = dataset_label.__len__()

    # create the raw image dataset
    dataset_raw = datasets.ImageFolder(path_raw, transform)
    index_raw = dataset_raw.class_to_idx[subfolder]
    n_raw = 0
    for i in range(dataset_raw.__len__()):
        if index_raw != dataset_raw.imgs[n_raw][1]:
            del dataset_raw.imgs[n_raw]
            n_raw = n_raw - 1
        n_raw = n_raw + 1
    len_dataset_raw = dataset_raw.__len__()

    # combine datasets
    dataset_combined = []
    if(len_dataset_label != len_dataset_raw):
        print("The length of two training sets are not equal! Please check!")
    else:
        for i in range(len_dataset_label):
            #print(dataset_label[i][0].size())
            dataset_combined.append(torch.cat((dataset_label[i][0], dataset_raw[i][0]), 2))
            #plt.imshow(dataset_combined[i][0].numpy().transpose((1, 2, 0)))
            #plt.show()
            #print(dataset_combined[i][0].size())


    Dataloader = torch.utils.data.DataLoader(dataset_combined, batch_size=batch_size, shuffle=shuffle, num_workers=16, pin_memory=True)


    return (len_dataset_label, len_dataset_raw, Dataloader)