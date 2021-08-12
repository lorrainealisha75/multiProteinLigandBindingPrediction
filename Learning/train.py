#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script uses the voxelized grids from the PDBBind dataset
to train the convolution autoencoder and then tests on the PCBA
dataset voxlized grids to output compressed representation of
the binding pocket.
"""

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from model_32 import ProteinLigandAffinities
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
import pandas as pd
import sys
import argparse
import random
import os

seed = 1234
random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def load_datapoints(path):
    datapoint = np.load(path)
    return datapoint

def plot_train_val_curves(train_losses, val_losses):
    plt.plot(np.array(train_losses), label = "Train loss")
    plt.plot(np.array(val_losses), label = "Validation loss")
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.gcf().subplots_adjust(left=0.15)
    plt.legend()
    plt.show()

    plt.savefig('train_val_curves', dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            metadata=None)

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

def main(args):

    dataset = torchvision.datasets.DatasetFolder(
                  root=args.vpath,
                  loader=load_datapoints,
                  extensions='.npy'
    )
    y = d = pd.Series(0, index=np.arange(len(dataset)))
    model = ProteinLigandAffinities(in_channel=8)
    print(get_device())
    model.to(get_device())
    model = model.double()

    criterion = nn.MSELoss()

    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skfold.split(dataset, y)):
        train_set = Subset(dataset, train_idx)
        train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, num_workers=5, drop_last=True)
        val_set = Subset(dataset, val_idx)
        val_loader = DataLoader(val_set, batch_size=args.batch_size,
                                shuffle=True, num_workers=5, drop_last=True)

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader), eta_min=args.lr)

        train_losses = np.zeros((args.nepochs,))
        train_accuracies = np.zeros((args.nepochs,))
        val_losses = np.zeros((args.nepochs,))
        val_accuracies = np.zeros((args.nepochs,))

        print('\nFold {}'.format(fold+1))
        print('--------------------------------------\n')

        for epoch in range(args.nepochs):

            train_loss = 0.0
            train_accuracy = 0.0
            model.train()

            print("Epoch {} / {}:".format(epoch + 1, args.nepochs))
            for tbatch, data in enumerate(train_loader, 0):
                voxel = data[0]
                #print('Voxel size: ', voxel.size())
                voxel = voxel.to(get_device())

                optimizer.zero_grad()
                output = model(voxel, False)
                #print('Output shape: ', output.shape)
                loss = criterion(output, voxel)
                train_loss += loss.data
                loss.backward()
                optimizer.step()

            train_losses[epoch] = train_loss/(tbatch+1)
            print('\nTraining loss for Epoch {} is {:.6f}\n'.format(epoch+1, train_losses[epoch]))


            model.eval()
            val_loss = 0
            val_accuracy = 0

            for vbatch, data in enumerate(val_loader):
                voxel = data[0]
                voxel = voxel.to(get_device())

                with torch.no_grad():
                    output = model(voxel, False)
                    loss = criterion(output, voxel)

                val_loss += loss.data.cpu().numpy()

            val_losses[epoch] = val_loss/(vbatch+1)
            print('Validation loss for Epoch {} is {:.6f}\n'.format(epoch+1, val_losses[epoch]))


            scheduler.step()
        break 

    plot_train_val_curves(train_losses, val_losses)


    print('Finished Training')

    #Test the auto encoder

    test_dataset = torchvision.datasets.DatasetFolder(
                  root=args.tpath,
                  loader=load_datapoints,
                  extensions='.npy'
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1
    )

    for i, data in enumerate(test_loader, 0):
      voxel = data[0]
      voxel_name = test_loader.dataset.samples[i][0]
      voxel_name = os.path.basename(voxel_name)
      voxel = voxel.to(get_device())
      output = model(voxel, True)
      np.save(args.cvpath+voxel_name, output.detach().cpu().numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vpath', type=str, required=True, help='Path to the directory with voxelized proteins')
    parser.add_argument('--tpath', type=str, required=True, help='Path to the test dataset')
    parser.add_argument('--cvpath', type=str, required=True, help='Path to the directory with compressed representations of the voxelized proteins')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for training')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate for training')
    parser.add_argument('--nepochs', type=int, required=True, help='No:of epochs for training')

    args = parser.parse_args()
    main(args)
