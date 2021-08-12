"""
This script describes the architecture of the convolution autoencoder
that will process the voxelized grids.
"""

import torch
import torch.nn as nn

class ProteinLigandAffinities(nn.Module):

    def __init__(self, in_channel):
        super(ProteinLigandAffinities, self).__init__()

        #Model for convolution over voxelized protein
        self.conv1 = nn.Conv3d(in_channel, 6, 5)
        self.conv2 = nn.Conv3d(6, 4, 3)
        self.conv3 = nn.Conv3d(4, 2, 3)
        self.pool = nn.MaxPool3d((2,2,2), stride=None)

        #Model for transpose convolution over the ligand descriptors
        self.tconv1 = nn.ConvTranspose3d(2, 4, 5, stride=2)
        self.tconv2 = nn.ConvTranspose3d(4, 6, 5, stride=2)
        self.tconv3 = nn.ConvTranspose3d(6, 8, 4)

        #relu activations
        self.relu = nn.ReLU()

        #Batchnorm
        self.batch_norm_1 = nn.BatchNorm3d(6)
        self.batch_norm_2 = nn.BatchNorm3d(4)
        self.batch_norm_3 = nn.BatchNorm3d(2)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

        self.tconv1.reset_parameters()
        self.tconv2.reset_parameters()
        self.tconv3.reset_parameters()

    def forward(self, voxel, need_compressed=False):

        voxel = self.conv1(voxel)
        voxel = self.batch_norm_1(voxel)
        voxel = self.relu(voxel)

        voxel = self.conv2(voxel)
        voxel = self.batch_norm_2(voxel)
        voxel = self.relu(voxel)

        voxel = self.pool(voxel)

        voxel = self.conv3(voxel)
        voxel = self.batch_norm_3(voxel)
        voxel = nn.LeakyReLU(negative_slope=0.1)(voxel)

        voxel = self.pool(voxel)

        compressed = nn.Dropout(p=0.4)(voxel)

        if need_compressed:
          voxel = torch.flatten(compressed, start_dim=1)
          return voxel

        else:
          voxel = self.tconv1(compressed)
          voxel = nn.LeakyReLU(negative_slope=0.1)(voxel)

          voxel = self.tconv2(voxel)
          voxel = nn.LeakyReLU(negative_slope=0.1)(voxel)

          voxel = self.tconv3(voxel)
          voxel = nn.LeakyReLU(negative_slope=0.1)(voxel)

          return voxel
