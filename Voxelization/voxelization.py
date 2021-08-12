#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script converts the pdb file to the voxel representation.
"""
from __future__ import division

import json
import argparse
import time
import ntpath
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm

from biopandas.pdb import PandasPdb

from sklearn.preprocessing import normalize

from binding_grid import Grid3DBuilder
from generate_aux_files import AuxFileBuilder


def site_voxelization(site, voxel_length):
    amino_acid_dict = {
        "ALA": 1,
        "ARG": 2,
        "ASN": 3,
        "ASP": 4,
        "CYS": 5,
        "GLN": 6,
        "GLU": 7,
        "GLY": 8,
        "HIS": 9,
        "ILE": 10,
        "LEU": 11,
        "LYS": 12,
        "MET": 13,
        "PHE": 14,
        "PRO": 15,
        "SER": 16,
        "THR": 17,
        "TRP": 18,
        "TYR": 19,
        "VAL": 20
    }

    coords = np.array(site.iloc[:, 0:3], dtype=np.float64)
    amino_acid = site.iloc[:, 3:None]['nearest_amino_acid'].to_list()
    #print(amino_acid)
    voxel_length = 32
    voxel_start = int(-voxel_length / 2 + 1)
    voxel_end = int(voxel_length / 2)
    voxel = np.zeros(shape=(voxel_length, voxel_length, voxel_length),
                     dtype=np.int64)
    ss = time.time()
    for x in range(voxel_start, voxel_end + 1, 1):
        for y in range(voxel_start, voxel_end + 1, 1):
            for z in range(voxel_start, voxel_end + 1, 1):
                temp_voxloc = [x, y, z]
                distances = np.linalg.norm(coords - temp_voxloc, axis=1)
                min_dist = np.min(distances)
                index = np.where(distances == min_dist)
                if min_dist < 0.01:
                    #print(amino_acid_dict[amino_acid[index[0][0]]])
                    voxel[x - voxel_start, y - voxel_start, z - voxel_start] = amino_acid_dict[amino_acid[index[0][0]]]

    print('\nThe total time for voxelization is: ' + str(time.time() - ss) + ' seconds')
    return voxel

def convert_to_vhse8(voxel, vhse8_norm):
    
    vector_dict = {
            1: vhse8_norm[0],
            2: vhse8_norm[1],
            3: vhse8_norm[2],
            4: vhse8_norm[3],
            5: vhse8_norm[4],
            6: vhse8_norm[5],
            7: vhse8_norm[6],
            8: vhse8_norm[7],
            9: vhse8_norm[8],
            10: vhse8_norm[9],
            11: vhse8_norm[10],
            12: vhse8_norm[11],
            13: vhse8_norm[12],
            14: vhse8_norm[13],
            15: vhse8_norm[14],
            16: vhse8_norm[15],
            17: vhse8_norm[16],
            18: vhse8_norm[17],
            19: vhse8_norm[18],
            20: vhse8_norm[19],
        }

    voxel = np.squeeze(voxel)
    vhse_array = np.zeros((32, 32, 32, 8))
    I, J, K = np.nonzero(voxel)
    indices = np.argwhere(voxel)

    for count, index in enumerate(indices):
        vhse_array[I[count]][J[count]][K[count]] = vector_dict[voxel[I[count]][J[count]][K[count]]]

    return np.transpose(vhse_array, (3, 0, 1, 2))


def convert_to_ohe(voxel, ohe):

    vector_dict = {
            1: ohe[0],
            2: ohe[1],
            3: ohe[2],
            4: ohe[3],
            5: ohe[4],
            6: ohe[5],
            7: ohe[6],
            8: ohe[7],
            9: ohe[8],
            10: ohe[9],
            11: ohe[10],
            12: ohe[11],
            13: ohe[12],
            14: ohe[13],
            15: ohe[14],
            16: ohe[15],
            17: ohe[16],
            18: ohe[17],
            19: ohe[18],
            20: ohe[19],
        }

    voxel = np.squeeze(voxel)
    ohe_array = np.zeros((32, 32, 32, 20))
    I, J, K = np.nonzero(voxel)
    indices = np.argwhere(voxel)

    for count, index in enumerate(indices):
        ohe_array[I[count]][J[count]][K[count]] = vector_dict[voxel[I[count]][J[count]][K[count]]]

    return np.transpose(ohe_array, (3, 0, 1, 2))


def visualize_voxel(voxel):
    voxel = voxel[0]
    cmap = cm.get_cmap('tab20', 21)
    fig, axs = plt.subplots(2, 4, figsize=(6, 3)) #, tight_layout=True)
    for sl in range(voxel.shape[0]):
        if sl > 7 and sl <= 11:
            psm = axs[0, sl-8].pcolormesh(voxel[sl], cmap=cmap, rasterized=True, vmin=0, vmax=20)
            axs[0, sl-8].set_aspect('equal')
            axs[0, sl-8].set_xticks([])
            axs[0, sl-8].set_yticks([])
        elif sl > 11 and sl <= 15:
            psm = axs[1, sl-12].pcolormesh(voxel[sl], cmap=cmap, rasterized=True, vmin=0, vmax=20)
            axs[1, sl-12].set_aspect('equal')
            axs[1, sl-12].set_xticks([])
            axs[1, sl-12].set_yticks([])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(psm, cax=cbar_ax)

    plt.savefig('899_6_3.png')


def normalize(v):
    """ vector normalization """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def vrrotvec(a, b):
    """ Function to rotate one vector to another, inspired by
    vrrotvec.m in MATLAB """
    a = normalize(a)
    b = normalize(b)
    ax = normalize(np.cross(a, b))
    angle = np.arccos(np.minimum(np.dot(a, b), [1]))
    if not np.any(ax):
        absa = np.abs(a)
        mind = np.argmin(absa)
        c = np.zeros((1, 3))
        c[mind] = 0
        ax = normalize(np.cross(a, c))
    r = np.concatenate((ax, angle))
    return r


def vrrotvec2mat(r):
    """ Convert the axis-angle representation to the matrix representation of the
    rotation """
    s = np.sin(r[3])
    c = np.cos(r[3])
    t = 1 - c

    n = normalize(r[0:3])

    x = n[0]
    y = n[1]
    z = n[2]

    m = np.array(
        [[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
         [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
         [t * x * z - s * y, t * y * z + s * x, t * z * z + c]]
    )
    return m


# Create a DF of the transformed protein coordinates based on the residue n

def select_protein_coords(path_to_transformed_pdb, residue_ids):
    ppdb = PandasPdb().read_pdb(path_to_transformed_pdb)
    protein_all_atoms_df = ppdb.df['ATOM']  # dataframe with list of protein atoms
    # we want to exclude the main chain atoms from the amino acid
    protein_df = protein_all_atoms_df.loc[(protein_all_atoms_df['atom_name'] != 'H') &
                                          (protein_all_atoms_df['atom_name'] != 'N') &
                                          (protein_all_atoms_df['atom_name'] != 'CA') &
                                          (protein_all_atoms_df['atom_name'] != 'C') &
                                          (protein_all_atoms_df['atom_name'] != 'O')]

    # Select the amino acid molecules whose residue ids are present in the auxiliary file
    selected_protein_df = protein_df[protein_df['residue_number'].isin(residue_ids)]
    return selected_protein_df


    

def clean_pdb(pdb_path):
    # Read the pdb file
    ppdb = PandasPdb().read_pdb(pdb_path)
    
    #Remove Hydrogen molecules and nucleotides like A, C, G, T
    rows_to_be_deleted = ppdb.df['ATOM'][ppdb.df['ATOM']['element_symbol'] == 'H']
    rows_to_be_deleted = rows_to_be_deleted.append(ppdb.df['ATOM'][ppdb.df['ATOM']['residue_name'] == 'DA'])
    rows_to_be_deleted = rows_to_be_deleted.append(ppdb.df['ATOM'][ppdb.df['ATOM']['residue_name'] == 'DC'])
    rows_to_be_deleted = rows_to_be_deleted.append(ppdb.df['ATOM'][ppdb.df['ATOM']['residue_name'] == 'DG'])
    rows_to_be_deleted = rows_to_be_deleted.append(ppdb.df['ATOM'][ppdb.df['ATOM']['residue_name'] == 'DT'])

    ppdb.df['ATOM'].drop(rows_to_be_deleted.index, inplace=True)
    return ppdb


class Vox3DBuilder(object):
    """
    This class convert the pdb file to the voxel representation for the input
    of deep learning architecture. The conversion is around 30 mins for each binding site.
    """

    def __init__(self):
        self.radius = 15
        self.number = 31
        vhse8 = np.array([
                      [0.15, -1.11, -1.35, -0.92, 0.02, -0.91, 0.36, -0.48],
                      [-1.47, 1.45, 1.24, 1.27, 1.55, 1.47, 1.30, 0.83],
                      [-0.99, 0.00, -0.37, 0.69, -0.55, 0.85, 0.73, -0.80],
                      [-1.15, 0.67, -0.41, -0.01, -2.68, 1.31, 0.03, 0.56],
                      [0.18, -1.67, -0.46, -0.21, 0.00, 1.20, -1.61, -0.19],
                      [-0.96, 0.12, 0.18, 0.16, 0.09, 0.42, -0.20, -0.41],
                      [-1.18, 0.40, 0.10, 0.36, -2.16, -0.17, 0.91, 0.02],
                      [-0.20, -1.53, -2.63, 2.28, -0.53, -1.18, 2.01, -1.34],
                      [-0.43, -0.25, 0.37, 0.19, 0.51, 1.28, 0.93, 0.65],
                      [1.27, -0.14, 0.30, -1.80, 0.30, -1.61, -0.16, -0.13],
                      [1.36, 0.07, 0.26, -0.80, 0.22, -1.37, 0.08, -0.62],
                      [-1.17, 0.70, 0.70, 0.80, 1.64, 0.67, 1.63, 0.13],
                      [1.01, -0.53, 0.43, 0.00, 0.23, 0.10, -0.86, -0.68],
                      [1.52, 0.61, 0.96, -0.16, 0.25, 0.28, -1.33, -0.20],
                      [0.22, -0.17, -0.50, 0.05, -0.01, -1.34, -0.19, 3.56],
                      [-0.67, -0.86, -1.07, -0.41, -0.32, 0.27, -0.64, 0.11],
                      [-0.34, -0.51, -0.55, -1.06, -0.06, -0.01, -0.79, 0.39],
                      [1.50, 2.06, 1.79, 0.75, 0.75, -0.13, -1.01, -0.85],
                      [0.61, 1.60, 1.17, 0.73, 0.53, 0.25, -0.96, -0.52],
                      [0.76, -0.92, -0.17, -1.91, 0.22, -1.40, -0.24, -0.03]
        ])

        self.ohe = np.array([
                      [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        ])

        self.vhse8_norm = normalize(vhse8)

    def voxelization(self, args, ppdb, pcba_id, protein, ligand, binding_site_coords, binding_site_ids): 
        protein_df = ppdb.df['ATOM']
        content = []
        content.append(binding_site_ids)
        content.append(binding_site_coords)

        residue_ids = content[0]
        if len(content[1]) != 0:
            pocket_df = protein_df[protein_df['residue_number'].isin(residue_ids)]
            pocket_coords = np.array([pocket_df['x_coord'], pocket_df['y_coord'], pocket_df['z_coord']]).T
            pocket_center = list([content[1]["x"], content[1]["y"], content[1]["z"]])
        else:
            print('No center is provided')
            pocket_df = protein_df[protein_df['residue_number'].isin(residue_ids)]
            pocket_coords = np.array([pocket_df['x_coord'], pocket_df['y_coord'], pocket_df['z_coord']]).T
            pocket_center = np.mean(pocket_coords, axis=0)

        protein_coords = np.array([protein_df['x_coord'], protein_df['y_coord'], protein_df['z_coord']]).T
        pocket_coords = pocket_coords - pocket_center  # center the pocket to 0,0,0
        protein_coords = protein_coords - pocket_center  # center the protein according to the pocket center
        inertia = np.cov(pocket_coords.T)
        e_values, e_vectors = np.linalg.eig(inertia)
        sorted_index = np.argsort(e_values)[::-1]
        sorted_vectors = e_vectors[:, sorted_index]
        # Align the first principal axes to the X-axes
        rx = vrrotvec(np.array([1, 0, 0]), sorted_vectors[:, 0])
        mx = vrrotvec2mat(rx)
        pa1 = np.matmul(mx.T, sorted_vectors)
        # Align the second principal axes to the Y-axes
        ry = vrrotvec(np.array([0, 1, 0]), pa1[:, 1])
        my = vrrotvec2mat(ry)
        transformation_matrix = np.matmul(my.T, mx.T)
        # transform the protein coordinates to the center of the pocket and align with the principal
        # axes with the pocket
        transformed_coords = (np.matmul(transformation_matrix, protein_coords.T)).T
        # Generate a new pdb file with transformed coordinates
        ppdb.df['ATOM']['x_coord'] = transformed_coords[:, 0]
        ppdb.df['ATOM']['y_coord'] = transformed_coords[:, 1]
        ppdb.df['ATOM']['z_coord'] = transformed_coords[:, 2]
        output_trans_pdb_path = 'temp_files/'+ligand + '_trans.pdb'

        print('\nOutput the binding pocket aligned pdb file to: ' + output_trans_pdb_path)
        ppdb.to_pdb(output_trans_pdb_path)

        selected_coords = select_protein_coords(output_trans_pdb_path, residue_ids)

        # Grid generation and DFIRE potential calculation
        print('\n...Generating pocket grid representation\n')
        pocket_grid = Grid3DBuilder.build(transformed_coords, selected_coords, self.radius, self.number)

        print('\n...Generating pocket voxel representation\n')
        pocket_voxel = site_voxelization(pocket_grid, self.number + 1)
        pocket_voxel = np.expand_dims(pocket_voxel, axis=0)
        visualize_voxel(pocket_voxel)
        pocket_voxel = convert_to_vhse8(pocket_voxel, self.vhse8_norm)
        #pocket_voxel = convert_to_ohe(pocket_voxel, self.ohe)
        
        print('\nSaving to '+args.vpath+'voxel_'+pcba_id+'_'+protein+'_'+ligand)
        np.save(args.vpath+'voxel_'+pcba_id+'_'+protein+'_'+ligand, pocket_voxel)
        print('\n-----------------------------------------------------------------------------\n')


def main(args):

    with open(args.plpath) as json_file:
        data = json.load(json_file)
        for protein in data:
            try:
                for structure in data[protein].keys():        
                    pdb_path = args.pdbpath+structure+'.pdb'
                    ppdb = clean_pdb(pdb_path)
                    aux_file_builder = AuxFileBuilder(ppdb, structure, data[protein][structure])
                    binding_site_coords, binding_site_ids = aux_file_builder.combine()
                    voxel = Vox3DBuilder()
                    voxel.voxelization(args, ppdb, protein, structure, data[protein][structure], binding_site_coords, binding_site_ids)
            
            except:
                print('\nCould not voxelize ', structure)
                print('--------------------------------------------------------------\n')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vpath', type=str, required=True, help='Path to the directory with the voxelized proteins')
    parser.add_argument('--pdbpath', type=str, required=True, help='Path to the directory containing PDB files linked to PCBA dataset assays')
    parser.add_argument('--plpath', type=int, required=True, help='Path to the file that contains mapping of the PDB file to the ligand name')

    args = parser.parse_args()
    main(args)

