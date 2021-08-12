#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script combines the protein descriptors (compressed representation
of the protein binding site with the ligand descriptors generated from
RDKit and assigns the binding label to create the final dataset that will
be used to train the LGBM classifier) 
"""
import pandas as pd
import numpy as np
import argparse
import random
import fnmatch
import json
import csv
import gc
import os

from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit import Chem


def find_voxelized_protein_file(protein, cvpath):
    for file in os.listdir(cvpath):
        pattern = 'voxel_'+protein+'_*.npy'
        if fnmatch.fnmatch(file, pattern):
            return file
    return None

def load_protein_voxel(path):
    x = np.load(path)[0]
    return x

def main(args):

    #Read in only column names
    col_names = pd.read_csv(args.dpath, sep=',',
                            header=0, nrows=0).columns

    #Setting a dictionary for column data types
    types_dict = {'mol_id': str, 'smiles': str}
    types_dict.update({col: float for col in col_names if col not in types_dict})

    #Read in entire csv file
    pcba_dataset = pd.read_csv(args.dpath, dtype=types_dict,
                               sep=',', header=0, error_bad_lines=False)

    protein_list = []
    with open(args.plpath) as json_file:
            data = json.load(json_file)
            for protein in data:
                for structure in data[protein].keys():    
                    protein_list.append('PCBA_'+protein)

    only_proteins = pcba_dataset.filter(protein_list)

    print('Number of columns is: ', len(only_proteins.columns))

    #Open JSON file which have relevant proteins
    with open(args.plpath, 'r') as f:
        protein_info = json.load(f)

    list_pos = pd.read_csv(args.plist, sep=',', header=None, error_bad_lines=False)
    list_pos = list(list_pos.iloc[0,:])

    only_proteins = (only_proteins).dropna(how='all')

    descriptor_list = [desc[0] for desc in Chem.Descriptors.descList]
    descriptor_list = descriptor_list[0:200]


    calculator = MolecularDescriptorCalculator(descriptor_list)
    descriptor_list.extend(['protein_voxel', 'label'])

    with open('final_dataset.csv', 'w+') as file_obj:
        writer = csv.writer(file_obj, delimiter=',')

        for col in only_proteins:
            colm = col[5:]
            
            if colm in protein_info and bool(protein_info[colm]):
                print(colm)
                voxel_name = find_voxelized_protein_file(colm, args.cvpath)
                
                if voxel_name == None:
                    print(colm+' not available.')
                    continue

                voxel = load_protein_voxel(os.path.join(args.cvpath, voxel_name))
                voxel = voxel.flatten() 
            
                #Get all ligands with positive and negative interaction
                positives = pcba_dataset.loc[pcba_dataset[col] == 1.0]['smiles'].to_list()
                negatives = pcba_dataset.loc[pcba_dataset[col] == 0.0]['smiles'].to_list()
                
                #Randomly select an equal number of negative interaction ligands as that of positives
                #to maintain a balanced dataset 
                if len(positives)<len(negatives):
                    if len(positives)<1500:              
                        negatives = random.sample(negatives, len(positives))
                    else:
                        length = random.sample(list_pos, 1)[0]
                        positives = random.sample(positives, length)
                        negatives = random.sample(negatives, length)
                else:
                    if len(negatives)<1500:
                        positives = random.sample(positives, len(negatives))
                    else:
                        length = random.sample(list_pos, 1)[0]
                        positives = random.sample(positives, length)
                        negatives = random.sample(negatives, length)
               
                for pos in positives:
                    mol = Chem.MolFromSmiles(pos)
                    if mol != None:
                        row = list(calculator.CalcDescriptors(mol))
                        row.extend(voxel)
                        row.append(1)
                        writer.writerow(row)

                for neg in negatives:

                    mol = Chem.MolFromSmiles(neg)
                    if mol != None:
                        row = list(calculator.CalcDescriptors(mol))
                        row.extend(voxel)
                        row.append(0)
                        writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cvpath', type=str, required=True, help='Path to the directory with compressed representations of the voxelized proteins')
    parser.add_argument('--dpath', type=str, required=True, help='Path to the PCBA dataset')
    parser.add_argument('--plpath', type=int, required=True, help='Path to the file that contains mapping of the PDB file to the ligand name')
    parser.add_argument('--plist', type=float, required=True, help='Path to the file that contains length of positive samples for each assay')

    args = parser.parse_args()
    main(args)