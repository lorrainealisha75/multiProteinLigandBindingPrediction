#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script extracts the ligand name from the PDB file
linked to the assay in the PCBA dataset.
"""
from biopandas.pdb import PandasPdb
import json
import requests
import random
import argparse

def get_ligand_names(pdb_file_path):
    try:
        protein = PandasPdb().read_pdb(pdb_file_path)
        hetatm = protein.df['HETATM']
        #Remove HEM as it is a cofactor and not a ligand
        hetatm = hetatm.loc[hetatm['residue_name'] != 'HEM']
        residue_names = hetatm['residue_name'].drop_duplicates().to_list()
        fila = []
        for ligand in residue_names:
            result = hetatm.loc[hetatm['residue_name'] == ligand]['element_symbol']
            if result.isin(['C']).any(): 
                fila.append(ligand)     
        return fila
    except IOError:
        print(pdb_file_path+' not found!') 

def main(args):
    info = {}
    used_proteins = []
    with open(args.appath) as json_file:
        data = json.load(json_file)
        for entry in data:
            info[entry['AID']] = {}
            for protein in entry['pdb_ids']:
                if entry['pdb_ids'][0] != None and protein not in used_proteins:
                    used_proteins.append(protein)

                    print(entry['AID'], protein)
                    print('https://files.rcsb.org/download/'+protein+'.pdb')

                    pdb_file = requests.get('https://files.rcsb.org/download/'+protein+'.pdb', allow_redirects=True)
                    pdb_file_path = 'pdb_files/'+protein+'.pdb'
                    open(pdb_file_path, 'w+').write(pdb_file.text)
                    residues = get_ligand_names(pdb_file_path)
                    if len(residues) > 0:
                        residue = random.choice(residues)
                        info[entry['AID']][protein] = residue
                        print(residue)
                        break

    with open('pdbid_ligandname_mapping.txt', 'w+') as outfile:
        json.dump(info, outfile, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--appath', type=str, required=True, help='Path to the file with the mapping between assay ids and protein pdb ids')

    args = parser.parse_args()
    main(args)              
