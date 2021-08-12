#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script processes the PDB file to extract the binding center coordinates
and the residue IDs within the binding radius
"""

from biopandas.pdb import PandasPdb
import json
import argparse


class AuxFileBuilder(object):

    def __init__(self, pdb, protein, ligand):
        self.pl_complex = pdb
        self.protein = protein
        self.ligand = ligand

        #Binding radius, adjustable hyperparameter
        self.radius = 7


    def combine(self):
        x, y, z = self.get_binding_site_centre()
        distances = self.get_distances((x, y, z))
        binding_residue_ids = self.get_binding_residue_ids(distances)
        ligand_ids = self.get_ligand_residue_id()
        all_ids = binding_residue_ids + ligand_ids
        binding_site_coords = {}
        binding_site_coords['x'] = x
        binding_site_coords['y'] = y
        binding_site_coords['z'] = z
        return (binding_site_coords, all_ids)


    def get_binding_site_centre(self):
        hetatm = self.pl_complex.df['HETATM']
        ligand = hetatm.loc[hetatm['residue_name'] == self.ligand]
        x = ligand['x_coord']
        y = ligand['y_coord']
        z = ligand['z_coord']
        return round(x.mean(), 2), round(y.mean(), 2), round(z.mean(), 2)


    def get_distances(self, binding_site_coord):
        return self.pl_complex.distance(xyz=binding_site_coord, records=('ATOM'))


    def get_binding_residue_ids(self, distances):
        atoms = self.pl_complex.df['ATOM'][distances < self.radius]
        atom_list = atoms.loc[atoms['element_symbol'] != 'H']['residue_number']
        return atom_list.drop_duplicates().tolist()


    def get_ligand_residue_id(self):
        hetatm = self.pl_complex.df['HETATM']
        ligand = hetatm.loc[hetatm['residue_name'] == self.ligand]
        residue_id = ligand['residue_number']
        return residue_id.drop_duplicates().tolist()

