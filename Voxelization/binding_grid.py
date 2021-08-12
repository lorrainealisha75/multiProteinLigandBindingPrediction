#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Script to generate the binding grid and filter out
unnecessary grid points
"""

import numpy as np
import scipy.spatial as sp

import time

import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
from sklearn.cluster import DBSCAN


# Generate sphere grid points
def sGrid(center, r, N):
    center = np.array(center)
    x = np.linspace(center[0]-r,center[0]+r,N)
    y = np.linspace(center[1]-r,center[1]+r,N)
    z = np.linspace(center[2]-r,center[2]+r,N)
    #Generate grid of points
    X,Y,Z = np.meshgrid(x,y,z)
    data = np.vstack((X.ravel(),Y.ravel(),Z.ravel())).T
    # indexing the interior points
    tree = sp.cKDTree(data)
    mask = tree.query_ball_point(center,1.01*r)
    points_in_sphere = data[mask]
    return points_in_sphere


# test if a point is inside a convex hull
def in_hull(p, hull):
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0


# binding site refinement
def site_refine(site,protein_coords):
    # distance matrix for the removal of the grid points that are too close (<= 2 A) to any protein atoms
    dist = cdist(site[:,0:3], protein_coords, 'euclidean')
    inside_site = []
    for i in range(len(dist)):
        if np.any(dist[i,:] < 1.1):
            continue
        else:
            inside_site.append(site[i,:])
    inside_site = np.array(inside_site)
    # remove any grid points outside the convex hull
    in_bool = in_hull(inside_site[:,0:3], protein_coords)
    hull_site = inside_site[in_bool]
    # remove isolated grid points
    iso_dist = cdist(hull_site[:,0:3],hull_site[:,0:3])
    labels = DBSCAN(eps = 1.414, min_samples = 3, metric = 'precomputed').fit_predict(iso_dist)
    unique, count = np.unique(labels, return_counts = True)
    sorted_label = [x for _,x in sorted(zip(count,unique))]
    sorted_label = np.array(sorted_label)
    null_index = np.argwhere(sorted_label == -1)
    cluster_labels = np.delete(sorted_label, null_index)
    save_labels = np.flip(cluster_labels, axis = 0)[0]
    final_label = np.zeros(labels.shape)
    for k in range(len(labels)):
        if labels[k] == save_labels:
            final_label[k] = 1
        else:
            continue
    final_label = np.array(final_label, dtype=bool)
    # potential energy normalization
    iso_site = hull_site[final_label]
    return iso_site


# Calculate nearest amino acid for each grid point in the refined binding site
def assign_amino_acid(site, selected_coords):
    amino_acid = selected_coords[['residue_number', 'residue_name']].drop_duplicates()
    amino_acid_list = []
    ss = time.time()
    for coordinate in site:
        distances = np.linalg.norm(selected_coords[['x_coord', 'y_coord', 'z_coord']] - coordinate, axis=1)

        # Distance of a coordinate in the binding site to all the residues in the auxiliary file
        selected_coords['distance_to_binding_residues'] = distances

        # Group by residue number and get minimum distance
        min_dist = selected_coords.groupby(['residue_number'])['distance_to_binding_residues'].min()

        # Get the 3-letter amino acid id corresponding to the residue with the least distance
        amino_acid_list.append(amino_acid.loc[amino_acid['residue_number'] == min_dist.idxmin()].iat[0, 1])

    print('\nThe total time of computation is: ' + str(time.time() - ss) + ' seconds')
    return amino_acid_list


# main function
class Grid3DBuilder(object):
    """ Given an align protein, generate the binding grid
    and calculate the amino acids nearest to each coordinate in the binding site """
    @staticmethod
    def build(protein_coords, selected_coords, r, N):
        """
        Input: protein coordinates, selected protein coordinates from the auxiliary file, radius, number of points
        along the radius.
        Output: dataframe of the binding grid, including coordinates and amino acid ids.
        """
        print('The radius of the binding grid is: ' + str(r))
        print('\nThe number of points along the diameter is: ' + str(N))
        binding_site = sGrid(np.array([0,0,0]),r,N)
        new_site = site_refine(binding_site, protein_coords)
        amino_acid_list = assign_amino_acid(new_site, selected_coords)
        binding_site = pd.DataFrame(data=new_site, columns=['x', 'y', 'z'])
        binding_site['nearest_amino_acid'] = amino_acid_list
        print('\nThe number of points in the refined binding set is ' + str(len(new_site)))

        return binding_site[['x', 'y', 'z', 'nearest_amino_acid']]

