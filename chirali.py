#!/usr/bin/env python3

import argparse
import copy
import json
import os
import pickle
import shutil
import sys

import numpy as np
from scipy import sparse
from tqdm import tqdm

import ase
import ase.io
import ase.build.tools
import ase.neighborlist as nl

def main():
    """
    Main function for creating the structure
    """
    # Parse arguments
    args = parse_command_line_arguments()
    structure = ChiralipyStructure(args.settings)

class ChiralipyStructure():
    """
    Class to help with the analysis of molecular dynamics trajectories for the analysis of chiral structural features
    """
    def __init__(self, settings):
        """
        Initialize a ChiralipyStructure instance.

        Parameters:
        - settings_path (str): The path to the settings JSON file.
        """
        self.__load_settings(settings) # read settings file

        print('Loading system; analysis will be done in the {:}-mode...'.format(self.mode))
        self.__load_model_system(self.mode) # read the model system

        print('Initializing system...')
        self.__initialize_descriptors() # initialize required data arrays for determination of variety of quantitites
        
        print('Evaluating descriptors...')
        self.__evaluate_descriptors(self.mode) # analyze the model system with the desired descriptors

    def __load_settings(self, file_path):
        """
        Load settings from the specified JSON file and attach them as attributes to the instance.

        Parameters:
        - file_path (str): The path to the JSON file.

        Raises:
        - FileNotFoundError: If the specified file is not found.
        - json.JSONDecodeError: If there is an issue decoding the JSON file.
        """
        try:
            with open(file_path, 'r') as json_file:
                settings = json.load(json_file)

                if isinstance(settings, dict): # ensure that the loaded data is a dictionary
                    self.__attach_settings_to_class(settings)
                else:
                    print(f"Error: Invalid JSON format in file {file_path}")
                    return None

        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file {file_path}: {e}")
            return None

    def __attach_settings_to_class(self, settings):
        """
        Attach settings from a dictionary as attributes to the class.

        Parameters:
        - settings (dict): The settings dictionary.
        """
        # Basic analysis settings
        self.structure_path = settings['structure']  # path to initial structure
        self.center_elements = settings['center_elements']  # elements at center of octahedra
        self.corner_elements = settings['corner_elements']  # elements at corners of octahedra
        self.cation_elements = settings['cation_elements']  # elements at headgroup of cations
        self.organic_elements = settings['organic_elements']  # elements found in organic molecules
        self.organic_elements = settings['organic_elements'] # elements found in organic molecules
        self.layer_tolerance = settings.get('layer_tolerance', 0.5) # tolerance used to determine layers in structures
        try:
            self.direction = {'x': 0, 'y': 1, 'z': 2}[settings['layer_directions']]  # direction of planar stacking
        except KeyError:
            # Handle the error by providing a default direction or taking appropriate action
            print("Warning: Invalid or missing 'layer_directions'. Using default direction 'z'.")
            self.direction = 2  # Set a default direction (e.g., 'z')

        # Set the calculation mode
        self.mode = settings.get('mode', 'structure') # determine the mode of analysis
        if self.mode == 'trajectory':
            self.trajectory_path = settings['trajectory'] # path to trajectory
            self.trajectory_format = settings.get('file_format', 'vasp-xdatcar') # file format of trajectory
            self.n_skip = settings.get('n_skip', 0)  # number of frames to skip for equilibration
            self.average = settings.get('average', False)  # boolean switch for analyzing time-averaged structure
            self.output_average = settings.get('output_average', 'average_structure.vasp')  # file to output the average structure to

        # Boolean switches of structural descriptors to analyze
        self.bool_intraoctahedral_distortions = settings.get('intraoctahedral_distortions', False)
        self.bool_octahedral_geometry = settings.get('octahedral_geometry', False)
        self.bool_interoctahedral_distortions = settings.get('interoctahedral_distortions', False)
        self.bool_planar_distortions = settings.get('planar_distortions', False)
        self.bool_hydrogen_bond_asymmetry = settings.get('hydrogen_bond_asymmetry', False)
        self.bool_planar_framework_helicity = settings.get('planar_framework_helicity', False)
        self.bool_out_of_plane_framework_helicity = settings.get('out_of_plane_framework_helicity', False)

        # Orientation of the cation headgroups using fingerprints
        self.bool_headgroup_orientation = settings.get('headgroup_orientation', False)
        self.headgroup_center = settings.get('headgroup_center', '')
        self.headgroup_end = settings.get('headgroup_end', '')
        self.headgroup_fingerprint = [self.headgroup_center, self.headgroup_end]

        # Orientation of the organic cations using fingerprints
        self.bool_cation_orientation = settings.get('cation_orientation', False)
        self.vector_fingerprints = settings.get('vector_fingerprints', [])

        # Computation of helicity in the arrangement of cations
        self.bool_cation_helicity = settings.get('cation_helicity', False)
        self.bool_cation_inversion = settings.get('cation_inversion', False)
        self.helicity_vectors = settings.get('helicity_vectors', None)

        # Optional arguments for fingerprinting
        self.N_mps = settings.get('N_mps', 10)
        self.cutoff_tolerance = settings.get('cutoff_tolerance', 0.1)

        # File output settings
        self.log = settings.get('log', 'chiralipy.log')  # output file for structure log
        self.print_log = settings.get('print_log', False)  # boolean switch for printing of log to terminal
        self.folder = settings.get('folder', 'chiralipy-output')  # output folder for trajectory arrays

    def __load_model_system(self, mode):
        """
        Load the model system from a structure file and optionally from a trajectory.

        Parameters:
        - mode (str): The loading mode. Should be one of ['structure', 'trajectory'].

        If mode is 'trajectory':
        - Reads in the base structure.
        - Obtains unique fingerprints in the model system.
        - Reads in the trajectory skipping a specified interval (self.n_skip).

        Note:
        - For static structures a modified supercell is used in the computations.
        """
        self.get_base_structure() # read in base structure
        self.unique_fingerprints = self.get_unique_fingerprints() # get unique fingerprints in model system
        self.n_base_octahedra = self.n_octahedra # number of octahedra in base structure

        if mode == 'trajectory':
            self.n_supercell = 1
            self.read_trajectory(self.n_skip, self.trajectory_format) # read in trajectory
        elif mode == 'structure':
            self.n_supercell = 2
            self.structure = self.structure.repeat([self.n_supercell, self.n_supercell, self.n_supercell]) # create supercell for small cells
            self.structure = ase.build.tools.sort(self.structure, tags=self.structure.get_atomic_numbers())
            self.n_octahedra = self.n_octahedra * self.n_supercell ** 3
        else:
            raise ValueError(f"Invalid mode '{mode}'. Supported modes: ['structure', 'trajectory']")

    def get_base_structure(self):
        """
        Read the base structure from a VASP structure file (POSCAR/CONTCAR).

        Reads the atomic structure from the specified VASP structure file (POSCAR/CONTCAR)
        using the Atomic Simulation Environment (ASE) library and assigns it to the
        'structure' attribute of the class instance.

        Additionally, calculates and updates the following attributes:
        - n_cages (int): Number of cages based on the specified cation elements.
        - n_octahedra (int): Number of octahedra based on the specified center elements.

        Note:
        - The 'cation_elements' and 'center_elements' attributes should be defined
          to identify the relevant atoms for calculating 'n_cages' and 'n_octahedra'.
        """
        self.structure = ase.io.read(self.structure_path, format='vasp')
        self.n_cages = int(len(np.array([atom.index for atom in self.structure if atom.symbol in self.cation_elements])) / 2) # number of cages based on cations
        self.n_octahedra = len(np.array([atom.index for atom in self.structure if atom.symbol in self.center_elements])) # number of octahedra based on centers

    def get_unique_fingerprints(self):
        """
        Determine the unique atomic fingerprints in structure.

        Returns:
        - numpy.ndarray: An array containing the unique fingerprints of organic molecules in the structure.

        Notes:
        - Utilizes the `get_molecular_information` method to determine molecular information,
          such as the number of molecules, molecular indices, and molecular fingerprints.
        """
        organic_indices = np.array([atom.index for atom in self.structure if atom.symbol in self.organic_elements])
        n_molecules, molecular_indices, molecular_fps = self.get_molecular_information(self.structure[organic_indices], self.N_mps, self.cutoff_tolerance)
        unique_fps = np.unique(molecular_fps)

        return unique_fps

    def read_trajectory(self, n_skip, trajectory_format):
        """
        Read the trajectory from a VASP trajectory file (XDATCAR).

        Parameters:
        - n_skip (int): Number of frames to skip at the beginning of the trajectory.
        - trajectory_format (str): File format string for ASE trajectory reader.

        Reads the atomic trajectory from the specified VASP trajectory file (XDATCAR)
        using the Atomic Simulation Environment (ASE) library and assigns it to the
        'trajectory' attribute of the class instance.

        The 'n_skip' parameter allows skipping a specified number of frames at the beginning
        of the trajectory, useful for equilibration periods.

        Additionally, updates the 'n_frames' attribute with the total number of frames in the trajectory.
        """
        self.trajectory = ase.io.read(self.trajectory_path, index=':', format=trajectory_format)[n_skip:]
        self.n_frames = len(self.trajectory)

    def __initialize_descriptors(self):
        """
        Initializes various descriptors for structural analysis.

        This method initializes various descriptors used in the structural analysis
        of the model system. It includes the following steps:

        1. Obtain properties of layers, sublattices, and layers of interest.
        2. Initialization of intra/interoctahedral distortion descriptors.
        3. Initialization of planar distortion descriptors if enabled.
        4. Initialization of hydrogen bond asymmetry descriptors if enabled.
        5. Initialization of framework helicity descriptors if enabled.
        6. Initialization of perpendicular framework helicity descriptors if enabled.
        7. Initialization of N-H bond orientation descriptors if enabled.
        8. Initialization of cation orientation descriptors if enabled.
        9. Initialization of cation helicity descriptors if enabled.
        
        Raises:
        - SystemExit: If errors occur during the initialization process.
        """
        self.get_layer_properties() # get properties of layers
        self.get_octahedral_lattice() # create a lattice from the octahedral centers

        # Intra/interoctahedral distortion descriptors
        self.get_octahedral_indices() # obtain indices of octahedra
        self.get_octahedral_angle_indices() # obtain indices of angles in octahedra
        self.get_cage_sides() # define cage sides

        # Planar distortion descriptors
        if self.bool_planar_distortions:
            self.get_axial_indices() # define the axial species of an octahedron
            self.get_equatorial_indices() # define the equatorial species of an octahedron

        # Hydrogen bond asymmetry descriptors
        if self.bool_hydrogen_bond_asymmetry:
            self.get_hydrogen_bond_indices() # define the species involved in hydrogen bonding

        # Framework helicity descriptors
        if self.bool_planar_framework_helicity:
            self.get_axial_lines(tolerance=self.layer_tolerance) # obtain lines in in-plane directions; parallel to major axes
            self.get_diagonal_lines(tolerance=self.layer_tolerance) # obtain lines in in-plane directions; parallel to diagonals

        # Perpendicular framework helicity descriptors
        if self.bool_out_of_plane_framework_helicity:
            self.get_symmetry_linked_octahedral_centers() # obtain symmetry linked octahedral centers
            self.get_out_of_plane_lines() # obtain lines in out-of-plane directions; in symmetry linked pairs

        # N-H bond orientation
        if self.bool_headgroup_orientation:
            for fp in np.unique(np.array(self.headgroup_fingerprint)):
                if fp not in self.unique_fingerprints:
                    sys.exit('Error 1: Headgroup fingerprints could not be matched to the model system!')
            self.get_headgroup_indices() # define the headgroup vectors

        # Cation orientation
        if self.bool_cation_orientation:
            for fp in np.unique(np.array(self.vector_fingerprints)):
                if fp not in self.unique_fingerprints:
                    sys.exit('Error 2: Vector fingerprints could not be matched to the model system!')
            self.get_vector_indices() # define internal vectors in molecules

        # Cation helicity descriptors
        if self.bool_cation_helicity:
            if not self.bool_headgroup_orientation:
                sys.exit('Error 3: Center and ends of the cation headgroups have to be defined for cation helicity!') # molecule indexing
            if not (self.bool_cation_orientation or self.helicity_vectors):
                sys.exit('Error 4: Cation orientation vectors have to be defined for cation helicity!') # orientation vectors required for helicity
            elif self.helicity_vectors == None:
                sys.exit('Error 5: The helicity vectors have to be defined for cation helicity!') # indication of relevant vectors
            self.get_cation_lines() # obtain lines of cations; in symmetry linked pairs

    def get_layer_properties(self):
        """
        Convert Cartesian direction (x, y, and z) to Miller indices and normal vector.

        Determines the Miller indices of the planes and the unit normal vector
        corresponding to the specified Cartesian direction.

        The Cartesian direction is represented by an integer:
        - 0 corresponds to the x-direction,
        - 1 corresponds to the y-direction,
        - 2 corresponds to the z-direction.

        Updates the following attributes:
        - miller (numpy.ndarray): An array representing the Miller indices of the planes.
        - normal_vector (numpy.ndarray): The unit normal vector of the planes.
        """
        cart2miller = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}
        self.miller = np.array(cart2miller[self.direction]) # miller indices of the planes
        self.normal_vector = self.get_normal_vector(self.structure) # unit normal vector of the planes

    def get_normal_vector(self, frame):
        """
        Obtain the normalized normal vector in Cartesian coordinates to the orientation of inorganic planes as indicated by the Miller index.

        Parameters:
        - frame (Frame): An object representing the crystal lattice frame, containing unit cell information.

        Returns:
        - normal_vector (numpy.ndarray): A 3D numpy array representing the normalized normal vector in Cartesian coordinates to the specified inorganic planes.
        """
        cell = frame.get_cell()[:] # get unit cell information
        planar_vectors = cell[(np.ones(3) - self.miller).astype(np.bool_)] # select vectors within planes
        normal_vector = np.cross(planar_vectors[0], planar_vectors[1]) # compute normal vector
        normal_vector = np.sign(normal_vector[self.direction]) * normal_vector / np.linalg.norm(normal_vector) # normalize and orient normal vector

        return normal_vector

    def get_octahedral_lattice(self):
        """
        Set up an octahedral lattice and divide it into sublayers using the octahedral centers.

        This method initializes an octahedral lattice based on the input structure and center elements.
        It then further subdivides the lattice into layers and sublayers, storing the results in instance variables.
        """
        octahedral_lattice, n_octahedral_lattice = self.get_substructure(self.structure, self.center_elements)
        self.octahedral_lattice = octahedral_lattice
        self.n_octahedral_lattice = n_octahedral_lattice

        octahedral_layers, full_octahedral_layers, n_layers = self.get_substructure_layers(self.structure, octahedral_lattice, self.center_elements, tolerance=self.layer_tolerance)
        self.octahedral_layers = octahedral_layers
        self.full_octahedral_layers = full_octahedral_layers
        self.n_layers = n_layers

        octahedral_sublayers, full_octahedral_sublayers = self.get_substructure_sublayers(octahedral_layers, self.structure, octahedral_lattice, self.center_elements)
        self.octahedral_sublayers = octahedral_sublayers
        self.full_octahedral_sublayers = full_octahedral_sublayers

    def get_substructure(self, structure, elements):
        """
        Set up a sublattice containing species of specific elements.

        Parameters:
        - structure (ase.Atoms): Atomic structure.
        - elements (list): A list of element symbols representing the species to include in the sublattice.

        Returns:
        - substructure (ase.Atoms): Atomic structure of the subset of the structure.
        - n_substructure (int): Number of atoms in the subset of the structure.
        """
        substructure = structure[np.array([atom.index for atom in structure if atom.symbol in elements])]
        n_substructure = len(substructure)

        return substructure, n_substructure

    def get_substructure_layers(self, structure, substructure, elements, tolerance=0.5):
        """
        Identify and retrieve layers of a substructure within a larger structure.

        Parameters:
        - structure (ase.Atoms): The larger structure containing the substructure.
        - substructure (ase.Atoms): The substructure of interest to identify layers.
        - elements (list): List of element symbols in the substructure.
        - tolerance (float, optional): Tolerance value for layer identification.
                                       Defaults to 0.5.

        Returns:
        - layers (numpy.ndarray): Array representing the layer indices for each identified layer.
        - full_layers (numpy.ndarray): Array containing the full indices of the identified layers in the original structure.
        - n_layers (int): Number of layers identified.
        """
        tags, levels = ase.geometry.get_layers(substructure, self.miller, tolerance=tolerance)
        n_layers = len(levels)

        layers = np.array([np.where(tags == idx)[0] for idx in range(n_layers)], dtype=int)
        full_layers = self.get_full_indices(structure, elements, layers)

        return layers, full_layers, n_layers

    def get_full_indices(self, full_structure, elements, indices):
        """
        Convert the indices of species in a sublattice to the indices of the full lattice.

        Parameters:
        - full_structure (ase.Atoms): Atomic structure of the full material.
        - elements (list): A list of element symbols representing the species present in the substructure.
        - indices (numpy.ndarray): 2D numpy array containing the indices of species in the substructure.

        Returns:
        - full_indices (numpy.ndarray): 2D numpy array with indices of species in the full lattice corresponding to the input indices.
        """
        shape = indices.shape
        full_indices = np.array([atom.index for atom in full_structure if atom.symbol in elements])
        indices = indices.ravel()

        return full_indices[indices].reshape(shape)

    def get_substructure_sublayers(self, layers, structure, substructure, elements):
        """
        Identify and retrieve sublayers within each layer of a substructure.

        Parameters:
        - layers (numpy.ndarray): Array representing the layer indices for each identified layer.
        - structure (ase.Atoms): The larger structure containing the substructure.
        - substructure (ase.Atoms): The substructure of interest to identify sublayers.
        - elements (list): List of element symbols in the substructure.

        Returns:
        - sublayers (numpy.ndarray): Array representing the indices of sublayers within each layer.
        - full_sublayers (numpy.ndarray): Array containing the full indices of the identified sublayers in the original structure.
        """
        n_layers, n_atoms_layer = layers.shape
        n_atoms_sublayer = int(n_atoms_layer / 2)
        
        sublayers = np.zeros((n_layers, 2, n_atoms_sublayer), dtype=int)
        
        for idx, layer in enumerate(layers):
            sites1, sites2 = [], [] # storage array for sites
            to_scan, scanned = [layer[0]], [] # initialize array to keep track of scanning

            while to_scan:
                site = to_scan.pop(0) # investigate first site from list and remove
                scanned.append(site) # add investigated site to 'scanned' set

                first_shell, second_shell = self.get_neighbor_shells(substructure, site, layer, cutoffs=[7, 10])

                sites1.append(site) # store investigated site
                sites2.extend(first_shell) # store first shell
                sites1.extend(second_shell) # store second shell

                to_scan.extend(jdx for jdx in second_shell if jdx not in scanned)

            sublayers[idx, 0, :] = np.array(list(set(sites1)), dtype=int) # store unique sites_1
            sublayers[idx, 1, :] = np.array(list(set(sites2)), dtype=int) # store unique sites_2

        full_sublayers = self.get_full_indices(structure, elements, sublayers)

        return sublayers, full_sublayers

    def get_neighbor_shells(self, structure, site, indices, cutoffs=[7, 10]):
        """
        Obtain the first and second neighbor shells around a species within specified cutoffs.

        Parameters:
        - site (int): The central site around which the neighbor shells are obtained.
        - indices (numpy.ndarray): 1D numpy array containing the indices of species in the sublattice.
        - cutoffs (list): List of two distance cutoffs specifying the range for the first and second neighbor shells.

        Returns:
        - first_shell (numpy.ndarray): 1D numpy array containing indices of species in the first neighbor shell.
        - second_shell (numpy.ndarray): 1D numpy array containing indices of species in the second neighbor shell.
        """
        distances = structure.get_distances(site, indices, mic=True)

        sorted_indices = indices[np.argsort(distances)]
        sorted_distances = np.sort(distances)

        first_cutoff_idx = np.searchsorted(sorted_distances, cutoffs[0], side='left')
        second_cutoff_idx = np.searchsorted(sorted_distances, cutoffs[1], side='left')

        first_shell = sorted_indices[1:first_cutoff_idx]
        second_shell = sorted_indices[first_cutoff_idx:second_cutoff_idx]

        return first_shell, second_shell

    def get_octahedral_indices(self):
        """
        Obtain indices of centers and six corners of the octahedra.

        Sets the following attributes:
        - octahedral_centers (numpy.ndarray): 3D numpy array representing the indices of octahedral centers.
        - octahedral_corners (numpy.ndarray): 4D numpy array representing the indices of corners in the octahedra.
        """
        centers = self.full_octahedral_sublayers

        octahedral_corners = np.zeros((self.n_layers, 2, int(self.n_octahedral_lattice / self.n_layers / 2), 6), dtype=int)
        for idx, layer in enumerate(centers):
            for jdx, sublayer in enumerate(layer):
                for kdx, center in enumerate(sublayer):
                    octahedral_corners[idx, jdx, kdx, :] = self.get_sorted_neighbors(self.structure, center, self.corner_elements, self.normal_vector, 6)

        self.octahedral_centers = centers
        self.octahedral_corners = octahedral_corners

    def get_octahedral_angle_indices(self):
        """
        Obtain indices of the X-M-X angles in the octahedra.

        Sets the following attributes:
        """
        octahedral_centers = self.octahedral_centers.flatten()
        octahedral_corners = self.octahedral_corners.reshape(-1, self.octahedral_corners.shape[-1])

        octahedral_angle_indices = []
        for idx, (center, corners) in enumerate(zip(octahedral_centers, octahedral_corners)):
            top, plane, bottom = corners[0], corners[1:-1], corners[-1]

            angle_indices = []
            angle_indices.extend([[top, center, x] for x in plane])
            for i, x in enumerate(plane):
                angle_indices.extend([[x, center, y] for y in plane[i + 1:]])
            angle_indices.extend([[bottom, center, x] for x in plane])
            angle_indices = np.array(angle_indices)

            angles = self.structure.get_angles(angle_indices, mic=True)
            octahedral_angle_indices.append(angle_indices[angles <= 150])
        octahedral_angle_indices = np.array(octahedral_angle_indices)

        self.octahedral_angle_indices = octahedral_angle_indices

    def get_sorted_neighbors(self, structure, center, target_elements, vector, n_neighbors):
        """
        Obtain indices of neighbors sorted based on their projection along a specified vector.

        Parameters:
        - structure (ase.Atoms): The structure containing the atoms.
        - center (int): Index of the central atom for which neighbors are sought.
        - target_elements (list): List of element symbols considered as neighbors.
        - vector (numpy.ndarray): The vector along which the neighbors are projected.
        - n_neighbors (int): Number of neighbors to consider.

        Returns:
        - sorted_neighbors (numpy.ndarray): Array containing indices of neighbors sorted based on their projection along the specified vector.
        """
        # Determine distance to neighbors 
        targets = np.array([atom.index for atom in structure if atom.symbol in target_elements])
        distances = structure.get_distances(center, targets, mic=True)
        neighbors = np.argpartition(distances, n_neighbors)[:n_neighbors] # number of neighbors forming neighborhood

        # Project distance vectors in a direction
        vectors = structure.get_distances(center, targets[neighbors], mic=True, vector=True) # distance vectors of neighbors
        projected_vectors = np.sum(vectors * vector, axis=1) # projection of distance vectors

        # Sort the neighbors based on projected distance vectors
        sorted_neighbors = targets[neighbors[projected_vectors.argsort()]]

        return sorted_neighbors

    def get_cage_information(self):
        """
        Determine general information for inorganic cages.

        Returns:
        - A list containing the following information:
            - centers (numpy.ndarray): Indices of atoms representing cation centers.
            - corners (numpy.ndarray): Indices of atoms representing corners of the inorganic cages.
            - edges (numpy.ndarray): Indices of atoms representing edges of the inorganic cages.
            - h_atoms (numpy.ndarray): Indices of hydrogen atoms in the structure.
            - unique_centers (numpy.ndarray): Indices of cation centers belonging solely to an inorganic cage.
            - cages (numpy.ndarray): 2D numpy array containing indices of corners for each unique cage.
        """
        centers = np.array([atom.index for atom in self.structure if atom.symbol in self.cation_elements])
        corners = np.array([atom.index for atom in self.structure if atom.symbol in self.center_elements])
        edges   = np.array([atom.index for atom in self.structure if atom.symbol in self.corner_elements])
        h_atoms = np.array([atom.index for atom in self.structure if atom.symbol in ['H']])

        cages, unique_centers = [], []
        for center in centers:
            center_corner_distances = self.structure.get_distances(center, corners, mic=True)
            local_corners = center_corner_distances.argsort()[:4]
            cage_corners = corners[local_corners]
            if set(cage_corners) not in cages:
                unique_centers.append(center)
                cages.append(set(cage_corners))
        unique_centers = np.array(unique_centers)
        cages = np.array([list(x) for x in cages])

        return [centers, corners, edges, h_atoms, unique_centers, cages]

    def get_cage_sides(self):
        """
        Determine the sides of the inorganic cages.

        Sets the following attributes:
        - unique_centers (numpy.ndarray): Indices of unique cation centers in the cages.
        - all_cage_sides (numpy.ndarray): 3D numpy array representing the sides of the inorganic cages.
        """
        centers, corners, edges, h_atoms, unique_centers, cages = self.get_cage_information()

        # Sides of inorganic cages (M-X-M)
        all_cage_sides = np.zeros((len(unique_centers), 4, 3), dtype=int)
        
        for idx_center, center in enumerate(unique_centers):
            cage_corners = cages[idx_center]

            cage_octahedral_corners = np.zeros((len(cage_corners), 6), dtype=int)
            for idx_corner, cage_corner in enumerate(cage_corners):
                corner_edge_distances = self.structure.get_distances(cage_corner, edges, mic=True)
                local_octahedral_corners = corner_edge_distances.argsort()[:6]
                cage_octahedral_corners[idx_corner] = edges[local_octahedral_corners]

            idx_side = 0
            cage_sides, cage_angles = np.zeros((4, 3), dtype=int), np.zeros(4)
            for jdx in range(len(cage_corners)):
                for kdx in range(jdx + 1, len(cage_corners)):
                    cage_corner1, cage_corner2 = cage_corners[jdx], cage_corners[kdx]
                    common_edge = list(set(cage_octahedral_corners[jdx]).intersection(cage_octahedral_corners[kdx]))
                    if common_edge:
                        common_edge = common_edge[0]
                        cage_sides[idx_side] = np.array([cage_corner1, common_edge, cage_corner2])
                        cage_angles[idx_side] = self.structure.get_angle(cage_corner1, common_edge, cage_corner2, mic=True)
                        idx_side += 1

            sorted_indices = cage_angles.argsort()
            all_cage_sides[idx_center] = cage_sides[sorted_indices]

        self.unique_centers = unique_centers
        self.all_cage_sides = all_cage_sides

    def get_axial_indices(self):
        """
        Determine the indices of the atoms for axial planar distortions.

        Sets the following attribute:
        - axial_indices (numpy.ndarray): 4D numpy array representing the indices of atoms for axial planar distortions.
        """
        # indices: layer, sublayer, sites_in_sublayer, [M, X_below, X_above]
        axial_indices = np.zeros((self.n_layers, 2, int(self.n_octahedral_lattice / self.n_layers / 2), 3), dtype=int)

        axial_indices[:, :, :, 0] = self.octahedral_centers # M sites
        axial_indices[:, :, :, 1] = self.octahedral_corners[:, :, :, 0] # X sites; below
        axial_indices[:, :, :, 2] = self.octahedral_corners[:, :, :, -1] # X sites; above

        self.axial_indices = axial_indices

    def get_equatorial_indices(self):
        """
        Determine the indices of the atoms for equatorial planar distortions.

        Sets the following attribute:
        - equatorial_indices (numpy.ndarray): 4D numpy array representing the indices of atoms for equatorial planar distortions.
        """
        # indices: layer, sublayer, sites_in_sublayer, [X_1, X_2, X_3, X_4]
        equatorial_indices = np.zeros((self.n_layers, 2, int(self.n_octahedral_lattice / self.n_layers / 2), 4), dtype=int)

        equatorial_indices[:, :, :, :] = self.octahedral_corners[:, :, :, 1:-1]

        self.equatorial_indices = equatorial_indices

    def get_hydrogen_bond_indices(self):
        """
        Determine the species that interact with each other to form hydrogen bonds.

        Sets the following attribute:
        - all_hydrogen_bond_indices (numpy.ndarray): 3D numpy array representing the indices of species involved in hydrogen bonds.
        """
        centers, corners, edges, h_atoms, unique_centers, cages = self.get_cage_information()

        # Relevant hydrogen bonds (-NH3 - X)
        all_hydrogen_bond_indices = np.zeros((len(cages), 2, 3), dtype=int)

        for idx_cage, cage in enumerate(cages):
            local_centers = np.zeros((len(cage), 8), dtype=int)
            for idx_corner, corner in enumerate(cage):
                corner_center_distances = self.structure.get_distances(corner, centers, mic=True)
                local_centers[idx_corner] = corner_center_distances.argsort()[:8]

            center_pair = []
            for local_center in local_centers[0]:
                common = np.all(np.any(local_centers == local_center, axis=1))
                if common:
                    center_pair.append(centers[local_center])
                    if len(center_pair) > 1:
                        break

            avg_d_hb = np.zeros(2)
            hb_indices = np.zeros((2, 3), dtype=int)

            for idx_center, center in enumerate(center_pair):
                h_hb = h_atoms[self.structure.get_distances(center, h_atoms, mic=True).argsort()[:3]]

                d_hb = np.zeros(3)
                for jdx in range(3):
                    d = self.structure.get_distances(h_hb[jdx], edges, mic=True)
                    hb = d.argsort()[0]
                    d_hb[jdx] = d[hb]

                sorter = d_hb.argsort()
                sorted_d_hb = d_hb[sorter]
                sorted_h_hb = h_hb[sorter]

                hb_indices[idx_center, :] = sorted_h_hb
                avg_d_hb[idx_center] = np.min(sorted_d_hb[:-1])

            all_hydrogen_bond_indices[idx_cage] = hb_indices[avg_d_hb.argsort()]

        self.all_hydrogen_bond_indices = all_hydrogen_bond_indices

    def get_headgroup_indices(self):
        """
        Construct arrays with atom indices corresponding to the headgroups in the structure.

        Sets the following attributes:
        - headgroup_indices (numpy.ndarray): 2D numpy array, containing indices belonging to headgroup fingerprints.
        - n_headgroup_vectors (int): Integer, indicating the number of distinct headgroup vectors.
        - n_molecules (int): Integer, representing the total number of identified molecules.
        """
        headgroup_fp = np.array(self.headgroup_fingerprint)

        organic_indices = np.array([atom.index for atom in self.structure if atom.symbol in self.organic_elements])
        n_molecules, molecular_indices, molecular_fps = self.get_molecular_information(self.structure[organic_indices], self.N_mps, self.cutoff_tolerance)
        headgroup_indices = self.match_headgroup_fingerprint(headgroup_fp, molecular_fps, molecular_indices)

        n_molecules, n_headgroup_indices = headgroup_indices.shape
        n_headgroup_vectors = n_headgroup_indices - 1

        self.headgroup_indices = headgroup_indices
        self.n_headgroup_vectors = n_headgroup_vectors
        self.n_molecules = n_molecules

    def get_axial_lines(self, tolerance=0.5):
        """
        Extract inorganic lines in inorganic planes along the direction of the lattice vectors.

        This method identifies inorganic lines in the crystal structure based on the specified
        Miller index direction.

        Parameters:
        - tolerance (float, optional): Tolerance value for line identification.
                                       Defaults to 0.5.

        Sets the 'axial_line_indices' attribute of the object that represent the inorganic lines.
        """
        # Obtain lattice vectors of interest
        miller_axial = {
            0: np.array([[ 0, +1,  0],
                         [ 0,  0, +1]]),
            1: np.array([[ 0,  0, +1],
                         [+1,  0,  0]]),
            2: np.array([[+1,  0,  0],
                         [ 0, +1,  0]])
        }

        directions_axial = {
            0: np.array([[ 0,  0, +1],
                         [ 0, +1,  0]]),
            1: np.array([[+1,  0,  0],
                         [ 0,  0, +1]]),
            2: np.array([[ 0, +1,  0],
                         [+1,  0,  0]])
        }

        self.miller_axial = miller_axial[self.direction]
        self.directions_axial = directions_axial[self.direction]
        
        # Calculate the number of axial lines per layer and related parameters
        self.n_axial_lines = int(2 * ((self.n_octahedral_lattice / self.n_layers) / 2) ** 0.5) # number of inorganic lines per layer
        self.n_unique_axial_lines = int(self.n_axial_lines / 2)
        self.len_axial_lines = int(2 * self.n_axial_lines)

        # Initialize array to store axial line indices
        axial_line_indices = np.zeros((2, self.n_layers, 2, self.n_unique_axial_lines, self.len_axial_lines), dtype=int)

        for idx, (d, plane) in enumerate(zip(self.directions_axial, self.miller_axial)):
            d = self.structure.get_cell().cartesian_positions(d)
            d = d / np.linalg.norm(d)

            for jdx, layer in enumerate(self.octahedral_layers):
                # Create a supercell to prevent spurious effects of periodic boundary conditions
                reps = np.array([3, 3, 3]) - 2 * self.miller
                trans = (np.array([1, 1, 1]) - self.miller) * (1 / 3)
                tmp_structure = self.octahedral_lattice[layer].repeat(reps)
                tmp_structure.translate(tmp_structure.get_cell().cartesian_positions(trans))

                # Divide supercell into layers
                tags, levels = ase.geometry.get_layers(tmp_structure, plane, tolerance=tolerance)
                selected_lines = np.unique(tags)

                for kdx in range(self.n_unique_axial_lines):
                    kdx1, kdx2, kdx3 = 2 * kdx, (2 * kdx) + 1, (2 * kdx) + 2
                    
                    # Obtain lines of M sites from structure
                    central_indices = np.where(tags == kdx2)[0]
                    central_line = tmp_structure[central_indices]

                    M_line1 = np.where(tags == kdx1)[0] # neighbor line 1
                    M_line2 = np.where(tags == kdx3)[0] # neighbor line 2

                    # Select unique M sites in close proximity on the line as a segment
                    central_line_distances = central_line.get_distances(0, np.arange(len(central_line)), mic=True)
                    central_segment = central_line_distances.argsort()[:int(self.len_axial_lines / 4 + 1)]
                    central_segment_indices = central_indices[central_segment]

                    # Sort the indices of the M sites along the line directions
                    vector_distances = central_line.get_distances(0, np.arange(len(central_line)), mic=True, vector=True)
                    projected_distances = np.sum(vector_distances[central_segment] * d, axis=1)
                    central_segment_indices_sorted = central_segment_indices[projected_distances.argsort()]

                    # Complete lines of M sites by linking two neighboring lines
                    full_M_line1, full_M_line2 = [], []
                    for ldx in range(len(central_segment_indices_sorted) - 1):
                        atom1, atom2 = central_segment_indices_sorted[ldx], central_segment_indices_sorted[ldx + 1]

                        M_link1 = self.find_linking_element(tmp_structure, atom1, atom2, M_line1, 2)
                        M_link2 = self.find_linking_element(tmp_structure, atom1, atom2, M_line2, 2)

                        full_M_line1.extend([atom1, M_link1])
                        full_M_line2.extend([atom1, M_link2])
                    full_M_line1.append(atom2)
                    full_M_line2.append(atom2)

                    full_M_line1 = self.get_mapped_indices(np.array(full_M_line1), tmp_structure, self.structure)
                    full_M_line2 = self.get_mapped_indices(np.array(full_M_line2), tmp_structure, self.structure)

                    # Complete inorganic lines by adding halide sites
                    halide_indices = np.array([atom.index for atom in self.structure if (atom.symbol in self.corner_elements)])

                    full_line1, full_line2 = [], []
                    for mdx in range(len(full_M_line1) - 1):
                        atom1, atom2 = full_M_line1[mdx], full_M_line1[mdx + 1]
                        atom3, atom4 = full_M_line2[mdx], full_M_line2[mdx + 1]

                        X_link1 = self.find_linking_element(self.structure, atom1, atom2, halide_indices, 6)
                        X_link2 = self.find_linking_element(self.structure, atom3, atom4, halide_indices, 6)

                        full_line1.extend([atom1, X_link1])
                        full_line2.extend([atom3, X_link2])

                    axial_line_indices[idx, jdx, 0, kdx, :] = np.array(full_line1)
                    axial_line_indices[idx, jdx, 1, kdx, :] = np.array(full_line2)

        self.axial_line_indices = axial_line_indices

    def get_diagonal_lines(self, tolerance=0.5):
        """
        Extract inorganic lines in inorganic planes in the diagonal direction.

        This method identifies inorganic lines in the crystal structure based on the specified
        Miller index direction.

        Parameters:
        - tolerance (float, optional): Tolerance value for line identification.
                                       Defaults to 0.5.

        Sets the 'diagonal_line_indices' attribute of the object that represent the inorganic lines.
        """
        # Obtain lattice directions of interest
        miller_diagonal = {
            0: np.array([[ 0, +1, +1],
                         [ 0, +1, -1]]),
            1: np.array([[+1,  0, +1],
                         [-1,  0, +1]]),
            2: np.array([[+1, +1,  0],
                         [+1, -1,  0]])
        }

        directions_diagonal = {
            0: np.array([[ 0, +1, -1],
                         [ 0, +1, +1]]),
            1: np.array([[-1,  0, +1],
                         [+1,  0, +1]]),
            2: np.array([[+1, -1,  0],
                         [+1, +1,  0]])
        }

        self.miller_diagonal = miller_diagonal[self.direction]
        self.directions_diagonal = directions_diagonal[self.direction]

        # Calculate the number of diagonal lines per layer and related parameters
        self.n_diagonal_lines = int(((self.n_octahedral_lattice / self.n_layers) / 2) ** 0.5) # number of inorganic lines per layer
        self.len_diagonal_lines = int(2 * self.n_octahedral_lattice / self.n_layers / self.n_diagonal_lines) # length of inorganic lines

        # Initialize array to store diagonal line indices
        diagonal_line_indices = np.zeros((2, self.n_layers, self.n_diagonal_lines, self.len_diagonal_lines), dtype=int)

        for idx, (d, plane) in enumerate(zip(self.directions_diagonal, self.miller_diagonal)):
            d = self.structure.get_cell().cartesian_positions(d)
            d = d / np.linalg.norm(d)

            for jdx, layer in enumerate(self.octahedral_layers):
                # Create a supercell to prevent spurious effects of periodic boundary conditions
                reps = np.array([3, 3, 3]) - 2 * self.miller
                trans = (np.array([1, 1, 1]) - self.miller) * (1 / 3)
                tmp_structure = self.octahedral_lattice[layer].repeat(reps)
                tmp_structure.translate(tmp_structure.get_cell().cartesian_positions(trans))

                # Divide supercell into layers
                tags, levels = ase.geometry.get_layers(tmp_structure, plane, tolerance=tolerance)
                selected_lines = np.unique(tags)

                kdx = 0
                for line in selected_lines:
                    # Obtain lines of M sites from structure
                    selected_indices = np.where(tags == line)[0]
                    selected_line = tmp_structure[selected_indices]

                    # Select unique M sites in close proximity on the line as a segment
                    line_distances = selected_line.get_distances(0, np.arange(len(selected_line)), mic=True)
                    line_segment = line_distances.argsort()[:int(self.len_diagonal_lines / 2 + 1)]
                    line_segment_indices = selected_indices[line_segment]

                    # Sort the indices of the M sites along the line directions
                    vector_distances = selected_line.get_distances(0, np.arange(len(selected_line)), mic=True, vector=True)
                    projected_distances = np.sum(vector_distances[line_segment] * d, axis=1)
                    line_segment_indices_sorted = line_segment_indices[projected_distances.argsort()]

                    # Mapped indices for duplicate checks
                    full_M_line = self.get_mapped_indices(np.array(line_segment_indices_sorted), tmp_structure, self.structure)
                    if len(full_M_line) == int(self.len_diagonal_lines / 2 + 1) and full_M_line[0] not in diagonal_line_indices[idx]:
                        halide_indices = np.array([atom.index for atom in self.structure if (atom.symbol in self.corner_elements)])
                        
                        # Complete inorganic lines by adding halide sites
                        full_line = []
                        for mdx in range(len(full_M_line) - 1):
                            atom1, atom2 = full_M_line[mdx], full_M_line[mdx + 1]

                            X_link = self.find_linking_element(self.structure, atom1, atom2, halide_indices, 6)

                            full_line.extend([atom1, X_link])

                        diagonal_line_indices[idx, jdx, kdx] = np.array(full_line)
                        kdx += 1

        self.diagonal_line_indices = diagonal_line_indices

    def find_linking_element(self, s, atom1, atom2, subset, n_elements):
        """
        Finds a linking element between two atoms in a subset of a structure.

        Parameters:
        - s (ase.Atoms): The structure containing atoms.
        - atom1, atom2 (int): The index of the atoms between which the linking element is sought.
        - subset (numpy.ndarray): Subset of atoms within which to search for the linking element.
        - n_elements (int): Number of elements to consider as potential linking elements; neighbors.

        Returns:
        - linking_element (int): The index of the identified linking element.
        """
        # Calculate distances from atom1 and atom2 to the subset of atoms
        distances1 = s.get_distances(atom1, subset, mic=True)
        distances2 = s.get_distances(atom2, subset, mic=True)

        # Find the nearest neighbors for atom1 and atom2 within the subset
        neighbors1 = set(subset[distances1.argsort()][:n_elements])
        neighbors2 = set(subset[distances2.argsort()][:n_elements])

        # Return the common element among the nearest neighbors of both atoms
        linking_element = neighbors1.intersection(neighbors2).pop()

        return linking_element

    def get_symmetry_linked_octahedral_centers(self):
        """
        Calculate symmetry-linked octahedral centers for a crystal structure.

        This method identifies pairs of atoms in different sublayers that are symmetry-linked.

        Sets the following attribute:
        - symmetry_linked_octahedral_centers: 3D numpy array representing the indices of octahedral centers linked by symmetry.
        """
        symmetry_links = {
            0: np.array([ 0, +1, +1]),
            1: np.array([+1,  0, +1]),
            2: np.array([+1, +1,  0])
        }

        symmetry_link = self.structure.get_cell().cartesian_positions(symmetry_links[self.direction])
        symmetry_link = symmetry_link / np.linalg.norm(symmetry_link)
        self.symmetry_link = symmetry_link

        sublayers1 = self.octahedral_sublayers[:, 0, :]
        sublayers2 = self.octahedral_sublayers[:, 1, :]

        n_layers, n_atoms_sublayers = sublayers1.shape # number of layers; number of atoms per sublayer
        n_neighbors = 4 # number of direct neighbors

        symmetry_linked_octahedral_centers = np.zeros((n_layers, n_atoms_sublayers, 2), dtype=int)
        for idx, (sublayer1, sublayer2) in enumerate(zip(sublayers1, sublayers2)):
            for jdx, atom in enumerate(sublayer1):
                # Select direct neighbors from other sublayer
                d = self.octahedral_lattice.get_distances(atom, sublayer2, mic=True)
                neighbors = sublayer2[d.argsort()][:n_neighbors]

                # Find the symmetry linked atom in the sublayer
                v = self.octahedral_lattice.get_distances(atom, neighbors, mic=True, vector=True)
                v = v / np.linalg.norm(v, axis=1, keepdims=True) # normalized vector
                projected_v = np.sum(v * self.symmetry_link, axis=1) # project distance vectors along symmetry link
                atom_linked = neighbors[projected_v.argsort()[-1]] # index of symmetry linked atom

                symmetry_linked_octahedral_centers[idx, jdx, :] = self.get_mapped_indices(np.array([atom, atom_linked]), self.octahedral_lattice, self.structure) # story symmetry linked pair

        self.symmetry_linked_octahedral_centers = symmetry_linked_octahedral_centers

    def get_out_of_plane_lines(self):
        """
        Calculate out-of-plane lines for symmetry-linked octahedral centers.

        This method initializes and populates an array with indices representing
        out-of-plane lines for each layer of symmetry-linked octahedral centers.

        Sets the following attribute:
        - out_of_plane_line_indices (numpy.ndarray): 4D numpy array representing the line segments in the out-of-plane direction.
        """
        symmetry_linked_octahedral_centers = self.symmetry_linked_octahedral_centers
        corners = np.array([atom.index for atom in self.structure if atom.symbol in self.corner_elements])

        n_layers, n_lines_layer, _ = symmetry_linked_octahedral_centers.shape # number of layers; number of atoms per sublayer
        n_vectors = 4 # number of vectors of out-of-plane helicity

        # Initialize array to store out-of-plane line indices
        out_of_plane_line_indices = np.zeros((self.n_layers, n_lines_layer, n_vectors, 2), dtype=int)

        for idx, layer in enumerate(symmetry_linked_octahedral_centers):
            for jdx, (center1, center2) in enumerate(layer):
                bottom1, _, _, _, _, top1 = self.get_sorted_neighbors(self.structure, center1, self.corner_elements, self.normal_vector, 6)
                bottom2, _, _, _, _, top2 = self.get_sorted_neighbors(self.structure, center2, self.corner_elements, self.normal_vector, 6)

                out_of_plane_line_indices[idx, jdx, :, :] = np.array([[bottom1, center1],
                                                                      [center1,    top1],
                                                                      [bottom2, center2],
                                                                      [center2,    top2]])

        self.out_of_plane_line_indices = out_of_plane_line_indices

    def get_molecular_information(self, s, N_mps, cutoff_tolerance):
        """
        Identify molecular indices and fingerprints from a structure.

        Parameters:
        - s (ase.Atoms): Atomic structure.
        - N_mps (int): Maximum number of atoms to consider for molecular identification.
        - cutoff_tolerance (float): Tolerance added to natural cutoffs for neighbor list construction.

        Returns:
        - n_molecules (int): Number of identified molecules.
        - molecular_indices (numpy.ndarray): Array containing indices of atoms belonging to each molecule.
        - molecular_fps (numpy.ndarray): Array containing fingerprints of each molecule.
        """
        cutoffs = np.array(nl.natural_cutoffs(s)) + cutoff_tolerance # generate standard cutoffs
        nlist = nl.build_neighbor_list(s, cutoffs=cutoffs, self_interaction=False) # create neighborlist

        cmat = nlist.get_connectivity_matrix(sparse=False) # connectivity matrix
        dmat = nl.get_distance_matrix(cmat, limit=N_mps).toarray() # distance matrix
        n_molecules, indices_molecules = sparse.csgraph.connected_components(cmat) # determine molecules from graph

        # Iterate over all molecules
        molecular_indices, molecular_fps = [], []
        for idx in range(n_molecules):
            mol_indices = np.array([i for i in range(len(indices_molecules)) if indices_molecules[i] == idx], dtype=int) # indices of species in molecule
            molecular_indices.append(mol_indices) # store molecular indices
            molecular_fps.append(self.get_molecular_fingerprint(s, dmat, mol_indices)) # store molecular fingerprints
        molecular_indices, molecular_fps = np.array(molecular_indices), np.array(molecular_fps)

        return n_molecules, molecular_indices, molecular_fps

    def match_headgroup_fingerprint(self, headgroup_fp, molecular_fps, indices):
        """
        Match the fingerprints of atoms constituting the head group with the molecular fingerprints.

        Parameters:
        - headgroup_fp (numpy.ndarray): Array containing fingerprints of atoms in the head group.
        - molecular_fps (numpy.ndarray): Array containing fingerprints of each molecule.
        - indices (numpy.ndarray): Array containing indices of atoms corresponding to molecular fingerprints.

        Returns:
        - headgroup_indices (numpy.ndarray): 2D array representing matched indices of headgroup fingerprints for each molecule.
        """
        n_molecules, _ = molecular_fps.shape

        headgroup_indices = np.zeros((n_molecules, 4), dtype=int)
        for idx, fp in enumerate(headgroup_fp):
            all_rows, all_cols = np.where(molecular_fps == fp) # match fingerprints
            n_matches = len(all_rows)

            n_multiple = int(n_matches / n_molecules)
            for jdx in range(n_multiple):
                    rows, cols = all_rows[jdx::n_multiple], all_cols[jdx::n_multiple] 
                    headgroup_indices[:, idx + jdx] = indices[rows, cols]

        return headgroup_indices

    def get_molecular_fingerprint(self, s, dmat, atom_indices):
        """
        Determine a string-based atomic fingerprint for the whole molecule.

        Parameters:
        - s (ase.Atoms): Atomic structure.
        - dmat (numpy.ndarray): Distance matrix.
        - atom_indices (numpy.ndarray): Array containing indices of atoms in the molecule.

        Returns:
        - molecular_fingerprint (list): List of string-based atomic fingerprints for each atom in the molecule.
        """
        elements = np.array(s.get_chemical_symbols())[atom_indices] # elements in structure
        unique_elements = np.unique(elements) # unique elements in structure

        molecular_dmat = dmat[atom_indices, :][:, atom_indices]

        molecular_fingerprint = []
        for idx in range(len(atom_indices)):
            distances = molecular_dmat[idx, :]
            fp = {}
            for element in unique_elements:
                active_indices = np.where(elements == element)[0]
                active_distances = distances[active_indices]
                fp[element] = np.sum(active_distances)
            molecular_fingerprint.append(self.fingerprint2string(fp))

        return molecular_fingerprint

    def fingerprint2string(self, fp):
        """
        Convert the fingerprint dictionary to a string.

        Parameters:
        - fp (dict): Fingerprint dictionary containing chemical symbols and their corresponding counts.

        Returns:
        - s (str): String representation of the fingerprint.

        Example:
        fp = {'H': 52, 'C': 18, 'N': 5}
        s = fingerprint2string(fp) = 'H52C18N5'
        """
        ptable = ase.data.chemical_symbols
        
        s = ''.join('{:}{:d}'.format(atom, fp[atom]) for atom in ptable if atom in fp)

        return s

    def get_vector_indices(self):
        """
        Construct arrays with atom indices corresponding to specific structural vectors in the structure.

        Sets the following attributes:
        - vector_indices (numpy.ndarray): 3D numpy array, containing indices belonging to vector fingerprints.
        - n_headgroup_vectors (int): Integer, indicating the number of distinct structural vectors.
        - n_molecules (int): Integer, representing the total number of identified molecules.
        """
        vector_fps = np.array(self.vector_fingerprints)

        organic_idxs = np.array([atom.index for atom in self.structure if atom.symbol in self.organic_elements])
        n_molecules, molecular_idxs, molecular_fps = self.get_molecular_information(self.structure[organic_idxs], self.N_mps, self.cutoff_tolerance)
        vector_indices = self.match_vector_fingerprints(vector_fps, molecular_fps, molecular_idxs)

        n_vectors, n_molecules, _ = vector_indices.shape

        self.vector_indices = vector_indices
        self.n_vectors = n_vectors
        self.n_molecules = n_molecules

    def match_vector_fingerprints(self, vector_fps, molecular_fps, indices):
        """
        Match the fingerprints of the vector endpoints with the molecule fingerprints.

        Parameters:
        - vector_fps (numpy.ndarray): Array containing fingerprints of the vector endpoints.
        - molecular_fps (numpy.ndarray): Array containing molecular fingerprints.
        - indices (numpy.ndarray): Array containing indices of atoms in the molecules.

        Returns:
        - vector_indices (numpy.ndarray): Array containing matched indices in the molecules for each vector.
        """
        n_vectors, _ = vector_fps.shape
        n_molecules, _ = molecular_fps.shape

        vector_indices = np.zeros((n_vectors, n_molecules, 2), dtype=int)
        for idx, vector_fp in enumerate(vector_fps):
            for jdx, fp in enumerate(vector_fp):
                rows, cols = np.where(molecular_fps == fp) # match fingerprints
                n_matches = len(rows)

                if n_matches > n_molecules:
                    rows, cols = rows[jdx::2], cols[jdx::2]

                vector_indices[idx, :, jdx] = indices[rows, cols]

        return vector_indices

    def get_mapped_indices(self, indices, s, s_base):
        """
        Convert the indices of a given supercell of a structure to those in the smaller base structure.

        Parameters:
        - indices (numpy.ndarray): Array containing indices of atoms in the supercell.
        - s (ase.Atoms): Supercell atomic structure.
        - s_base (ase.Atoms): Base atomic structure.

        Returns:
        - base_indices (numpy.ndarray): Array containing mapped indices in the smaller base structure.
        """
        base_cell = s_base.get_cell() # small basis cell
        base_positions = np.around(np.around(s_base.get_scaled_positions(), decimals=3) % 1, decimals=3) # positions in the basis cell

        positions = s[indices].get_positions() # supercell positions
        mapped_positions = np.around(np.around(base_cell.scaled_positions(positions), decimals=3) % 1, decimals=3) # supercell positions mapped onto basis cell

        base_indices = np.zeros(len(indices), dtype=int)
        for idx, (idx_atom, r) in enumerate(zip(indices, mapped_positions)):
            base_idx = np.squeeze(np.where(np.all(base_positions == r, axis=1)))
            base_indices[idx] = base_idx

        return base_indices

    def get_cation_lines(self):
        """
        Generate symmetry-linked organic cations and associated vectors.

        This method performs a series of steps to generate symmetry-linked organic cations
        and associated vectors within a combined structure of inorganic cage centers and organic cation headgroups.

        Steps:
        1. Define symbols for inorganic cage centers and organic cations (default: 'C' and 'N').
        2. Create a combined structure of cage centers and cation headgroups.
        3. Create a subset of the structure containing only inorganic cage centers.
        4. Identify layers within the subset of cage centers.
        5. Extract sublayers within each layer of cage centers.
        6. Obtain symmetry-linked inorganic cages based on sublayers.
        7. Obtain full indices of symmetry-linked inorganic cages in the combined structure.
        8. Obtain symmetry-linked organic cations based on symmetry-linked inorganic cages.
        9. Obtain full indices of symmetry-linked organic cations in the original structure.
        10. Match symmetry-linked cation indices to orientation vectors.
        11. Store symmetry-linked cations and associated vectors in the class attributes.

        Sets the following attributes:
        - symmetry_linked_cations (numpy.ndarray): Array containing pairs of indices representing symmetry-linked sites within each sublayer.
        - symmetry_linked_vectors (numpy.ndarray): Array containing the indices of symmetry-linked organic cations within each cage.
        """
        # Atom symbols
        cage_symbol = 'C'
        cation_symbol = 'N'

        # Create a combined structure of cage centers and cation headgroups
        cage_center_positions = self.get_cage_center_positions()
        cation_headgroup_positions = self.get_cation_headgroup_positions()
        combined_structure = self.create_cage_centers_cation_headgroups_structure(cage_center_positions, cation_headgroup_positions, symbols=[cage_symbol, cation_symbol])

        # Create a subset of the structure of the cage centers
        substructure, n_substructure = self.get_substructure(combined_structure, [cage_symbol])
        layers, full_layers, n_layers = self.get_substructure_layers(combined_structure, substructure, cage_symbol, tolerance=self.layer_tolerance)
        sublayers, full_sublayers = self.get_substructure_sublayers(layers, combined_structure, substructure, [cage_symbol])

        # Obtain symmetry linked inorganic cages
        symmetry_linked_cages = self.get_symmetry_linked_sites(substructure, sublayers)
        symmetry_linked_cages = self.get_full_indices(combined_structure, [cage_symbol, cation_symbol], symmetry_linked_cages)

        # Obtain symmetry linked organic cations
        symmetry_linked_cations = self.get_symmetry_linked_cations(combined_structure, symmetry_linked_cages)
        symmetry_linked_cations = self.get_full_indices(self.structure, ['N'], symmetry_linked_cations)
        self.symmetry_linked_cations = symmetry_linked_cations

        # Obtain symmetry linked organic vectors
        symmetry_linked_vectors = self.match_cation_indices(symmetry_linked_cations, self.headgroup_indices[:, 0]) # match cation indices to orientation vectors
        self.symmetry_linked_vectors = symmetry_linked_vectors

    def get_cage_center_positions(self):
        """
        Get fractional coordinates of inorganic cage centers.

        Returns:
        - r_centers (numpy.ndarray): An array containing the fractional coordinates of the inorganic cage centers.
                                     Each row corresponds to a unique inorganic cage.
        """
        all_cage_sides = self.all_cage_sides # indices of cage sides; corners and edges
        unique_centers = self.unique_centers # unique cation headgroups belonging to cage
        n_cages = len(all_cage_sides) # number of unique inorganic cages

        scaled_positions = self.structure.get_scaled_positions()
        r_centers = np.zeros((n_cages, 3))

        for idx, (cage_sides, unique_center) in enumerate(zip(all_cage_sides, unique_centers)):
            cage_corners = np.unique(cage_sides[:, ::2]) # extract the indices of the cage corners; first and third column

            r_ref = scaled_positions[unique_center]
            r = scaled_positions[cage_corners]
            r_mic = r - np.round(r - r_ref)
            r_centers[idx] = np.mean(r_mic, axis=0)

        return r_centers

    def get_cation_headgroup_positions(self):
        """
        Get fractional coordinates of cation headgroups.

        Returns:
        - r_headgroups (numpy.ndarray): An array containing the fractional coordinates of cation headgroups.
                                        Each row corresponds to a unique cation headgroup.
        """
        cation_headgroups = np.array([atom.index for atom in self.structure if atom.symbol in self.cation_elements])
        n_cations = len(cation_headgroups)

        scaled_positions = self.structure.get_scaled_positions()
        r_headgroups = np.zeros((n_cations, 3))

        for idx, cation_headgroup in enumerate(cation_headgroups):
            r_headgroups[idx] = scaled_positions[cation_headgroup]

        return r_headgroups

    def create_cage_centers_cation_headgroups_structure(self, r_centers, r_headgroups, symbols=['C', 'N']):
        """
        Create a combined structure with inorganic cage centers and cation headgroup positions.

        Parameters:
        - r_centers (numpy.ndarray): Array containing the fractional coordinates of inorganic cage centers.
        - r_headgroups (numpy.ndarray): Array containing the fractional coordinates of cation headgroups.
        - symbols (list, optional): Symbols for the inorganic cage center and cation headgroup, respectively.
                                      Defaults to ['C', 'N'].

        Returns:
        - combined_structure (ase.Atoms): Atomic structure representing the combined structure with inorganic cage centers and cation headgroups.
        """
        unit_cell = self.structure.get_cell()
        cage_symbol, cation_symbol = symbols
        n_centers = len(r_centers)
        n_headgroups = len(r_headgroups)

        r_centers_scaled = unit_cell.cartesian_positions(r_centers)
        centers = ase.Atoms('{:}{:.0f}'.format(cage_symbol, n_centers), positions=r_centers_scaled, cell=unit_cell, pbc=[1, 1, 1])

        r_headgroups_scaled = unit_cell.cartesian_positions(r_headgroups)
        headgroups = ase.Atoms('{:}{:.0f}'.format(cation_symbol, n_headgroups), positions=r_headgroups_scaled, cell=unit_cell, pbc=[1, 1, 1])

        combined_structure = centers + headgroups

        return combined_structure

    def get_symmetry_linked_sites(self, structure, sublayers):
        """
        Identify symmetry-linked sites within sublayers of a structure.

        Parameters:
        - structure (ase.Atoms): The larger structure containing the sublayers.
        - sublayers (numpy.ndarray): Array representing the indices of sublayers within each layer.

        Returns:
        - symmetry_linked_sites (numpy.ndarray): Array containing pairs of indices representing symmetry-linked sites within each sublayer.
        """
        symmetry_links = {
            0: np.array([ 0, +1, +1]),
            1: np.array([+1,  0, +1]),
            2: np.array([+1, +1,  0])
        }

        symmetry_link = structure.get_cell().cartesian_positions(symmetry_links[self.direction])
        symmetry_link = symmetry_link / np.linalg.norm(symmetry_link)

        sublayers1 = sublayers[:, 0, :]
        sublayers2 = sublayers[:, 1, :]

        n_layers, n_atoms_sublayers = sublayers1.shape # number of layers; number of atoms per sublayer
        n_neighbors = 4 # number of direct neighbors

        symmetry_linked_sites = np.zeros((n_layers, n_atoms_sublayers, 2), dtype=int)
        for idx, (sublayer1, sublayer2) in enumerate(zip(sublayers1, sublayers2)):
            for jdx, atom in enumerate(sublayer1):
                # Select direct neighbors from other sublayer
                d = structure.get_distances(atom, sublayer2, mic=True)
                neighbors = sublayer2[d.argsort()][:n_neighbors]

                # Find the symmetry linked atom in the sublayer
                v = structure.get_distances(atom, neighbors, mic=True, vector=True)
                v = v / np.linalg.norm(v, axis=1, keepdims=True) # normalized vector
                projected_v = np.sum(v * symmetry_link, axis=1) # project distance vectors along symmetry link
                atom_linked = neighbors[projected_v.argsort()[-1]] # index of symmetry linked atom

                symmetry_linked_sites[idx, jdx, :] = np.array([atom, atom_linked]) # store symmetry linked atom pair

        return symmetry_linked_sites

    def get_symmetry_linked_cations(self, structure, cage_indices):
        """
        Identify and retrieve symmetry-linked organic cations based on cage indices.

        Parameters:
        - structure (ase.Atoms): The larger structure containing the organic cations.
        - cage_indices (numpy.ndarray): Array representing the indices of cage atoms within each layer.

        Returns:
        - cation_indices (numpy.ndarray): Array containing the indices of symmetry-linked organic cations within each cage.
        """
        n_layers, n_sites, n_cages_per_link = cage_indices.shape
        n_cations_per_link = 2 * n_cages_per_link

        cation_indices = np.zeros((n_layers, n_sites, n_cations_per_link), dtype=int)
        for idx, layer in enumerate(cage_indices):
            for jdx, (atom1, atom2) in enumerate(layer):
                cation_indices[idx, jdx, :2] = self.get_sorted_neighbors(structure, atom1, self.cation_elements, self.normal_vector, 2)
                cation_indices[idx, jdx, 2:] = self.get_sorted_neighbors(structure, atom2, self.cation_elements, self.normal_vector, 2)

        cation_indices = cation_indices - np.min(cation_indices)

        return cation_indices

    def match_cation_indices(self, cation_indices, target_indices):
        """
        Match cation indices based on a set of target headgroups.

        Parameters:
        - cation_indices (numpy.ndarray): Array containing indices of cations to be matched.
        - target_indices (numpy.ndarray): Array containing indices of target headgroups for matching.

        Returns:
        - matched_cation_indices (numpy.ndarray): Array containing matched indices of cations based on the target headgroups.
        """
        shape = cation_indices.shape
        flat_cation_indices = cation_indices.ravel()

        matched_cation_indices = []
        for index in flat_cation_indices:
            matched_cation_indices.append(np.where(target_indices == index)[0][0])
        matched_cation_indices = np.array(matched_cation_indices).reshape(shape)

        return matched_cation_indices

    def __evaluate_descriptors(self, mode):
        """
        Evaluate the descriptors for the model system for a single structure or along a trajectory.

        Parameters:
        - mode (str): Specifies the mode; ['structure', 'trajectory'].
        """
        if mode == 'structure':
            self.__analyze_structure(self.structure, only_inorganic=False) # analyze structure, including organic descriptors
        elif mode == 'trajectory':
            if self.average:
                self.get_average_structure(self.output_average) # obtain averaged atomic positions
                self.__analyze_structure(self.average_structure, only_inorganic=True) # analyze averaged trajectory, exclude organic descriptors
            self.__analyze_trajectory() # analyze trajectory

    def get_average_structure(self, output_file):
        """
        Average the coordinates in the trajectory into an averaged structure.

        Parameters:
        - output_file (str): Path to the output file where the average structure will be saved.
        """
        centers = np.array([atom.index for atom in self.structure if atom.symbol in self.center_elements])

        initial_frame = self.trajectory[0]
        initial_positions = initial_frame.get_scaled_positions()
        initial_com = initial_frame[centers].get_center_of_mass(scaled=True)

        avg_cell = np.zeros((3, 3))
        avg_positions = np.zeros((len(self.structure), 3))

        for idx, frame in tqdm(enumerate(self.trajectory), total=self.n_frames, desc='Structure averaging'):
            scaled_positions = frame.get_scaled_positions()
            scaled_com = frame[centers].get_center_of_mass(scaled=True)

            delta_pbc = np.round(scaled_positions - initial_positions)
            delta_com = scaled_com - initial_com

            avg_cell += frame.get_cell()[:] / self.n_frames
            avg_positions += (scaled_positions - delta_pbc - delta_com) / self.n_frames

        average_structure = copy.deepcopy(self.structure)
        average_structure.set_cell(avg_cell)
        average_structure.set_scaled_positions(avg_positions)
        average_structure.write(output_file, format='vasp')

        self.average_structure = average_structure

    def __analyze_structure(self, structure, only_inorganic=False):
        """
        Analyze a single structure using various structural descriptors.

        Parameters:
        - structure (ase.Atoms): The atomic structure to be analyzed.
        - only_inorganic (bool): If True, only analyze inorganic descriptors.

        The function evaluates several structural descriptors based on the specified analysis flags:
        - Intraoctahedral distortions
        - Interoctahedral distortions
        - Planar distortions
        - Hydrogen bond asymmetry
        - Framework helicity (in-plane)
        - Framework helicity (out-of-plane)
        - Headgroup orientation vectors
        - Cation orientation vectors and helicity

        The results of the analysis are stored in class attributes such as `dict_intraoctahedral_distortions`,
        `dict_interoctahedral_distortions`, `dict_planar_distortions`, `dict_hydrogen_bond_asymmetry`,
        `dict_framework_helicity`, `dict_perpendicular_framework_helicity`, and `dict_cation_helicity`.

        Additionally, a summary of the analysis is output to the specified log.
        """
        # Intraoctahedral distortions: Robinson et al. Science (1971) DOI: 10.1126/science.172.3983.567
        if self.bool_intraoctahedral_distortions:
            intraoctahedral_distortions, octahedral_bond_lengths, octahedral_bond_angles = self.get_intraoctahedral_distortions(structure)

            # intraoctahedral_distortions, octahedral_bond_lengths, octahedral_bond_angles
            delta_d, sigma2 = intraoctahedral_distortions.T
            delta_d = np.mean(delta_d)
            sigma2 = np.mean(sigma2)

            self.dict_intraoctahedral_distortions = {
                'delta_d': delta_d,
                'sigma2': sigma2,
            }

            # octahedral geometry: bond_lengths and bond_angles
            bond_lengths = octahedral_bond_lengths[::self.n_supercell ** 2][:self.n_base_octahedra]
            bond_angles = octahedral_bond_angles[::self.n_supercell ** 2][:self.n_base_octahedra]

            self.dict_octahedral_geometry = {
                'bond_lengths': bond_lengths,
                'bond_angles': bond_angles,
            }

        # Interoctahedral distortions: Jana et al. Nat. Commun. (2021) DOI: 10.1038/s41467-021-25149-7
        if self.bool_interoctahedral_distortions:
            interoctahedral_distortions = self.get_interoctahedral_distortions(structure)
            delta_beta, delta_beta_in, delta_beta_out, max_D, max_D_in, max_D_out = interoctahedral_distortions.T

            delta_beta = np.mean(delta_beta)
            delta_beta_in = np.mean(delta_beta_in)
            delta_beta_out = np.mean(delta_beta_out)

            max_D = np.max(max_D)
            max_D_in = np.max(max_D_in)
            max_D_out = np.max(max_D_out)

            self.dict_interoctahedral_distortions = {
                'delta_beta': delta_beta,
                'delta_beta_in': delta_beta_in,
                'delta_beta_out': delta_beta_out,
                'max_D': max_D,
                'max_D_in': max_D_in,
                'max_D_out': max_D_out,
            }

        # Planar distortions: Apergi et al. J. Phys. Chem. Lett. (2023) DOI: 10.1021/acs.jpclett.3c02705
        if self.bool_planar_distortions:
            # Axial distortions
            ax_distortions = self.get_axial_distortions(self.structure)
            sign_M, sign_X_bottom, sign_X_top = np.sign(ax_distortions[0]), np.sign(ax_distortions[1]), np.sign(ax_distortions[2])

            axial_distortions = self.get_axial_distortions(structure)
            delta_M, delta_X_bottom, delta_X_top = axial_distortions

            if np.all(sign_M):
                delta_M = sign_M * delta_M
            if np.all(sign_X_bottom):
                delta_X_bottom = sign_X_bottom * delta_X_bottom
            if np.all(sign_X_top):
                delta_X_top = sign_X_top * delta_X_top

            delta_M = np.mean(delta_M)
            delta_X_bottom = np.mean(delta_X_bottom)
            delta_X_top = np.mean(delta_X_top)
            delta_X_ax = (delta_X_bottom + delta_X_top) / 2

            # Equatorial distortions
            equatorial_distortions = self.get_equatorial_distortions(structure)
            delta_X_eq_1, delta_X_eq_2, delta_X_eq_3, delta_X_eq_4, delta_X_eq_5, delta_X_eq_6 = equatorial_distortions

            delta_X_eq_1 = np.mean(delta_X_eq_1)
            delta_X_eq_2 = np.mean(delta_X_eq_2)
            delta_X_eq_3 = np.mean(delta_X_eq_3)
            delta_X_eq_4 = np.mean(delta_X_eq_4)
            delta_X_eq_5 = np.mean(delta_X_eq_5)
            delta_X_eq_6 = np.mean(delta_X_eq_6)

            self.dict_planar_distortions = {
                'delta_M':           delta_M,
                'delta_X_ax':        delta_X_ax,
                'delta_X_ax_bottom': delta_X_bottom,
                'delta_X_ax_top':    delta_X_top,
                'delta_X_eq':        delta_X_eq_2,
                'delta_X_eq_1':      delta_X_eq_1,
                'delta_X_eq_2':      delta_X_eq_2,
                'delta_X_eq_3':      delta_X_eq_3,
                'delta_X_eq_4':      delta_X_eq_4,
                'delta_X_eq_5':      delta_X_eq_5,
                'delta_X_eq_6':      delta_X_eq_6,
            }

        # Hydrogen bonds asymmetry
        if self.bool_hydrogen_bond_asymmetry and not only_inorganic:
            hydrogen_bond_info, hydrogen_bond_asymmetry = self.get_hydrogen_bond_asymmetry(structure)
            
            mean_r_hb, mean_r_bar_hb, mean_delta_r_hb = hydrogen_bond_info
            r_hb_small = np.array([mean_r_hb[0, 0], mean_r_hb[0, 1], mean_r_hb[0, 2]])
            r_bar_hb_small = mean_r_bar_hb[0]
            r_hb_large = np.array([mean_r_hb[1, 0], mean_r_hb[1, 1], mean_r_hb[1, 2]])
            r_bar_hb_large = mean_r_bar_hb[1]

            delta_r_hb = hydrogen_bond_asymmetry
            delta_r_hb = np.mean(delta_r_hb)

            self.dict_hydrogen_bond_asymmetry = {
                'r_hb_small':       r_hb_small,
                'r_bar_hb_small':   r_bar_hb_small,
                'r_hb_large':       r_hb_large,
                'r_bar_hb_large':   r_bar_hb_large,
                'delta_r_hb':       delta_r_hb,
            }

        # Headgroup orientation vectors
        if self.bool_headgroup_orientation and not only_inorganic:
            headgroup_vectors = self.get_headgroup_vectors(structure)

        # Cation orientation vectors
        if self.bool_cation_orientation and not only_inorganic:
            orientation_vectors = self.get_orientation_vectors(structure)

            # Cation helicity
            if not only_inorganic:

                # Cation vector helicity
                if self.bool_cation_helicity:
                    helicity_vectors = np.array(self.helicity_vectors) - 1
                    epsilon_cations = self.get_cation_helicity(structure, self.symmetry_linked_vectors, helicity_vectors, orientation_vectors)

                    self.dict_cation_helicity = {
                        'epsilon_cations': epsilon_cations,
                    }

                # Cation inversion
                if self.bool_cation_inversion:
                    helicity_vectors = np.array(self.helicity_vectors) - 1
                    zeta_cations = self.get_cation_inversion(structure, self.symmetry_linked_vectors, helicity_vectors, orientation_vectors)

                    self.dict_cation_inversion = {
                        'zeta_cations': zeta_cations,
                    }

        # Framework helicity (in-plane)
        if self.bool_planar_framework_helicity:
            epsilon_axial_full, epsilon_diagonal_full = self.get_planar_framework_helicity(structure)
            n_unique_layers = int(self.n_layers / self.n_supercell)

            # Planar axial framework helicity
            epsilon_axial = np.zeros((2, n_unique_layers, 2)) # average helicity per layer for individual helices
            for idx in range(n_unique_layers):
                epsilon_axial[0, idx, :] = epsilon_axial_full[0, idx::n_unique_layers].mean(axis=(0, 2))
                epsilon_axial[1, idx, :] = epsilon_axial_full[1, idx::n_unique_layers].mean(axis=(0, 2))

            # Planar diagonal framework helicity
            epsilon_diagonal = np.zeros((2, n_unique_layers, self.n_diagonal_lines)) # average helicity per layer for individual helices
            for idx in range(n_unique_layers):
                epsilon_diagonal[0, idx, :] = epsilon_diagonal_full[0, idx::n_unique_layers].mean(axis=0)
                epsilon_diagonal[1, idx, :] = epsilon_diagonal_full[1, idx::n_unique_layers].mean(axis=0)

            self.dict_planar_framework_helicity = {
                # Planar axial helicity
                'epsilon_MX4_axial_1':      epsilon_axial[0],
                'epsilon_MX4_axial_2':      epsilon_axial[1],
                # # Planar diagonal helicity
                'epsilon_MX4_diagonal_1':   epsilon_diagonal[0],
                'epsilon_MX4_diagonal_2':   epsilon_diagonal[1],
            }

        # Framework helicity (out-of-plane)
        if self.bool_out_of_plane_framework_helicity:
            epsilon_out_of_plane = self.get_out_of_plane_helicity(structure)
            n_unique_layers = int(self.n_layers / self.n_supercell)

            # Out of plane framework helicity
            epsilon_out_of_plane_layers = np.zeros(n_unique_layers)
            for idx in range(n_unique_layers):
                epsilon_out_of_plane_layers[idx] = np.mean(epsilon_out_of_plane[idx::n_unique_layers])

            self.dict_out_of_plane_framework_helicity = {
                'epsilon_MX4_out_of_plane': epsilon_out_of_plane_layers
            }

        self.__output_summary(output=self.log, only_inorganic=only_inorganic)

    def __output_summary(self, output, only_inorganic):
        """
        Print a summary of the structural distortions to the screen and save to an output file.

        Parameters:
        - output (str): Path to the output file.
        - only_inorganic (bool): If True, only include inorganic descriptors in the summary.

        The function prints and saves a summary of various structural distortions:
        - Intraoctahedral distortions
        - Interoctahedral distortions
        - Planar distortions
        - Hydrogen bond asymmetry
        - Framework helicity
        - Perpendicular framework helicity
        - Headgroup orientation vectors
        - Cation orientation vectors and helicity

        The summary includes relevant information and values for each type of distortion.
        """
        # Auxiliary strings
        major_break = '=' * 48 + '\n'
        minor_break = '-' * 48 + '\n'
        small_break = '- ' * 24 + '\n'

        # ChiraliPy header
        summary_text = (
            major_break +
            '===={:^40}====\n'.format('ChiraliPy') +
            major_break +
            'Path = {:}\n'.format(self.structure_path) +
            major_break + '\n'
        )

        # Function to add a section to the summary
        def add_section(title, content):
            nonlocal summary_text
            summary_text += (
                major_break +
                '{:^48}\n'.format(title) +
                major_break +
                content +
                major_break + '\n'
            )

        # Intraoctahedral distortions
        if self.bool_intraoctahedral_distortions:
            r = self.dict_intraoctahedral_distortions
            add_section('Intraoctahedral distortions', (
                f'sigma2   = {r["sigma2"]:.2f}°^2\n'
                f'delta_d  = {r["delta_d"]:.2f} x 10^-5\n'
            ))

        # Octahedral geometry
        if self.bool_octahedral_geometry:
            r = self.dict_octahedral_geometry
            n_octahedra = len(r["bond_lengths"])

            bond_lengths = 'Bond lengths (Å) | {:^8}{:^8}{:^8}{:^8}{:^8}{:^8}\n'.format('T', 'M1', 'M2', 'M3', 'M4', 'B')
            for idx_octahedron in range(n_octahedra):
                bond_lengths += 'Oct. {:.0f}         = {:>+8.3f}{:>+8.3f}{:>+8.3f}{:>+8.3f}{:>+8.3f}{:>+8.3f}\n'.format(idx_octahedron, *r["bond_lengths"][idx_octahedron])

            bond_angles = 'Bond angles (°) |  {:^8}{:^8}{:^8}{:^8}{:^8}{:^8}{:^8}{:^8}{:^8}{:^8}{:^8}{:^8}\n'.format('T', 'T', 'T', 'T', 'M', 'M', 'M', 'M', 'B', 'B', 'B', 'B')
            for idx_octahedron in range(n_octahedra):
                bond_angles += 'Oct. {:.0f}         = {:>+8.2f}{:>+8.2f}{:>+8.2f}{:>+8.2f}{:>+8.2f}{:>+8.2f}{:>+8.2f}{:>+8.2f}{:>+8.2f}{:>+8.2f}{:>+8.2f}{:>+8.2f}\n'.format(idx_octahedron, *r["bond_angles"][idx_octahedron])

            add_section('Octahedral geometry', (
                f'{bond_lengths}'
                f'{minor_break}'
                f'{bond_angles}'
            ))

        # Interoctahedral distortions
        if self.bool_interoctahedral_distortions:
            r = self.dict_interoctahedral_distortions
            add_section('Interoctahedral distortions', (
                f'delta_beta     = {r["delta_beta"]:.3f}°\n'
                f'delta_beta_in  = {r["delta_beta_in"]:.3f}°\n'
                f'delta_beta_out = {r["delta_beta_out"]:.3f}°\n'
                f'{minor_break}'
                f'max_D          = {r["max_D"]:.3f}°\n'
                f'max_D_in       = {r["max_D_in"]:.3f}°\n'
                f'max_D_out      = {r["max_D_out"]:.3f}°\n'
            ))

        # Planar distortions
        if self.bool_planar_distortions:
            r = self.dict_planar_distortions
            add_section('Planar distortions', (
                f'delta_M            = {r["delta_M"]:.3f} Å\n'
                f'delta_X_eq         = {r["delta_X_eq"]:.3f} Å\n'
                f'delta_X_ax         = {r["delta_X_ax"]:.3f} Å\n'
                f'{minor_break}'
                f'delta_X_ax_bottom  = {r["delta_X_ax_bottom"]:.3f} Å\n'
                f'delta_X_ax_top     = {r["delta_X_ax_top"]:.3f} Å\n'
                f'delta_X_eq_1       = {r["delta_X_eq_1"]:.3f} Å\n'
                f'delta_X_eq_2       = {r["delta_X_eq_2"]:.3f} Å (linear X-M-X angle; DOI: 10.1021/acs.jpclett.3c02705)\n'
                f'delta_X_eq_3       = {r["delta_X_eq_3"]:.3f} Å\n'
                f'delta_X_eq_4       = {r["delta_X_eq_4"]:.3f} Å\n'
                f'delta_X_eq_5       = {r["delta_X_eq_5"]:.3f} Å (equivalent to delta_X_eq_2)\n'
                f'delta_X_eq_6       = {r["delta_X_eq_6"]:.3f} Å (equivalent to delta_X_eq_1)\n'
            ))

        # Hydrogen bonds asymmetry
        if self.bool_hydrogen_bond_asymmetry and not only_inorganic:
            r = self.dict_hydrogen_bond_asymmetry
            add_section('Hydrogen bond asymmetry', (
                f'delta_r_hb       = {r["delta_r_hb"]:.3f} Å\n'
                f'{minor_break}'    
                f'r_bar_hb (small) = {r["r_bar_hb_small"]:.3f} Å\n'
                f'r_hb (small)     = {r["r_hb_small"][0]:.3f} Å (1)\n'
                f'                 = {r["r_hb_small"][1]:.3f} Å (2)\n'
                f'                 = {r["r_hb_small"][2]:.3f} Å (3)\n'
                f'{minor_break}'
                f'r_bar_hb (large) = {r["r_bar_hb_large"]:.3f} Å\n'
                f'r_hb (large)     = {r["r_hb_large"][0]:.3f} Å (1)\n'
                f'                 = {r["r_hb_large"][1]:.3f} Å (2)\n'
                f'                 = {r["r_hb_large"][2]:.3f} Å (3)\n'
            ))

        # Cation helicity
        if self.bool_cation_helicity and not only_inorganic:
            r = self.dict_cation_helicity
            add_section('Cation helicity', (
                f'norm_vector    = ({self.miller[0]:>+5.2f}; {self.miller[1]:>+5.2f}; {self.miller[2]:>+5.2f})\n'
                f'{minor_break}'
                f'epsilon        = {np.mean(r["epsilon_cations"]):>+8.3f} x 10^-3\n'
            ))

        # Cation inversion
        if self.bool_cation_inversion and not only_inorganic:
            r = self.dict_cation_inversion
            add_section('Cation inversion', (
                f'norm_vector    = ({self.miller[0]:>+5.2f}; {self.miller[1]:>+5.2f}; {self.miller[2]:>+5.2f})\n'
                f'{minor_break}'
                f'zeta           = {np.mean(r["zeta_cations"]):>+8.3f}\n'
            ))

        # Framework helicity (in-plane)
        if self.bool_planar_framework_helicity:
            r = self.dict_planar_framework_helicity
            n_unique_layers = len(r["epsilon_MX4_axial_1"])

            # Function to create string decoupling helicity per layer
            def create_detailed_helicity_string(result, result_string, n_unique_layers):
                s = ''
                for idx in range(n_unique_layers):
                    s += small_break
                    s += 'Layer {:.0f}   = {:>+8.3f} x 10^-3\n'.format(idx, np.mean(result[result_string][idx]))
                    for jdx in range(len(result[result_string][idx])):
                        s += '          = {:>+8.3f} x 10^-3 ({:.0f})\n'.format(result[result_string][idx][jdx], jdx)

                return s

            add_section('Planar framework helicity (axial)', (
                f'vector_1  = ({self.directions_axial[0][0]:>+5.2f}; {self.directions_axial[0][1]:>+5.2f}; {self.directions_axial[0][2]:>+5.2f})\n'
                f'epsilon_1 = {np.mean(r["epsilon_MX4_axial_1"]):>+8.3f} x 10^-3\n'
                f'{create_detailed_helicity_string(r, "epsilon_MX4_axial_1", n_unique_layers)}'
                f'{minor_break}'
                f'vector_2  = ({self.directions_axial[1][0]:>+5.2f}; {self.directions_axial[1][1]:>+5.2f}; {self.directions_axial[1][2]:>+5.2f})\n'
                f'epsilon_2 = {np.mean(r["epsilon_MX4_axial_2"]):>+8.3f} x 10^-3\n'
                f'{create_detailed_helicity_string(r, "epsilon_MX4_axial_2", n_unique_layers)}'
            ))

            add_section('Planar framework helicity (diagonal)', (
                f'vector_1  = ({self.directions_diagonal[0][0]:>+5.2f}; {self.directions_diagonal[0][1]:>+5.2f}; {self.directions_diagonal[0][2]:>+5.2f})\n'
                f'epsilon_1 = {np.mean(r["epsilon_MX4_diagonal_1"]):>+8.3f} x 10^-3\n'
                f'{create_detailed_helicity_string(r, "epsilon_MX4_diagonal_1", n_unique_layers)}'
                f'{minor_break}'
                f'vector_2  = ({self.directions_diagonal[1][0]:>+5.2f}; {self.directions_diagonal[1][1]:>+5.2f}; {self.directions_diagonal[1][2]:>+5.2f})\n'
                f'epsilon_2 = {np.mean(r["epsilon_MX4_diagonal_2"]):>+8.3f} x 10^-3\n'
                f'{create_detailed_helicity_string(r, "epsilon_MX4_diagonal_2", n_unique_layers)}'
            ))

        # Framework helicity (out-of-plane)
        if self.bool_out_of_plane_framework_helicity:
            r = self.dict_out_of_plane_framework_helicity
            n_unique_layers = len(r["epsilon_MX4_out_of_plane"])

            epsilon_out_of_plane = ''
            for idx_layer in range(n_unique_layers):
                epsilon_out_of_plane += 'Layer {:.0f}        = {:>+8.3f} x 10^-3\n'.format(idx_layer, np.mean(r["epsilon_MX4_out_of_plane"][idx_layer]))

            add_section('Out-of-plane framework helicity', (
                f'normal_vector  = ({self.miller[0]:>+5.2f}; {self.miller[1]:>+5.2f}; {self.miller[2]:>+5.2f})\n'
                f'epsilon        = {np.mean(r["epsilon_MX4_out_of_plane"]):>+8.3f} x 10^-3\n'
                f'{minor_break}'
                f'{epsilon_out_of_plane}'
            ))

        # Output the summary to screen
        if self.print_log:
            print(summary_text) # output the summary to the screen

        # Output to file
        with open(output, 'w') as f:
            f.write(summary_text)

    def __analyze_trajectory(self):
        """
        Analyze a trajectory using various structural descriptors.

        The function evaluates several structural descriptors based on the specified analysis flags:
        - Intraoctahedral distortions
        - Interoctahedral distortions
        - Planar distortions
        - Hydrogen bond asymmetry
        - Framework helicity (in-plane)
        - Framework helicity (out-of-plane)
        - Headgroup orientation vectors
        - Cation orientation vectors and helicity

        The results of the analysis are stored in class attributes such as `dict_intraoctahedral_distortions`,
        `dict_interoctahedral_distortions`, `dict_planar_distortions`, `dict_hydrogen_bond_asymmetry`,
        `dict_framework_helicity`, `dict_perpendicular_framework_helicity`, and `dict_cation_helicity`.
        """
        results = {} # dictionary to store results

        # Intraoctahedral distortions: Robinson et al. Science (1971) DOI: 10.1126/science.172.3983.567
        if self.bool_intraoctahedral_distortions:
            delta_d = np.zeros((self.n_frames, self.n_octahedra))
            sigma2 = np.zeros(delta_d.shape)

            bond_lengths = np.zeros((self.n_frames, self.n_octahedra, 6))
            bond_angles = np.zeros((self.n_frames, self.n_octahedra, 12))

            for idx, frame in tqdm(enumerate(self.trajectory), total=self.n_frames, desc='Intra. distortions'):
                intraoctahedral_distortions, octahedral_bond_lengths, octahedral_bond_angles = self.get_intraoctahedral_distortions(frame)
                delta_d[idx], sigma2[idx] = intraoctahedral_distortions.T

                bond_lengths[idx] = octahedral_bond_lengths
                bond_angles[idx] = octahedral_bond_angles

            results['intraoctahedral_distortions'] = {
                'delta_d': delta_d,
                'sigma2':  sigma2,
            }

            if self.bool_octahedral_geometry:
                results['octahedral_geometry'] = {
                    'bond_lengths': bond_lengths,
                    'bond_angles': bond_angles,
                }

        # Interoctahedral distortions: Jana et al. Nat. Commun. (2021) DOI: 10.1038/s41467-021-25149-7
        if self.bool_interoctahedral_distortions:
            delta_beta = np.zeros((self.n_frames, self.n_cages))
            delta_beta_in = np.zeros(delta_beta.shape)
            delta_beta_out = np.zeros(delta_beta.shape)

            max_D = np.zeros((self.n_frames, self.n_cages))
            max_D_out = np.zeros(max_D.shape)
            max_D_in = np.zeros(max_D.shape)

            for idx, frame in tqdm(enumerate(self.trajectory), total=self.n_frames, desc='Inter. distortions'):
                interoctahedral_distortions = self.get_interoctahedral_distortions(frame)
                delta_beta[idx], delta_beta_in[idx], delta_beta_out[idx], max_D[idx], max_D_in[idx], max_D_out[idx] = interoctahedral_distortions.T

            results['interoctahedral_distortions'] = {
                'delta_beta':      delta_beta,
                'delta_beta_in':   delta_beta_in,
                'delta_beta_out':  delta_beta_out,
                'max_D':           max_D,
                'max_D_in':        max_D_in,
                'max_D_out':       max_D_out,
            }

        # Planar distortions: Apergi et al. J. Phys. Chem. Lett. (2023) DOI: 10.1021/acs.jpclett.3c02705
        if self.bool_planar_distortions:
            # Axial distortions
            ax_distortions = self.get_axial_distortions(self.structure)
            sign_M, sign_X_bottom, sign_X_top = np.sign(ax_distortions[0]), np.sign(ax_distortions[1]), np.sign(ax_distortions[2])

            delta_M = np.zeros((self.n_frames, self.n_layers))
            delta_X_bottom = np.zeros(delta_M.shape)
            delta_X_top = np.zeros(delta_M.shape)

            # Equatorial distortions
            delta_X_eq_1 = np.zeros((self.n_frames, self.n_layers, 2, int(self.n_octahedral_lattice / self.n_layers / 2)))
            delta_X_eq_2 = np.zeros(delta_X_eq_1.shape)
            delta_X_eq_3 = np.zeros(delta_X_eq_1.shape)
            delta_X_eq_4 = np.zeros(delta_X_eq_1.shape)
            delta_X_eq_5 = np.zeros(delta_X_eq_1.shape)
            delta_X_eq_6 = np.zeros(delta_X_eq_1.shape)

            for idx, frame in tqdm(enumerate(self.trajectory), total=self.n_frames, desc='Planar distortions'):
                # Axial distortions
                axial_distortions = self.get_axial_distortions(frame)
                delta_M[idx], delta_X_bottom[idx], delta_X_top[idx] = axial_distortions

                if np.all(sign_M):
                    delta_M[idx] = sign_M * delta_M[idx]
                if np.all(sign_X_bottom):
                    delta_X_bottom[idx] = sign_X_bottom * delta_X_bottom[idx]
                if np.all(sign_X_top):
                    delta_X_top[idx] = sign_X_top * delta_X_top[idx]

                # Equatorial distortions
                equatorial_distortions = self.get_equatorial_distortions(frame)
                delta_X_eq_1[idx], delta_X_eq_2[idx], delta_X_eq_3[idx], delta_X_eq_4[idx], delta_X_eq_5[idx], delta_X_eq_6[idx] = equatorial_distortions

            results['planar_distortions'] = {
                'delta_M':         delta_M,
                'delta_X_bottom':  delta_X_bottom,
                'delta_X_top':     delta_X_top,
                'delta_X_eq':      delta_X_eq_2,
                'delta_X_eq_1':    delta_X_eq_1,
                'delta_X_eq_2':    delta_X_eq_2,
                'delta_X_eq_3':    delta_X_eq_3,
                'delta_X_eq_4':    delta_X_eq_4,
                'delta_X_eq_5':    delta_X_eq_5,
                'delta_X_eq_6':    delta_X_eq_6,
            }

        # Hydrogen bonds asymmetry
        if self.bool_hydrogen_bond_asymmetry:
            delta_r_hb = np.zeros((self.n_frames, self.n_cages))

            for idx, frame in tqdm(enumerate(self.trajectory), total=self.n_frames, desc='Hydrogen bond asymmetry'):
                _, hydrogen_bond_asymmetry = self.get_hydrogen_bond_asymmetry(frame) # unused: hydrogen_bond_info
                delta_r_hb[idx] = hydrogen_bond_asymmetry

            results['hydrogen_bonds'] = {
                'delta_r_hb':   delta_r_hb,
            }

        # Headgroup orientation vectors
        if self.bool_headgroup_orientation:
            headgroup_vectors = np.zeros((self.n_frames, self.n_headgroup_vectors, self.n_molecules, 3))

            for idx, frame in tqdm(enumerate(self.trajectory), total=self.n_frames, desc='Headgroup vectors'):
                headgroup_vectors[idx] = self.get_headgroup_vectors(frame)

            results['headgroup_orientation'] = {
                'headgroup_vectors': headgroup_vectors,
            }

        # Cation orientation vectors
        if self.bool_cation_orientation:
            orientation_vectors = np.zeros((self.n_frames, self.n_vectors, self.n_molecules, 3))

            for idx, frame in tqdm(enumerate(self.trajectory), total=self.n_frames, desc='Orientation vectors'):
                orientation_vectors[idx] = self.get_orientation_vectors(frame)

            results['cation_orientation'] = {
                'orientation_vectors': orientation_vectors,
            }

            # Cation helicity
            if self.bool_cation_helicity:
                helicity_vectors = np.array(self.helicity_vectors) - 1
                epsilon_cations = np.zeros((self.n_frames, self.n_layers, int(self.n_octahedra / self.n_layers / 2)))

                # Computation of helicity per inorganic layer
                epsilon_cations_averaged = epsilon_cations.mean(axis=-1)

                for idx, frame in tqdm(enumerate(self.trajectory), total=self.n_frames, desc='Cation helicity'):
                    epsilon_cations[idx] = self.get_cation_helicity(frame, self.symmetry_linked_vectors, helicity_vectors, orientation_vectors[idx])

                results['cation_helicity'] = {
                    'epsilon_cations': epsilon_cations,
                    'epsilon_cations_averaged': epsilon_cations_averaged,
                }

            # Cation inversion
            if self.bool_cation_inversion:
                helicity_vectors = np.array(self.helicity_vectors) - 1
                zeta_cations = np.zeros((self.n_frames, self.n_layers, int(self.n_octahedra / self.n_layers)))

                for idx, frame in tqdm(enumerate(self.trajectory), total=self.n_frames, desc='Cation inversion'):
                    zeta_cations[idx] = self.get_cation_inversion(frame, self.symmetry_linked_vectors, helicity_vectors, orientation_vectors[idx]).reshape(self.n_layers, int(self.n_octahedra / self.n_layers))

                results['cation_inversion'] = {
                    'zeta_cations': zeta_cations,
                }

        # Framework helicity (in-plane)
        if self.bool_planar_framework_helicity:
            epsilon_axial = np.zeros((self.n_frames, len(self.directions_axial), self.n_layers, 2, self.n_unique_axial_lines))
            epsilon_diagonal = np.zeros((self.n_frames, len(self.directions_diagonal), self.n_layers, self.n_diagonal_lines))

            for idx, frame in tqdm(enumerate(self.trajectory), total=self.n_frames, desc='Planar framework helicity'):
                epsilon_axial[idx], epsilon_diagonal[idx] = self.get_planar_framework_helicity(frame)

            # Computation of helicity per inorganic layer
            epsilon_axial_averaged = epsilon_axial.mean(axis=(-2, -1))
            epsilon_diagonal_averaged = epsilon_diagonal.mean(axis=-1)

            results['planar_framework_helicity'] = {
                'epsilon_axial_1': epsilon_axial[:, 0],
                'epsilon_axial_2': epsilon_axial[:, 1],
                'epsilon_diagonal_1': epsilon_diagonal[:, 0],
                'epsilon_diagonal_2': epsilon_diagonal[:, 1],
                'epsilon_axial_averaged_1': epsilon_axial_averaged[:, 0],
                'epsilon_axial_averaged_2': epsilon_axial_averaged[:, 1],
                'epsilon_diagonal_averaged_1': epsilon_diagonal_averaged[:, 0],
                'epsilon_diagonal_averaged_2': epsilon_diagonal_averaged[:, 1],
            }

        # Framework helicity (out-of-plane)
        if self.bool_out_of_plane_framework_helicity:
            epsilon_out_of_plane = np.zeros((self.n_frames, self.n_layers, int(self.n_octahedra / self.n_layers / 2)))

            for idx, frame in tqdm(enumerate(self.trajectory), total=self.n_frames, desc='Out-of-plane framework helicity'):
                epsilon_out_of_plane[idx] = self.get_out_of_plane_framework_helicity(frame)

            # Computation of helicity per inorganic layer
            epsilon_out_of_plane_averaged = epsilon_out_of_plane.mean(axis=-1)

            results['out_of_plane_framework_helicity'] = {
                'epsilon_out_of_plane': epsilon_out_of_plane,
                'epsilon_out_of_plane_averaged': epsilon_out_of_plane_averaged,
            }

        self.__save_arrays(results)

    def __create_folder(self, folder):
        """
        Create a folder.

        If the folder already exists, remove it along with its contents.
        
        Parameters:
        - folder (str): The path to the folder to be created or replaced.

        Raises:
        - OSError: If there is an issue with creating or removing the folder.
        """
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.mkdir(folder)

    def __array_to_pickle(self, folder, name, array):
        """
        Save a NumPy array to a .pickle file.

        Parameters:
        - folder (str): The path to the directory where the .pickle file will be saved.
        - name (str): The base name of the .pickle file (excluding extension).
        - array (numpy.ndarray): The NumPy array to be saved.

        Raises:
        - OSError: If there is an issue with creating or writing to the .pickle file.
        - TypeError: If the 'array' parameter is not a NumPy array.

        Note:
        The .pickle file will be saved with the specified name in the provided folder.
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("The 'array' parameter must be a NumPy array.")

        file_path = os.path.join(folder, name + '.pickle')

        with open(file_path, 'wb') as file_io:
            pickle.dump(array, file_io)

    def __save_arrays(self, results):
        """
        Save analyzed data to .pickle files.

        This function iterates through different types of analyzed data and saves them to .pickle files in the specified folder.

        Raises:
        - OSError: If there is an issue with creating or writing to the .pickle files.
        """
        self.__create_folder(self.folder)

        for descriptor, data_dict in results.items():
            for name, data in data_dict.items():
                self.__array_to_pickle(self.folder, name, data)

    def get_intraoctahedral_distortions(self, frame):
        """
        Calculate intraoctahedral distortions based on the method described in:
        "https://doi.org/10.1126/science.172.3983.567"

        Parameters:
        - frame (Frame): The frame containing the atomic coordinates.

        Returns:
        - intraoctahedral_distortions (numpy.ndarray): An array containing distortions for each octahedron, where each row represents an octahedron,
                                                       and columns represent bond length distortion and bond angle variance.
        - octahedral_bond_lengths (numpy.ndarray): An array containing bond lengths for each octahedron, where each row represents an octahedron,
                                                       and columns represent different bond lengths. Sorted from bottom -> top M-X bonds.
        - octahedral_bond_angles (numpy.ndarray): An array containing bond angles for each octahedron, where each row represents an octahedron,
                                                       and columns represent different bond angles. Sorted from bottom -> top X-M-X angles.
        """
        octahedral_centers = self.octahedral_centers.flatten()
        octahedral_corners = self.octahedral_corners.reshape(-1, self.octahedral_corners.shape[-1])
        octahedral_angle_indices = self.octahedral_angle_indices

        intraoctahedral_distortions = np.zeros((len(octahedral_centers), 2)) # delta_d; sigma^2
        octahedral_bond_lengths = np.zeros((len(octahedral_centers), 6)) # 6 M-X bonds lengths per octahedron
        octahedral_bond_angles = np.zeros((len(octahedral_centers), 12)) # 12 X-M-X bond angles per octahedron
        for idx, (center, corners, angle_indices) in enumerate(zip(octahedral_centers, octahedral_corners, octahedral_angle_indices)):
            bond_length_distortion, bond_lengths = self.bond_length_distortion(frame, center, corners)
            bond_angle_variance, bond_angles = self.bond_angle_variance(frame, angle_indices)
            
            intraoctahedral_distortions[idx] = bond_length_distortion, bond_angle_variance
            octahedral_bond_lengths[idx] = bond_lengths
            octahedral_bond_angles[idx] = bond_angles

        return intraoctahedral_distortions, octahedral_bond_lengths, octahedral_bond_angles

    def bond_length_distortion(self, frame, center, corners):
        """
        Calculate the bond length distortion of an octahedron.

        The bond length distortion is defined as: delta_d = 1/6 * SUM[(d_i - d_0) / d_0]^2,
        where d_i is the distance between the octahedral center and each corner,
        and d_0 is the mean distance between the center and all corners.

        Parameters:
        - frame (Frame): The molecular frame containing atomic coordinates.
        - center (numpy.ndarray): Index of the octahedral center.
        - corners (numpy.ndarray): Indices of the octahedral corners.

        Returns:
        - delta_d (float): The bond length distortion value.
        - bond_lengths (numpy.ndarray): The bond lengths around the octahedral center.
        """
        distances = frame.get_distances(center, corners, mic=True)
        d0 = np.mean(distances)

        delta_d = np.sum((distances - d0) ** 2 / d0 ** 2) / len(corners) / 1E-5
        bond_lengths = distances

        return delta_d, bond_lengths

    def bond_angle_variance(self, frame, angle_indices):
        """
        Calculate the bond angle variance of an octahedron.

        The bond angle variance is defined as: sigma2 =  SUM[(theta_i - 90)^2 / 11],
        where theta_i is the angle between the octahedral center and each corner,
        and the sum is taken over all possible angles.

        Parameters:
        - frame (Frame): The molecular frame containing atomic coordinates.
        - angle_indices (numpy.ndarray): Indices of the X-M-X angles in octahedra.

        Returns:
        - sigma2 (float): The bond angle variance value.
        - bond_angles (numpy.ndarray): The bond angles around the octahedral center.
        """
        angles = frame.get_angles(angle_indices, mic=True)
        sigma2 = np.sum((angles - 90.) ** 2) / 11.0 # divide by 11
        bond_angles = angles

        return sigma2, bond_angles

    def get_interoctahedral_distortions(self, frame):
        """
        Calculate interoctahedral distortions based on the method described in:
        "https://doi.org/10.1038/s41467-021-25149-7"

        Parameters:
        - frame (Frame): The molecular frame containing atomic coordinates.

        Returns:
        - interoctahedral_distortions (numpy.ndarray): An array containing interoctahedral distortions for each set of cage sides.
                        Each row represents a set of cage sides, and columns represent different distortions.
        """
        all_cage_sides = self.all_cage_sides

        delta_beta, delta_beta_in, delta_beta_out = np.zeros(all_cage_sides.shape[0]), np.zeros(all_cage_sides.shape[0]), np.zeros(all_cage_sides.shape[0])
        max_D, max_D_in, max_D_out = np.zeros(all_cage_sides.shape[0]), np.zeros(all_cage_sides.shape[0]), np.zeros(all_cage_sides.shape[0])

        interoctahedral_distortions = np.zeros((all_cage_sides.shape[0], 6))
        for idx, cage_sides in enumerate(all_cage_sides):
            interoctahedral_distortions[idx] = self.get_cage_distortion(frame, cage_sides)

        return interoctahedral_distortions

    def get_cage_distortion(self, frame, sides):
        """
        Calculate cage distortions:
        1) delta_beta
        2) delta_beta_in (in-plane)
        3) delta_beta_out (out-of-plane)
        4) max_D
        5) max_D_in (in-plane)
        6) max_D_out (out-of-plane)

        Parameters:
        - frame (Frame): The molecular frame containing atomic coordinates.
        - sides (numpy.ndarray): Coordinates of the octahedral corners.

        Returns:
        - Tuple: A tuple of floats containing the calculated distortions.
            1. delta_beta (float)
            2. delta_beta_in (float)
            3. delta_beta_out (float)
            4. max_D (float)
            5. max_D_in (float)
            6. max_D_out (float)
        """
        normal_vector = self.get_normal_vector(frame)

        N_sides = sides.shape[0]
        corners1, centers, corners2 = sides.T

        L1, L2, L3 = np.zeros(N_sides), np.zeros(N_sides), np.zeros(N_sides)
        d, x = np.zeros(N_sides), np.zeros(N_sides)
        for idx, (corner1, center, corner2) in enumerate(zip(corners1, centers, corners2)):
            L1[idx] = frame.get_distance(corner1,  center, mic=True)
            L2[idx] = frame.get_distance(corner2,  center, mic=True)
            L3[idx] = frame.get_distance(corner1, corner2, mic=True)

            r_comp_corner1 = np.dot(frame.get_positions()[corner1], normal_vector)
            r_comp_corner2 = np.dot(frame.get_positions()[corner2], normal_vector)
            r_comp_center  = np.dot(frame.get_positions()[ center], normal_vector)
            
            d[idx] = r_comp_corner1 - r_comp_corner2
            x[idx] = r_comp_center - (r_comp_corner1 + r_comp_corner2) / 2

        # In-plane angle: beta_in
        a1 = np.sqrt(L1 ** 2 - (x - d / 2) ** 2)
        a2 = np.sqrt(L2 ** 2 - (x + d / 2) ** 2)
        a3 = np.sqrt(L3 ** 2 - d ** 2)
        cos_beta_in = np.clip((a1 ** 2 + a2 ** 2 - a3 ** 2) / (2 * a1 * a2), -1.0, 1.0)
        beta_in = np.arccos(cos_beta_in) * 180 / np.pi
        beta_in_S, beta_in_L = np.mean(beta_in[:2]), np.mean(beta_in[2:])
        delta_beta_in = beta_in_L - beta_in_S
        max_D_in = 180 - np.min(beta_in)

        # Out-of-plane angle: beta_out
        c = (a1 ** 2 + a3 ** 2 - a2 ** 2) / (2 * a3)
        cos_beta_out = np.clip((c ** 2 + x ** 2 - a3 * c) / (np.sqrt(c ** 2 + x ** 2) * np.sqrt((a3 - c) ** 2 + x ** 2)), -1.0, 1.0)
        beta_out = np.arccos(cos_beta_out) * 180 / np.pi
        beta_out_S, beta_out_L = np.mean(beta_out[:2]), np.mean(beta_out[2:])
        delta_beta_out = beta_out_L - beta_out_S
        max_D_out = 180 - np.min(beta_out)

        # Angle: beta
        beta = frame.get_angles(sides, mic=True)
        beta_S, beta_L = np.mean(beta[:2]), np.mean(beta[2:])
        delta_beta = beta_L - beta_S
        max_D = 180 - np.min(beta)

        return delta_beta, delta_beta_in, delta_beta_out, max_D, max_D_in, max_D_out

    def get_axial_distortions(self, frame):
        """
        Calculate axial planar distortions based on the method described in:
        "https://doi.org/10.1021/acs.jpclett.3c02705"

        Parameters:
        - frame (Frame): The molecular frame containing atomic coordinates.

        Returns:
        - axial_distortions (numpy.ndarray): An array containing axial planar distortions.
        """
        axial_indices = self.axial_indices
        normal_vector = self.get_normal_vector(frame)

        positions      = frame.get_positions()[axial_indices]
        components     = np.sum(positions * normal_vector, axis=-1) # normal components to layer

        delta_M        = np.squeeze(np.diff(np.mean(components[:, :, :, 0], axis=2), axis=1))
        delta_X_bottom = np.squeeze(np.diff(np.mean(components[:, :, :, 1], axis=2), axis=1))
        delta_X_top    = np.squeeze(np.diff(np.mean(components[:, :, :, 2], axis=2), axis=1))

        axial_distortions = np.zeros((3, self.n_layers))
        axial_distortions[0] = delta_M
        axial_distortions[1] = delta_X_bottom
        axial_distortions[2] = delta_X_top

        return axial_distortions

    def get_equatorial_distortions(self, frame):
        """
        Calculate equatorial planar distortions based on the method described in:
        "https://doi.org/10.1021/acs.jpclett.3c02705"

        Parameters:
        - frame (Frame): The molecular frame containing atomic coordinates.

        Returns:
        - equatorial_distortions (numpy.ndarray): An array containing equatorial planar distortions.
        """
        equatorial_indices = self.equatorial_indices
        normal_vector = self.get_normal_vector(frame)

        positions      = frame.get_positions()[equatorial_indices]
        components     = np.sum(positions * normal_vector, axis=-1) # normal components to layer

        r_X_1 = components[:, :, :, 0]
        r_X_2 = components[:, :, :, 1]
        r_X_3 = components[:, :, :, 2]
        r_X_4 = components[:, :, :, 3]
        r_X = [r_X_1, r_X_2, r_X_3, r_X_4]

        # Initialize equatorial pairs
        n_X_eq, n_pairs = 4, 6
        equatorial_pairs = np.array([(i, j) for i in range(n_X_eq) for j in range(i + 1, n_X_eq)])

        # Calculate equatorial distortions
        equatorial_distortions = np.zeros((n_pairs, self.n_layers, 2, int(self.n_octahedral_lattice / self.n_layers / 2)))
        for idx, (idx1, idx2) in enumerate(equatorial_pairs):
            equatorial_distortions[idx] = r_X[idx2] - r_X[idx1]
        
        return equatorial_distortions

    def get_hydrogen_bond_asymmetry(self, frame):
        """
        Calculate hydrogen bond asymmetry above and below the inorganic planes.

        Parameters:
        - frame (Frame): The molecular frame containing atomic coordinates.

        Returns:
        - Tuple: A tuple containing hydrogen bond information and asymmetry values.
            1. hydrogen_bond_info (list): decomposed hydrogen bond information
            2. hydrogen_bond_asymmetry (float): hydrogen bond asymmetry
        """
        hydrogen_bond_indices = self.all_hydrogen_bond_indices
        hb_shape = hydrogen_bond_indices.shape

        x = np.array([atom.index for atom in self.structure if atom.symbol in self.corner_elements])

        r_hb, sorted_r_hb = np.zeros(hb_shape), np.zeros(hb_shape)
        for idx_cage in range(hb_shape[0]):
            for idx_cation in range(hb_shape[1]):
                for idx_hb in range(hb_shape[2]):
                    h = hydrogen_bond_indices[idx_cage, idx_cation, idx_hb]
                    r_hb[idx_cage, idx_cation, idx_hb] = np.min(frame.get_distances(h, x, mic=True))
                sorted_r_hb[idx_cage, idx_cation, :] = r_hb[idx_cage, idx_cation, :][r_hb[idx_cage, idx_cation, :].argsort()]

        r_bar_hb = np.mean(sorted_r_hb[:, :, :-1], axis=2)
        delta_r_hb = np.diff(r_bar_hb, axis=1)
        hydrogen_bond_asymmetry = np.squeeze(delta_r_hb)

        mean_r_hb = np.mean(sorted_r_hb, axis=0) # unique bonds
        mean_r_bar_hb = np.mean(r_bar_hb, axis=0) # mean average bonds
        mean_delta_r_hb = np.mean(delta_r_hb) # mean bond difference
        hydrogen_bond_info = [mean_r_hb, mean_r_bar_hb, mean_delta_r_hb]

        return hydrogen_bond_info, hydrogen_bond_asymmetry

    def get_planar_framework_helicity(self, frame):
        """
        Calculate the planar helicity of the inorganic framework along two directions.

        1. Axial direction in the plane of the inorganic framework.
        2. Diagonal direction in the plane of the inorganic framework.

        Parameters:
        - frame (Frame): The frame containing the atomic coordinates.

        Returns:
        - epsilon_axial (numpy.ndarray): Array containing axial planar helicities based on vector chirality.
        - epsilon_diagonal (numpy.ndarray): Array containing diagonal planar helicities based on vector chirality.
        """
        epsilon_axial = self.get_axial_framework_helicity(frame)
        epsilon_diagonal = self.get_diagonal_framework_helicity(frame)

        return epsilon_axial, epsilon_diagonal

    def get_axial_framework_helicity(self, frame):
        """
        Calculate the planar axial helicity of the inorganic framework using symmetry equivalent octahedra.

        Parameters:
        - frame (Frame): The frame containing the atomic coordinates.

        Returns:
        - epsilon_p (numpy.ndarray): Array containing out-of-plane helicities based on vector chirality.
        """
        axial_line_indices = self.axial_line_indices
        directions_axial = self.directions_axial

        epsilon = np.zeros((len(self.directions_axial), self.n_layers, 2, self.n_unique_axial_lines, 3))
        epsilon_p = np.zeros((len(self.directions_axial), self.n_layers, 2, self.n_unique_axial_lines))

        for idx, (indices, d) in enumerate(zip(axial_line_indices, directions_axial)):
            d = frame.get_cell().cartesian_positions(d)
            d = d / np.linalg.norm(d)

            for jdx, layer in enumerate(indices):
                for kdx, helix in enumerate(layer):
                    for ldx, line in enumerate(helix):
                        delta_r = np.zeros((self.len_axial_lines, 3))
                        for mdx in range(self.len_axial_lines):
                            delta_r[mdx, :] = frame.get_distance(line[(mdx) % self.len_axial_lines], line[(mdx + 1) % self.len_axial_lines], mic=True, vector=True)

                        epsilon[idx, jdx, kdx, ldx, :], epsilon_p[idx, jdx, kdx, ldx] = self.compute_vector_chirality(delta_r, self.len_axial_lines, d, normalize=True)

        return epsilon_p # projected helicity along inorganic lines; [direction 1, direction 2]

    def get_diagonal_framework_helicity(self, frame):
        """
        Calculate the planar diagonal helicity of the inorganic framework using symmetry equivalent octahedra.

        Parameters:
        - frame (Frame): The frame containing the atomic coordinates.

        Returns:
        - epsilon_p (numpy.ndarray): Array containing out-of-plane helicities based on vector chirality.
        """
        diagonal_line_indices = self.diagonal_line_indices
        directions_diagonal = self.directions_diagonal

        epsilon = np.zeros((len(self.directions_diagonal), self.n_layers, self.n_diagonal_lines, 3))
        epsilon_p = np.zeros((len(self.directions_diagonal), self.n_layers, self.n_diagonal_lines))

        for idx, (indices, d) in enumerate(zip(diagonal_line_indices, directions_diagonal)):
            d = frame.get_cell().cartesian_positions(d)
            d = d / np.linalg.norm(d)

            for jdx, layer in enumerate(indices):
                for kdx, line in enumerate(layer):
                    delta_r = np.zeros((self.len_diagonal_lines, 3))
                    for ldx in range(self.len_diagonal_lines):
                        delta_r[ldx, :] = frame.get_distance(line[(ldx) % self.len_diagonal_lines], line[(ldx + 1) % self.len_diagonal_lines], mic=True, vector=True)

                    epsilon[idx, jdx, kdx, :], epsilon_p[idx, jdx, kdx] = self.compute_vector_chirality(delta_r, self.len_diagonal_lines, d, normalize=True)

        return epsilon_p # projected helicity along inorganic lines; [direction 1, direction 2]

    def get_out_of_plane_helicity(self, frame):
        """
        Calculate the out-of-plane helicity of the inorganic framework using symmetry equivalent octahedra.

        Parameters:
        - frame (Frame): The frame containing the atomic coordinates.

        Returns:
        - epsilon_out_of_plane (numpy.ndarray): Array containing out-of-plane helicities based on vector chirality.
        """
        epsilon_out_of_plane = self.get_out_of_plane_framework_helicity(frame)

        return epsilon_out_of_plane

    def get_out_of_plane_framework_helicity(self, frame):
        """
        Calculate the out-of-plane helicity of the inorganic framework using symmetry equivalent octahedra.

        Parameters:
        - frame (Frame): The frame containing the atomic coordinates.

        Returns:
        - epsilon_p (numpy.ndarray): Array containing out-of-plane helicities based on vector chirality.
        """
        out_of_plane_line_indices = self.out_of_plane_line_indices
        normal_vector = self.normal_vector

        n_layers, n_lines_layer, len_lines, _ = out_of_plane_line_indices.shape

        epsilon = np.zeros((n_layers, n_lines_layer, 3))
        epsilon_p = np.zeros((n_layers, n_lines_layer))

        for idx, layer in enumerate(out_of_plane_line_indices):
            for jdx, line in enumerate(layer):
                delta_r = np.zeros((len_lines, 3))
                for kdx, (atom1, atom2) in enumerate(line):
                    delta_r[kdx, :] = frame.get_distance(atom1, atom2, mic=True, vector=True)

                epsilon[idx, jdx, :], epsilon_p[idx, jdx] = self.compute_vector_chirality(delta_r, len_lines, normal_vector, normalize=True)

        return epsilon_p # projected helicity in out-of-plane direction

    def get_cation_helicity(self, frame, cation_indices, helicity_vectors, orientation_vectors):
        """
        Determine the helicity of cations using vector chirality.

        Parameters:
        - frame (int): The frame containing the atomic coordinates.
        - cation_indices (numpy.ndarray): Array containing indices of cations for which helicity is determined.
        - helicity_vectors (slice or list): Index or list of indices specifying orientation vectors for helicity computation.
        - orientation_vectors (numpy.ndarray): Array containing orientation vectors of cations.

        Returns:
        - epsilon_p (numpy.ndarray): Array containing cation helicities based on vector chirality.
        """
        selected_orientation_vectors = orientation_vectors[helicity_vectors]
        normal_vector = self.normal_vector

        # For two vectors the cross product of vectors is used in computation of helicity
        if len(helicity_vectors) == 2:
            v1, v2 = selected_orientation_vectors
            selected_orientation_vectors = np.cross(v1, v2)
        else:
            selected_orientation_vectors = selected_orientation_vectors[0]

        n_layers, n_groups_layer, len_groups = cation_indices.shape

        epsilon = np.zeros((n_layers, n_groups_layer, 3))
        epsilon_p = np.zeros((n_layers, n_groups_layer))

        for idx, layer in enumerate(cation_indices):
            for jdx, group in enumerate(layer):
                delta_r = selected_orientation_vectors[group]

                projected_delta_r = np.sum(delta_r * normal_vector, axis=1) # projection of distance vectors
                sign_projected_delta_r = np.tile(np.sign(projected_delta_r)[np.newaxis].T, (1, 3))
                delta_r = delta_r * sign_projected_delta_r

                epsilon[idx, jdx, :], epsilon_p[idx, jdx] = self.compute_vector_chirality(delta_r, len_groups, normal_vector, normalize=True)

        return epsilon_p

    def get_cation_inversion(self, frame, cation_indices, helicity_vectors, orientation_vectors):
        """
        Determine the inversion of cations using vector orientations.

        Parameters:
        - frame (int): The frame containing the atomic coordinates.
        - cation_indices (numpy.ndarray): Array containing indices of cations for which helicity is determined.
        - helicity_vectors (slice or list): Index or list of indices specifying orientation vectors for helicity computation.
        - orientation_vectors (numpy.ndarray): Array containing orientation vectors of cations.

        Returns:
        - zeta_p (numpy.ndarray): Array containing cation collinearity as measure of inversion symmetry.
        """
        selected_orientation_vectors = orientation_vectors[helicity_vectors]
        normal_vector = self.normal_vector

        # For two vectors the cross product of vectors is used in computation of helicity
        if len(helicity_vectors) == 2:
            v1, v2 = selected_orientation_vectors
            selected_orientation_vectors = np.cross(v1, v2)
        else:
            selected_orientation_vectors = selected_orientation_vectors[0]

        n_layers, n_groups_layer, len_groups = cation_indices.shape

        zeta = np.zeros((n_layers, n_groups_layer, 2, 3))
        zeta_p = np.zeros((n_layers, n_groups_layer, 2))

        for idx, layer in enumerate(cation_indices):
            for jdx, group in enumerate(layer):
                delta_r = selected_orientation_vectors[group]
                
                projected_delta_r = np.sum(delta_r * normal_vector, axis=1) # projection of distance vectors
                sign_projected_delta_r = np.tile(np.sign(projected_delta_r)[np.newaxis].T, (1, 3))
                delta_r = delta_r * sign_projected_delta_r

                v1, v2, v3, v4 = delta_r

                zeta[idx, jdx, :, :] = np.array([
                    np.cross(v1, v2),
                    np.cross(v3, v4)
                ])

                zeta_p[idx, jdx, :] = np.array([
                    np.dot(normal_vector, np.cross(v1, v2)),
                    np.dot(normal_vector, np.cross(v3, v4))
                ])

        return zeta_p

    def compute_vector_chirality(self, vectors, n_vectors, projection_vector, normalize=True):
        """
        Compute the vector chirality from a set of orientation vectors.

        Parameters:
        - vectors (numpy.ndarray): Input orientation vectors.
        - n_vectors (int): Number of unique vectors to consider.
        - projection_vector (numpy.ndarray): Projection vector for chirality projection.
        - normalize (bool, optional): Whether to normalize the input vectors.
                                      Defaults to True.

        Returns:
        - tuple: Tuple containing the total chirality (epsilon) and projected chirality (epsilon_p).
        """
        if normalize:
            vectors = self.normalize_vectors(vectors, axis=1)

        # Tile vectors for pairwise operation
        vectors_tiled = np.tile(vectors, (2 ,1))
        v1, v2 = vectors_tiled[:n_vectors], vectors_tiled[1:n_vectors + 1]

        # Compute total vector chirality (epsilon)
        epsilon = np.sum(np.cross(v1, v2), axis=0)

        # Compute projected vector chirality (epsilon_p)
        epsilon_p = np.dot(projection_vector, epsilon)

        # Normalize the chirality values
        epsilon   /= (n_vectors * 1E-3)
        epsilon_p /= (n_vectors * 1E-3)

        return epsilon, epsilon_p

    def normalize_vectors(self, vectors, axis=1):
        """
        Normalize vectors along a specified axis.

        Parameters:
        - vectors (numpy.ndarray): Input vectors to be normalized.
        - axis (int, optional): Axis along which normalization is performed. Defaults to 1.

        Returns:
        - normalized_vectors (numpy.ndarray): Normalized vectors with the same dimensions.
        """
        normalized_vectors = vectors / np.linalg.norm(vectors, axis=axis, keepdims=True)

        return normalized_vectors

    def atomic_vector(self, frame, at1, at2):
        """
        Calculate the normalized vector between two atoms.

        Parameters:
        - frame (Frame): The molecular frame containing atomic coordinates.
        - at1 (int): Index of the first atom.
        - at2 (int): Index of the second atom.

        Returns:
        - atomic_vector (numpy.ndarray): The normalized vector pointing from at1 to at2.
        """
        r_vector = frame.get_distance(at1, at2, mic=True, vector=True)
        atomic_vector = r_vector / np.linalg.norm(r_vector)

        return atomic_vector

    def get_orientation_vectors(self, frame):
        """
        Calculate orientation vectors of organic moieties using indices from the molecular graph.

        Parameters:
        - frame (Frame): The molecular frame containing atomic coordinates.

        Returns:
        - orientation_vectors (numpy.ndarray): An array containing orientation vectors for organic moieties.
                                               The array has dimensions (n_vectors, n_molecules, 3).
        """
        vector_indices = self.vector_indices
        n_vectors = self.n_vectors
        n_molecules = self.n_molecules

        orientation_vectors = np.zeros((n_vectors, n_molecules, 3))
        for idx, vector_idxs in enumerate(vector_indices):
            for jdx, (at1, at2) in enumerate(vector_idxs):
                orientation_vectors[idx, jdx, :] = self.atomic_vector(frame, at1, at2)

        return orientation_vectors

    def get_headgroup_vectors(self, frame):
        """
        Calculate orientation vectors of headgroups using indices from the molecular graph.

        Parameters:
        - frame (Frame): The molecular frame containing atomic coordinates.

        Returns:
        - headgroup_vectors (numpy.ndarray): An array containing orientation vectors for headgroups.
                                             The array has dimensions (n_headgroup_vectors, n_molecules, 3).
        """
        headgroup_indices = self.headgroup_indices
        n_headgroup_vectors = self.n_headgroup_vectors
        n_molecules = self.n_molecules

        headgroup_vectors = np.zeros((n_headgroup_vectors, n_molecules, 3))
        for idx, (at1, at2, at3, at4) in enumerate(headgroup_indices):
            headgroup_vectors[0, idx, :] = self.atomic_vector(frame, at1, at2)
            headgroup_vectors[1, idx, :] = self.atomic_vector(frame, at1, at3)
            headgroup_vectors[2, idx, :] = self.atomic_vector(frame, at1, at4)

        return headgroup_vectors

def parse_command_line_arguments():
    """
    Parse command line arguments for analyzing the chirality of metal halide perovskite structures or trajectories.

    Returns:
    - args (argparse.Namespace): An object containing parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Analyze the chirality of metal halide perovskite structures or trajectories.")

    parser.add_argument("-s", "--settings", help="Path to the settings json file.", required=True, type=str, default="settings.json")

    return parser.parse_args()

if __name__ == "__main__":
    main()
