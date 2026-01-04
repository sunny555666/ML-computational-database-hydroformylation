import os
import numpy as np
from ase import Atoms
from ase.io import read, write
from dscribe.descriptors import SOAP

# SOAP descriptor parameters (modify according to your chemical system)
soap_params = {
    'species': ['Rh', 'C', 'O', 'P', 'H'],  # Add/modify elements as needed
    'r_cut': 6.0,
    'n_max': 8,
    'l_max': 6,
    'periodic': False
}

# Create SOAP descriptor object
soap = SOAP(**soap_params)

def find_nearest_atoms_by_element(atoms, reference_atoms, target_element, num_atoms=1):
    """
    Find the nearest atoms of a specific element to reference atoms
    
    Parameters:
    atoms: ASE Atoms object
    reference_atoms: List of reference atoms
    target_element: Target element symbol (e.g., 'H', 'C', 'O')
    num_atoms: Number of nearest atoms to find
    
    Returns:
    nearest_atoms: List of nearest atoms of target element
    """
    target_atoms = [atom for atom in atoms if atom.symbol == target_element]
    
    if not target_atoms:
        print(f"No {target_element} atoms found in the structure.")
        return []
    
    # Calculate minimum distance from each target atom to all reference atoms
    atom_distances = []
    for target_atom in target_atoms:
        min_distance = float('inf')
        for ref_atom in reference_atoms:
            distance = np.linalg.norm(target_atom.position - ref_atom.position)
            min_distance = min(min_distance, distance)
        atom_distances.append((target_atom, min_distance))
    
    # Sort by distance and select nearest atoms
    atom_distances.sort(key=lambda x: x[1])
    nearest_atoms = [atom for atom, _ in atom_distances[:num_atoms]]
    
    # Output distance information
    print(f"Found {len(target_atoms)} {target_element} atoms in total")
    print(f"Nearest {num_atoms} {target_element} atom(s) to reference atoms:")
    for i, (atom, distance) in enumerate(atom_distances[:num_atoms]):
        print(f"  {target_element} atom {i+1}: distance = {distance:.3f} Å")
    
    return nearest_atoms

def find_atoms_in_range(atoms, reference_atoms, target_element, exclude_indices=None):
    """
    Find atoms of a specific element, optionally excluding certain indices
    
    Parameters:
    atoms: ASE Atoms object
    reference_atoms: List of reference atoms for distance calculation
    target_element: Target element symbol
    exclude_indices: List of atom indices to exclude (e.g., last N atoms)
    
    Returns:
    nearest_atom: Nearest atom of target element (excluding specified indices)
    """
    if exclude_indices is None:
        exclude_indices = []
    
    # Get atoms to consider (excluding specified indices)
    atoms_to_consider = [atoms[i] for i in range(len(atoms)) if i not in exclude_indices]
    target_atoms = [atom for atom in atoms_to_consider if atom.symbol == target_element]
    
    if not target_atoms:
        print(f"No {target_element} atoms found in the structure (after exclusions).")
        return None
    
    # Calculate minimum distance from each target atom to all reference atoms
    atom_distances = []
    for target_atom in target_atoms:
        min_distance = float('inf')
        for ref_atom in reference_atoms:
            distance = np.linalg.norm(target_atom.position - ref_atom.position)
            min_distance = min(min_distance, distance)
        atom_distances.append((target_atom, min_distance))
    
    # Sort by distance and select nearest atom
    atom_distances.sort(key=lambda x: x[1])
    nearest_atom = atom_distances[0][0]
    
    # Output distance information
    print(f"Found {len(target_atoms)} {target_element} atoms in range")
    print(f"Nearest {target_element} atom: distance = {atom_distances[0][1]:.3f} Å")
    
    return nearest_atom

def find_bonded_atoms(atoms, reference_atom, target_element, bond_threshold=1.8):
    """
    Find atoms of a specific element bonded to a reference atom
    
    Parameters:
    atoms: ASE Atoms object
    reference_atom: Reference atom to find bonds to
    target_element: Target element symbol to search for
    bond_threshold: Distance threshold for bond detection (Å)
    
    Returns:
    bonded_atom: Nearest bonded atom of target element
    """
    target_atoms = [atom for atom in atoms if atom.symbol == target_element]
    
    if not target_atoms:
        print(f"No {target_element} atoms found in the structure.")
        return None
    
    # Calculate distances from target atoms to reference atom
    atom_distances = []
    for target_atom in target_atoms:
        distance = np.linalg.norm(target_atom.position - reference_atom.position)
        if distance <= bond_threshold:
            atom_distances.append((target_atom, distance))
    
    if not atom_distances:
        print(f"No {target_element} atoms found within {bond_threshold} Å of the reference atom.")
        return None
    
    # Sort by distance and select nearest bonded atom
    atom_distances.sort(key=lambda x: x[1])
    bonded_atom = atom_distances[0][0]
    
    # Output bond information
    print(f"Found {len(atom_distances)} {target_element} atom(s) bonded to reference atom")
    print(f"Nearest bonded {target_element} atom: distance = {atom_distances[0][1]:.3f} Å")
    
    return bonded_atom

def get_atoms_by_indices(atoms, indices):
    """
    Get atoms by their indices
    
    Parameters:
    atoms: ASE Atoms object
    indices: List of atom indices
    
    Returns:
    selected_atoms: List of atoms at specified indices
    """
    return [atoms[i] for i in indices if i < len(atoms)]

# Configuration section - modify these variables according to your needs
INPUT_FOLDER = r'path/to/your/input/folder'  # Modify this path
OUTPUT_SUFFIX = '_local_structure'           # Suffix for output files
FILE_EXTENSION = '.xyz'                      # Input file extension

# Atom selection configuration
CENTER_ELEMENTS = ['Rh']                     # Central atoms for distance calculation
ADDITIONAL_ELEMENTS = ['P']                  # Additional atoms to include
TARGET_ELEMENTS = ['H', 'C']                # Elements to find nearest neighbors
BONDED_ELEMENT = 'O'                        # Element to find bonded to nearest C

# Special atom indices (modify as needed for your system)
SPECIAL_INDICES = [-15, -14, -13, -12]      # Negative indices for atoms from end
EXCLUDE_LAST_N = 15                         # Number of atoms to exclude when finding nearest C

def main():
    """
    Main function to process all files in the target folder
    """
    if not os.path.exists(INPUT_FOLDER):
        print(f"Input folder {INPUT_FOLDER} does not exist!")
        return
    
    # Process all files with specified extension
    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(FILE_EXTENSION):
            file_path = os.path.join(INPUT_FOLDER, filename)
            
            try:
                atoms = read(file_path, format='xyz')
            except Exception as e:
                print(f"Failed to read {filename}: {e}")
                continue
            
            print(f"\nProcessing {filename}...")
            
            # Find center atoms (e.g., Rh)
            center_atoms = [atom for atom in atoms if atom.symbol in CENTER_ELEMENTS]
            if not center_atoms:
                print(f"No center atoms ({CENTER_ELEMENTS}) found in {filename}, skipping.")
                continue
            
            # Find additional atoms (e.g., P)
            additional_atoms = [atom for atom in atoms if atom.symbol in ADDITIONAL_ELEMENTS]
            
            # Create new structure with selected atoms
            selected_atoms = Atoms()
            
            # Add center atoms
            for atom in center_atoms:
                selected_atoms.append(atom)
            
            # Add additional atoms
            for atom in additional_atoms:
                selected_atoms.append(atom)
            
            # Find and add nearest atoms of target elements
            for element in TARGET_ELEMENTS:
                if element == 'H':
                    # Find nearest H atoms
                    nearest_atoms = find_nearest_atoms_by_element(atoms, center_atoms, element, num_atoms=1)
                    for atom in nearest_atoms:
                        selected_atoms.append(atom)
                
                elif element == 'C':
                    # Find nearest C atom (excluding last N atoms)
                    exclude_indices = list(range(len(atoms) - EXCLUDE_LAST_N, len(atoms)))
                    nearest_atom = find_atoms_in_range(atoms, center_atoms, element, exclude_indices)
                    if nearest_atom:
                        selected_atoms.append(nearest_atom)
                        
                        # Find bonded O atom to this C
                        bonded_o = find_bonded_atoms(atoms, nearest_atom, BONDED_ELEMENT, bond_threshold=1.8)
                        if bonded_o:
                            selected_atoms.append(bonded_o)
            
            # Add special atoms by indices (e.g., last few atoms)
            total_atoms = len(atoms)
            special_atoms = []
            for idx in SPECIAL_INDICES:
                actual_idx = total_atoms + idx if idx < 0 else idx
                if 0 <= actual_idx < total_atoms:
                    special_atoms.append(atoms[actual_idx])
            
            for atom in special_atoms:
                selected_atoms.append(atom)
            
            # Output structure information
            print(f"New structure created from {filename}:")
            print(f"Total atoms in new structure: {len(selected_atoms)}")
            for i, atom in enumerate(selected_atoms):
                print(f"Atom {i}: {atom.symbol} at {atom.position}")
            
            # Save new structure as XYZ file
            base_name = os.path.splitext(filename)[0]
            xyz_filename = f"{base_name}{OUTPUT_SUFFIX}.xyz"
            xyz_path = os.path.join(INPUT_FOLDER, xyz_filename)
            write(xyz_path, selected_atoms)
            print(f"Structure saved to {xyz_path}")
            
            # Calculate SOAP descriptors
            try:
                soap_descriptors = soap.create(selected_atoms)
                
                print(f"SOAP descriptors calculated:")
                print(f"Shape: {soap_descriptors.shape}")
                
                # Save SOAP descriptors
                npy_filename = f"{base_name}{OUTPUT_SUFFIX}_soap.npy"
                npy_path = os.path.join(INPUT_FOLDER, npy_filename)
                np.save(npy_path, soap_descriptors)
                
                csv_filename = f"{base_name}{OUTPUT_SUFFIX}_soap.csv"
                csv_path = os.path.join(INPUT_FOLDER, csv_filename)
                np.savetxt(csv_path, soap_descriptors, delimiter=',', fmt='%s')
                
                print(f"SOAP descriptors saved to {csv_path}")
                
            except Exception as e:
                print(f"Failed to calculate SOAP descriptors for {filename}: {e}")
            
            print("-" * 50)

if __name__ == "__main__":
    main()