# ORCA Molecular Feature Extraction Script
# This script extracts 13 key structural and electronic features from ORCA output files
# for organometallic complexes containing Rh-H bonds
#
# Features extracted:
# 1. NMR shielding anisotropy values (Rh, H, C3)
# 2. NMR XZ tensor component (C1) 
# 3. NPA charges (H atom)
# 4. Structural parameters (distances, angles)
# 5. Sterimol descriptors
# 6. Buried volume parameters
# 7. Dispersion interactions
#
# Note: The alkene coordinates are located in the last part of the input geometry files

import os
import pandas as pd
import re
import numpy as np
from ase import Atoms
from ase.io import read
from morfeus import Dispersion, Sterimol, BuriedVolume
import io
import sys
import math

def find_nearest_h_atom(atoms, rh_atom):
    """
    Find the H atom nearest to the Rh atom
    
    Parameters:
    atoms: ASE Atoms object
    rh_atom: Rh atom object
    
    Returns:
    nearest_h_index: Index of nearest H atom (1-based)
    """
    h_atoms = [atom for atom in atoms if atom.symbol == 'H']
    
    if not h_atoms:
        print("No H atoms found in the structure.")
        return None
    
    # Calculate distances between all H atoms and Rh atom
    h_distances = []
    for i, h_atom in enumerate(h_atoms):
        distance = np.linalg.norm(h_atom.position - rh_atom.position)
        # Find the real index of this H atom in the total atom list
        real_index = None
        for j, atom in enumerate(atoms):
            if atom.symbol == 'H' and np.allclose(atom.position, h_atom.position, atol=1e-6):
                real_index = j + 1  # 1-based indexing
                break
        h_distances.append((real_index, distance))
    
    # Sort by distance and select the nearest H atom
    h_distances.sort(key=lambda x: x[1])
    nearest_h_index, nearest_distance = h_distances[0]
    
    print(f"Found {len(h_atoms)} H atoms in total")
    print(f"Nearest H atom to Rh: index {nearest_h_index}, distance = {nearest_distance:.3f} Å")
    
    return nearest_h_index

def parse_coordinates_from_orca_output(output_content):
    """Parse coordinate information from ORCA output file"""
    try:
        # Look for the final coordinate section in ORCA output
        coord_patterns = [
            r'CARTESIAN COORDINATES \(ANGSTROEM\)(.*?)(?=CARTESIAN COORDINATES \(A\.U\.\)|$)',
            r'FINAL SINGLE POINT ENERGY.*?CARTESIAN COORDINATES \(ANGSTROEM\)(.*?)(?=CARTESIAN COORDINATES \(A\.U\.\)|$)'
        ]
        
        coord_section = None
        for pattern in coord_patterns:
            matches = re.findall(pattern, output_content, re.DOTALL)
            if matches:
                coord_section = matches[-1]  # Take the last match
                break
        
        if not coord_section:
            print("No coordinate section found in ORCA output file")
            return None
        
        # Parse coordinate data
        lines = coord_section.strip().split('\n')
        atoms_data = []
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    symbol = parts[0]
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    
                    atoms_data.append({
                        'symbol': symbol,
                        'position': [x, y, z]
                    })
                except (ValueError, IndexError):
                    continue
        
        if not atoms_data:
            print("No valid coordinate data found")
            return None
        
        # Create simple atoms object simulation
        class SimpleAtom:
            def __init__(self, symbol, position):
                self.symbol = symbol
                self.position = np.array(position)
        
        class SimpleAtoms:
            def __init__(self, atoms_data):
                self.atoms = [SimpleAtom(data['symbol'], data['position']) for data in atoms_data]
            
            def __iter__(self):
                return iter(self.atoms)
            
            def __len__(self):
                return len(self.atoms)
            
            def __getitem__(self, index):
                return self.atoms[index]
        
        atoms = SimpleAtoms(atoms_data)
        print(f"Successfully parsed {len(atoms)} atoms from ORCA output file")
        return atoms
        
    except Exception as e:
        print(f"Error in manual coordinate parsing: {e}")
        return None

def extract_npa_charge(output_content, atom_index):
    """
    Extract NPA charge for specified atom from ORCA output
    
    Parameters:
    output_content: ORCA output file content
    atom_index: Atom index (1-based)
    
    Returns:
    npa_charge: NPA charge value, returns None if not found
    """
    # Find NPA analysis section in ORCA output
    npa_patterns = [
        r'NATURAL POPULATION ANALYSIS(.*?)(?=NATURAL BOND ORBITAL|$)',
        r'NPA CHARGES(.*?)(?=NATURAL BOND|$)'
    ]
    
    npa_section = None
    for pattern in npa_patterns:
        match = re.search(pattern, output_content, re.DOTALL | re.IGNORECASE)
        if match:
            npa_section = match.group(1)
            break
    
    if not npa_section:
        print("NPA analysis section not found in ORCA output")
        return None
    
    # Match data for the specified atom
    pattern = rf"(\w+)\s+{atom_index}\s+([-\d\.]+)"
    match = re.search(pattern, npa_section)
    
    if match:
        npa_charge = float(match.group(2))
        print(f"Found NPA charge for atom {atom_index}: {npa_charge}")
        return npa_charge
    else:
        print(f"No NPA charge found for atom {atom_index}")
        return None

def extract_nmr_and_npa_data(output_content):
    """Extract NMR and NPA data from ORCA output file"""
    results = {}
    
    # Extract total number of atoms
    total_atoms_match = re.search(r'Number of atoms\s*\.*\s*(\d+)', output_content)
    if not total_atoms_match:
        print("Total number of atoms not found in ORCA output content.")
        return results
    else:
        total_atoms = int(total_atoms_match.group(1))
    
    print(f"Total atoms: {total_atoms}")

    # Use ASE to read atomic coordinate information
    try:
        # Extract coordinate information from ORCA output content
        atoms = parse_coordinates_from_orca_output(output_content)
        if atoms is None:
            return results
    except Exception as e:
        print(f"Error reading coordinates: {e}")
        return results
    
    # Find Rh atom
    rh_atoms = [atom for atom in atoms if atom.symbol == 'Rh']
    if len(rh_atoms) != 1:
        print(f"Warning: Found {len(rh_atoms)} Rh atoms, expected exactly 1")
        return results
    
    rh_atom = rh_atoms[0]
    # Find the real index of Rh atom
    rh_index = None
    for i, atom in enumerate(atoms):
        if atom.symbol == 'Rh':
            rh_index = i + 1  # 1-based indexing
            break
    
    print(f"Rh atom found at index {rh_index}")
    
    # Find the H atom closest to Rh
    nearest_h_index = find_nearest_h_atom(atoms, rh_atom)
    if nearest_h_index is None:
        print("Could not find nearest H atom")
        return results
    
    # Calculate atomic numbers for C1 and C3 (15th and 13th from the end)
    c1_atom = total_atoms - 14  # 15th from the end
    c3_atom = total_atoms - 12  # 13th from the end
    print(f"C1 atom (15th from end): {c1_atom}")
    print(f"C3 atom (13th from end): {c3_atom}")
    
    # Extract NMR shielding section from ORCA output
    nmr_patterns = [
        r'CHEMICAL SHIELDING SUMMARY(.*?)(?=TIMINGS|$)',
        r'Nucleus.*Isotropic.*Anisotropy(.*?)(?=TIMINGS|$)'
    ]
    
    nmr_text = ""
    for pattern in nmr_patterns:
        nmr_section = re.search(pattern, output_content, re.DOTALL | re.IGNORECASE)
        if nmr_section:
            nmr_text = nmr_section.group(1)
            break
    
    if not nmr_text:
        print("NMR shielding section not found in ORCA output.")
        return results
    
    # Extract NMR data for specified atoms
    target_atoms = [
        (rh_index, 'Rh'),
        (nearest_h_index, 'H'),
        (c3_atom, 'C3'),
        (c1_atom, 'C1')
    ]
    
    for atom_num, label in target_atoms:
        if atom_num is None:
            continue
            
        # Pattern to match NMR data
        nmr_pattern = rf"\s+{atom_num}\s+\w+\s+([-\d\.]+)\s+([-\d\.]+)"
        match = re.search(nmr_pattern, nmr_text)
        
        if match:
            if label in ['Rh', 'H', 'C3']:
                # Extract Anisotropy data
                anisotropy = float(match.group(2))
                results[f'{label}_Anisotropy'] = anisotropy
                print(f"Found {label} Anisotropy: {anisotropy}")
            
            # For C1, need to extract XZ data from full tensor
            if label == 'C1':
                # Look for full tensor data to get XZ component
                full_tensor_pattern = rf"\s+{atom_num}\s+\w+.*?XZ:\s*([-\d\.]+)"
                full_match = re.search(full_tensor_pattern, nmr_text, re.DOTALL)
                if full_match:
                    xz = float(full_match.group(1))
                    results[f'{label}_XZ'] = xz
                    print(f"Found C1 XZ: {xz}")
        else:
            print(f"No NMR data found for atom {atom_num} ({label})")
    
    # Extract NPA charge for H atom
    h_npa_charge = extract_npa_charge(output_content, nearest_h_index)
    if h_npa_charge is not None:
        results['NBO_H'] = h_npa_charge
    
    return results

def parse_input_file(file_path):
    """Parse input geometry file (.inp or .xyz)"""
    elements = []
    coordinates = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        reading_coordinates = False
        coord_section_found = False
        
        for line in lines:
            line = line.strip()
            
            # Look for coordinate section markers
            if line.startswith('*') and 'xyz' in line.lower():
                reading_coordinates = True
                coord_section_found = True
                continue
            
            # Stop reading if we hit another section
            if reading_coordinates and line.startswith('*'):
                break
                
            if reading_coordinates and line:
                parts = line.split()
                
                if len(parts) >= 4:
                    try:
                        element = parts[0]
                        coord = [float(parts[1]), float(parts[2]), float(parts[3])]
                        elements.append(element)
                        coordinates.append(coord)
                    except ValueError:
                        continue
    
    if not coordinates or len(elements) != len(coordinates):
        print(f"Parsing failed: {file_path}")
        return None, None
    
    print(f"Successfully parsed {len(elements)} atoms from input file")
    print("Note: Alkene coordinates are located at the end of the coordinate list")
    return elements, coordinates

def find_rh_atom(elements):
    """Find the index of Rh atom"""
    for i, element in enumerate(elements):
        if element == 'Rh':
            return i
    return None

def find_nearest_h_atom_input(elements, coordinates, rh_index):
    """Find the index of H atom nearest to Rh"""
    h_indices = [i for i, element in enumerate(elements) if element == 'H']
    
    if not h_indices:
        return None
    
    rh_coord = np.array(coordinates[rh_index])
    min_distance = float('inf')
    nearest_h_index = None
    
    for h_index in h_indices:
        h_coord = np.array(coordinates[h_index])
        distance = np.linalg.norm(h_coord - rh_coord)
        if distance < min_distance:
            min_distance = distance
            nearest_h_index = h_index
    
    print(f"Found nearest H atom at index {nearest_h_index}, distance = {min_distance:.3f} Å")
    return nearest_h_index

def find_nearest_p_atom(elements, coordinates, rh_index):
    """Find the index of P atom nearest to Rh"""
    p_indices = [i for i, element in enumerate(elements) if element == 'P']
    
    if not p_indices:
        return None
    
    rh_coord = np.array(coordinates[rh_index])
    min_distance = float('inf')
    nearest_p_index = None
    
    for p_index in p_indices:
        p_coord = np.array(coordinates[p_index])
        distance = np.linalg.norm(p_coord - rh_coord)
        if distance < min_distance:
            min_distance = distance
            nearest_p_index = p_index
    
    print(f"Found nearest P atom at index {nearest_p_index}, distance = {min_distance:.3f} Å")
    return nearest_p_index

def find_all_p_atoms(elements, coordinates, rh_index):
    """Find all P atom indices and sort by distance from Rh"""
    p_indices = [i for i, element in enumerate(elements) if element == 'P']
    
    if not p_indices:
        return []
    
    # Sort by distance
    rh_coord = np.array(coordinates[rh_index])
    p_indices_sorted = sorted(p_indices, key=lambda idx: np.linalg.norm(np.array(coordinates[idx]) - rh_coord))
    
    return p_indices_sorted

def find_nearest_c_to_rh(elements, coordinates, rh_index):
    """Find the index of C atom nearest to Rh"""
    c_indices = [i for i, element in enumerate(elements) if element == 'C']
    
    if not c_indices:
        return None
    
    rh_coord = np.array(coordinates[rh_index])
    min_distance = float('inf')
    nearest_c_index = None
    
    for c_index in c_indices:
        c_coord = np.array(coordinates[c_index])
        distance = np.linalg.norm(c_coord - rh_coord)
        if distance < min_distance:
            min_distance = distance
            nearest_c_index = c_index
    
    print(f"Found nearest C atom to Rh at index {nearest_c_index}, distance = {min_distance:.3f} Å")
    return nearest_c_index

def find_nearest_o_to_c(elements, coordinates, c_index):
    """Find the index of O atom nearest to C"""
    o_indices = [i for i, element in enumerate(elements) if element == 'O']
    
    if not o_indices:
        return None
    
    c_coord = np.array(coordinates[c_index])
    min_distance = float('inf')
    nearest_o_index = None
    
    for o_index in o_indices:
        o_coord = np.array(coordinates[o_index])
        distance = np.linalg.norm(o_coord - c_coord)
        if distance < min_distance:
            min_distance = distance
            nearest_o_index = o_index
    
    print(f"Found nearest O atom to C at index {nearest_o_index}, distance = {min_distance:.3f} Å")
    return nearest_o_index

def find_nearest_atom_to_p(elements, coordinates, p_index, rh_index):
    """Find the index of nearest atom to P (excluding Rh) - improved version"""
    # Get indices of all non-Rh atoms
    non_rh_indices = [i for i in range(len(elements)) if i != rh_index]
    
    if not non_rh_indices:
        return None
    
    p_coord = np.array(coordinates[p_index])
    min_distance = float('inf')
    nearest_atom_index = None
    
    for atom_index in non_rh_indices:
        if atom_index == p_index:  # Skip P atom itself
            continue
            
        atom_coord = np.array(coordinates[atom_index])
        distance = np.linalg.norm(atom_coord - p_coord)
        if distance < min_distance:
            min_distance = distance
            nearest_atom_index = atom_index
    
    if nearest_atom_index is not None:
        print(f"Found nearest atom to P at index {nearest_atom_index} ({elements[nearest_atom_index]}), distance = {min_distance:.3f} Å")
    else:
        print("No nearest atom found for P")
    
    return nearest_atom_index

def calculate_distance(coord1, coord2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2)

def calculate_angle(coordA, coordB, coordC):
    """Calculate angle ABC in degrees"""
    ba = np.array(coordA) - np.array(coordB)
    bc = np.array(coordC) - np.array(coordB)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_sterimol_parameters(elements, coordinates, atom_index, p_index):
    """Calculate Sterimol parameters"""
    try:
        sterimol = Sterimol(elements, coordinates, atom_index, p_index)
        return sterimol.B_1_value
    except Exception as e:
        print(f"Error calculating Sterimol parameters: {e}")
        return None

def validate_coordinates(coordinates):
    """Validate coordinate data"""
    for i, coord in enumerate(coordinates):
        if any(not isinstance(x, (int, float)) or math.isnan(x) for x in coord):
            print(f"Abnormal coordinates for atom {i}: {coord}")
            return False
    return True

def calculate_buried_volume(elements, coordinates, index, excluded_atoms, z_axis_atoms=None, xz_plane_atoms=None, radius=3.5):
    """Calculate buried volume"""
    try:
        bv = BuriedVolume(elements, coordinates, index, excluded_atoms=excluded_atoms, 
                         z_axis_atoms=z_axis_atoms, xz_plane_atoms=xz_plane_atoms, radius=radius)
        return bv.fraction_buried_volume
    except Exception as e:
        print(f"Error calculating buried volume: {e}")
        return None

def extract_structural_features(elements, coordinates):
    """Extract structural features from input geometry file"""
    results = {}
    
    if not validate_coordinates(coordinates):
        print("Coordinate validation failed")
        return results

    if len(coordinates) < 15:
        print("Insufficient atomic coordinates")
        return results

    try:
        # Find key atoms
        rh_index = find_rh_atom(elements)
        if rh_index is None:
            print("Rh atom not found")
            return results
        
        nearest_h_index = find_nearest_h_atom_input(elements, coordinates, rh_index)
        if nearest_h_index is None:
            print("H atom not found")
            return results
        
        # Find C atom nearest to Rh
        nearest_c_to_rh_index = find_nearest_c_to_rh(elements, coordinates, rh_index)
        if nearest_c_to_rh_index is None:
            print("C atom connected to Rh not found")
            return results
        
        # Find O atom nearest to C
        nearest_o_to_c_index = find_nearest_o_to_c(elements, coordinates, nearest_c_to_rh_index)
        if nearest_o_to_c_index is None:
            print("O atom connected to C not found")
            return results
        
        # Find all P atoms and sort by distance
        all_p_indices = find_all_p_atoms(elements, coordinates, rh_index)
        if len(all_p_indices) < 2:
            print("Insufficient P atoms found")
            return results
        
        # Take the two nearest P atoms
        nearest_p_indices = all_p_indices[:2]
        print(f"Found nearest P atoms at indices: {nearest_p_indices}")
        
        # Modified here: use new function name
        nearest_atom_to_p_index = find_nearest_atom_to_p(elements, coordinates, nearest_p_indices[0], rh_index)
        
        # Get C1 and C2 indices (15th and 14th from the end)
        # Note: Alkene coordinates are at the end of the coordinate list
        c1_index = len(coordinates) - 15  # 15th from the end
        c2_index = len(coordinates) - 14  # 14th from the end
        
        print(f"Rh index: {rh_index}, H index: {nearest_h_index}")
        print(f"C index (nearest to Rh): {nearest_c_to_rh_index}, O index (nearest to C): {nearest_o_to_c_index}")
        print(f"P indices: {nearest_p_indices}, nearest atom to P (for Sterimol): {nearest_atom_to_p_index}")
        print(f"C1 index: {c1_index}, C2 index: {c2_index}")
        print("Note: C1 and C2 are part of the alkene moiety at the end of the structure")

        # Calculate required parameters
        # 1. H-C2 distance
        results['H-C2'] = calculate_distance(coordinates[nearest_h_index], coordinates[c2_index])
        
        # 2. Rh-H distance
        results['Rh-H'] = calculate_distance(coordinates[rh_index], coordinates[nearest_h_index])
        
        # 3. H-Rh-C1 angle
        results['H-Rh-C1'] = calculate_angle(coordinates[nearest_h_index], coordinates[rh_index], coordinates[c1_index])
        
        # 4. Rh-C2 distance
        results['Rh-C2'] = calculate_distance(coordinates[rh_index], coordinates[c2_index])
        
        # 5. C1-Rh-C2 angle
        results['C1-Rh-C2'] = calculate_angle(coordinates[c1_index], coordinates[rh_index], coordinates[c2_index])

        # 6. Calculate Dispersion parameters to get P_int_H (using H index+1)
        disp = Dispersion(elements, coordinates)
        try:
            # Use H index+1 (convert from 0-based to 1-based indexing)
            results['P_int_H'] = disp.atom_p_int[nearest_h_index + 1]
        except (KeyError, IndexError):
            results['P_int_H'] = None
            print(f"Unable to get P_int value for H atom")

        # 7. Calculate Sterimol_B1
        if nearest_atom_to_p_index is not None:
            # Add additional check: ensure suitable atoms are selected for Sterimol calculation
            atom_element = elements[nearest_atom_to_p_index]
            p_element = elements[nearest_p_indices[0]]
            distance = calculate_distance(coordinates[nearest_atom_to_p_index], coordinates[nearest_p_indices[0]])
            
            print(f"Sterimol calculation preparation:")
            print(f"  Selected atom: {atom_element} (index {nearest_atom_to_p_index})")
            print(f"  P atom: {p_element} (index {nearest_p_indices[0]})")
            print(f"  Distance: {distance:.3f} Å")
            
            # Check if distance is reasonable (typical chemical bond lengths are 1-3 Å)
            if 1.0 <= distance <= 4.0:
                sterimol_b1 = calculate_sterimol_parameters(elements, coordinates, nearest_atom_to_p_index +1, nearest_p_indices[0]+1)
            else:
                print(f"Distance {distance:.3f} Å seems unreasonable for Sterimol calculation, skipping")
                sterimol_b1 = None
        else:
            print("No atoms found around P atom, skipping Sterimol calculation")
            sterimol_b1 = None

        results['Sterimol_B1'] = sterimol_b1

        # 8. Calculate V_bur (buried volume)
        num_atoms = len(coordinates)
        # Exclude Rh, nearest C, O connected to C, and last 15 atoms
        excluded_atoms = [
            rh_index + 1,           # Rh (convert to 1-based indexing)
            nearest_c_to_rh_index + 1,  # C atom nearest to Rh
            nearest_o_to_c_index + 1    # O atom connected to C
        ] + list(range(num_atoms - 15, num_atoms))  # Last 15 atoms (alkene part)
        
        # Set z-axis atoms as 15th and 14th from the end (using 1-based indexing)
        z_axis_atoms = [num_atoms - 15 + 1, num_atoms - 14 + 1]
        # Set xz-plane atoms as two P atom indices (using 1-based indexing)
        xz_plane_atoms = [idx + 1 for idx in nearest_p_indices]
        
        print(f"V_bur calculation parameters:")
        print(f"  Rh index: {rh_index + 1}")
        print(f"  Excluded atoms: {excluded_atoms}")
        print(f"  Z-axis atoms: {z_axis_atoms}")
        print(f"  XZ-plane atoms: {xz_plane_atoms}")
        
        # Use Rh index+1 (convert from 0-based to 1-based indexing)
        v_bur = calculate_buried_volume(elements, coordinates, rh_index + 1, 
                                      excluded_atoms=excluded_atoms,
                                      z_axis_atoms=z_axis_atoms, 
                                      xz_plane_atoms=xz_plane_atoms, 
                                      radius=3.5)
        results['V_bur'] = v_bur

    except Exception as e:
        print(f"Error extracting structural features: {str(e)}")
        return {}
    
    return results

def process_files(folder_path):
    """Process ORCA output files and input geometry files"""
    all_results = []

    if not os.path.isdir(folder_path):
        print(f"Specified folder path does not exist: {folder_path}")
        return

    # Get all files and group by base filename
    files_by_basename = {}
    
    for file in os.listdir(folder_path):
        if file.endswith(('.out', '.inp', '.xyz')):
            basename = os.path.splitext(file)[0]  # Remove extension
            if basename not in files_by_basename:
                files_by_basename[basename] = {}
            
            if file.endswith('.out'):
                files_by_basename[basename]['output'] = file
            elif file.endswith(('.inp', '.xyz')):
                files_by_basename[basename]['input'] = file
    
    print(f"Found {len(files_by_basename)} file groups")

    # Process each file pair
    for basename, files in files_by_basename.items():
        if 'output' not in files or 'input' not in files:
            print(f"File group {basename} missing output or input file, skipping")
            continue
            
        print(f"\nProcessing file group: {basename}")
        
        result = {'File Name': basename}
        
        # Process ORCA output file to extract NMR data
        output_path = os.path.join(folder_path, files['output'])
        try:
            with open(output_path, 'r', encoding='utf-8', errors='ignore') as file:
                output_content = file.read()
                nmr_data = extract_nmr_and_npa_data(output_content)
                result.update(nmr_data)
        except Exception as e:
            print(f"Error processing output file {files['output']}: {e}")
            continue
        
        # Process input geometry file to extract structural features
        input_path = os.path.join(folder_path, files['input'])
        try:
            elements, coordinates = parse_input_file(input_path)
            if elements and coordinates:
                structural_data = extract_structural_features(elements, coordinates)
                result.update(structural_data)
            else:
                print(f"Unable to parse input file: {files['input']}")
                continue
        except Exception as e:
            print(f"Error processing input file {files['input']}: {e}")
            continue
        
        all_results.append(result)
        print(f"Successfully processed file group: {basename}")

    # Save results to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(folder_path, 'orca_feature_extraction_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nAll results saved to: {csv_path}")
        print(f"Successfully processed {len(all_results)} file groups")
        
        # Display results preview
        print("\nExtracted data preview:")
        print(df.head())
        
        # Display feature summary
        print(f"\nFeature extraction summary:")
        print(f"Total features extracted: {len(df.columns)-1}")  # -1 for File Name column
        feature_names = [col for col in df.columns if col != 'File Name']
        print(f"Features: {', '.join(feature_names)}")
        
    else:
        print("No results to save.")

def main():
    """
    Main function to run the feature extraction
    
    Usage:
    1. Update the folder_path variable below to point to your data directory
    2. Ensure your folder contains paired ORCA output files (.out) and input geometry files (.inp or .xyz)
    3. Run the script
    
    Expected file structure:
    - compound1.out (ORCA output file)
    - compound1.inp (ORCA input file with geometry)
    - compound2.out
    - compound2.inp
    - etc.
    
    The script will generate: orca_feature_extraction_results.csv
    """
    
    # Update this path to your data folder
    folder_path = "/path/to/your/orca/files"
    
    print("="*60)
    print("ORCA Molecular Feature Extraction Script")
    print("="*60)
    print("This script extracts 13 key features for organometallic complexes:")
    print("1. Rh_Anisotropy - NMR shielding anisotropy for Rh")
    print("2. H_Anisotropy - NMR shielding anisotropy for H")
    print("3. C3_Anisotropy - NMR shielding anisotropy for C3")
    print("4. C1_XZ - NMR XZ tensor component for C1")
    print("5. NBO_H - NPA charge on H atom")
    print("6. H-C2 - Distance between H and C2")
    print("7. Rh-H - Distance between Rh and H")
    print("8. H-Rh-C1 - Angle H-Rh-C1")
    print("9. Rh-C2 - Distance between Rh and C2")
    print("10. C1-Rh-C2 - Angle C1-Rh-C2")
    print("11. P_int_H - Dispersion interaction parameter for H")
    print("12. Sterimol_B1 - Sterimol B1 descriptor")
    print("13. V_bur - Buried volume fraction")
    print("="*60)
    print(f"Processing files in: {folder_path}")
    print("Note: Alkene coordinates should be at the end of input geometry")
    print("="*60)
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist!")
        print("Please update the folder_path variable in the main() function.")
        return
    
    # Start processing
    process_files(folder_path)
    print("="*60)
    print("Feature extraction completed!")
    print("="*60)

# Example usage and testing function
def test_with_sample_data():
    """
    Test function for development - replace with actual data paths
    """
    # Example folder structure for testing
    test_folder = "/path/to/test/data"  # Update this for testing
    
    print("Running test with sample data...")
    print(f"Test folder: {test_folder}")
    
    if os.path.exists(test_folder):
        process_files(test_folder)
    else:
        print("Test folder not found. Please update test_folder path.")

if __name__ == "__main__":
    # Run the main function
    main()
    
    # Uncomment the line below for testing
    # test_with_sample_data()