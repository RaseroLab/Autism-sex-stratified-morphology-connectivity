# cortical_mind_matrix_generator.py

import os
import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from MIND_helpers import calculate_mind_network

def parse_freesurfer_lut_cortical(lut_file_path):
    """Parse FreeSurfer LUT to get cortical region labels (34 left, 34 right)."""
    cortical_labels = {}
    if not os.path.exists(lut_file_path):
        print(f"LUT file not found: {lut_file_path}")
        return {}

    try:
        with open(lut_file_path, 'r') as file:
            for line in file:
                if not line.strip() or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        label = int(parts[0])
                        region = ' '.join(parts[1:])
                        if 'ctx-' in region:
                            cortical_labels[label] = region
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error reading LUT file: {e}")
    return cortical_labels

def extract_voxel_distributions(vbm_image_path, parcellation_image_path, lut):
    """Extract voxel intensities for each cortical region from VBM and parcellation images."""
    if not (os.path.exists(vbm_image_path) and os.path.exists(parcellation_image_path)):
        print(f"Missing file: {vbm_image_path if not os.path.exists(vbm_image_path) else parcellation_image_path}")
        return {}

    vbm_data = nib.load(vbm_image_path).get_fdata()
    parc_data = nib.load(parcellation_image_path).get_fdata()

    distributions = {}
    valid_lut = {label: name for label, name in lut.items() if label in np.unique(parc_data)}

    for label, region in valid_lut.items():
        mask = np.isclose(parc_data, label, atol=1e-3)
        voxels = vbm_data[mask]
        if voxels.size >= 10:
            distributions[region] = voxels

    return distributions

def compute_mind_similarity(distributions, subject_id):
    """Compute cortical MIND similarity matrix from region-wise distributions."""
    region_names = list(distributions.keys())
    df = pd.DataFrame([(region, val) for region, values in distributions.items() for val in values],
                      columns=['Label', 'Value'])
    try:
        return calculate_mind_network(df, ['Value'], region_names, resample=True)
    except Exception as e:
        print(f"Skipping subject {subject_id} due to error in MIND computation: {e}")
        return None

def process_subject(subject_id, paths, lut):
    """Process a single subject to extract cortical similarity matrix."""
    vbm_image_path = os.path.join(paths['vbm_images_dir'], f"mwp1{subject_id}_T1w.nii_output.mgz")
    parcellation_image_path = os.path.join(paths['freesurfer_subjects_dir'], f"{subject_id}/mri/aparc+aseg.mgz")
    output_path = os.path.join(paths['output_results_dir'], f"ACEMIND-Cortical-{subject_id}.csv")

    if not os.path.exists(vbm_image_path) or not os.path.exists(parcellation_image_path):
        print(f"Skipping {subject_id}: Missing required MRI files")
        return

    print(f"Processing subject: {subject_id}")
    distributions = extract_voxel_distributions(vbm_image_path, parcellation_image_path, lut)
    if distributions:
        mind_matrix = compute_mind_similarity(distributions, subject_id)
        if mind_matrix is not None:
            mind_matrix.to_csv(output_path, index=True)
            print(f"Saved MIND matrix for {subject_id} at {output_path}")

def process_all_subjects(paths, lut, n_jobs=-1):
    """Process all subjects using parallel jobs and save outputs."""
    subject_ids = [subj for subj in os.listdir(paths['freesurfer_subjects_dir'])
                   if os.path.isdir(os.path.join(paths['freesurfer_subjects_dir'], subj))]

    os.makedirs(paths['output_results_dir'], exist_ok=True)

    Parallel(n_jobs=n_jobs)(
        delayed(process_subject)(subj, paths, lut)
        for subj in subject_ids
    )

def main():
    # All paths point to Desktop for user testing or development
    desktop_path = os.path.expanduser("~/Desktop")

    paths = {
        'freesurfer_subjects_dir': desktop_path,
        'vbm_images_dir': desktop_path,
        'output_results_dir': desktop_path,
        'freesurfer_lut_file': os.path.join(desktop_path, "FreeSurferColorLUT.txt")
    }

    # Load LUT for cortical labels
    lut = parse_freesurfer_lut_cortical(paths['freesurfer_lut_file'])

    # Process all subjects and save results to Desktop
    process_all_subjects(paths, lut, n_jobs=-1)

if __name__ == "__main__":
    main()
