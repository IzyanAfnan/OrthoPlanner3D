# src/mesh_generation.py
# Converts a NIfTI segmentation mask into a raw PyVista surface mesh using the Marching Cubes algorithm.

import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes
import pyvista as pv
from pathlib import Path

def extract_bone_mesh(
        seg_nifti_path: str,
        label_id:  int,
        save_path: str) -> pv.PolyData:

    """
    Extract a single bone label from a segmentation NIfTI file and convert it to a 3D surface mesh using Marching Cubes.
    
    Args:
        seg_NIfTI_path : Path to the segmentation NIfTI (.nni.gz)
        label_id       : Integer label for the target bone
        save_path      : Where to save the raw stl file 
    """

    seg_img= nib.load(seg_nifti_path)
    seg_data= seg_img.get_fdata().astype(int)
    affine= seg_img.affine

    binary_mask= (seg_data == label_id).astype(np.uint8)
    voxel_count= int(binary_mask.sum())

    if voxel_count == 0:
        available= np.unique(seg_data).tolist()
        raise ValueError(
            f"Label {label_id} not found in {seg_nifti_path}. \n"
            f"Available labels {available}"
        )

    print(f"label {label_id} : {voxel_count} voxels found")

    voxel_spacing= np.abs([
        affine[0, 0], affine[1, 1], affine[2, 2]
    ])

    vertices, faces, _, _ = marching_cubes(
        binary_mask,
        level= 0.5,
        spacing= voxel_spacing
    )

    faces_pv= np.hstack([
        np.full((len(faces), 1), 3, dtype= int), faces
    ])

    mesh= pv.PolyData(vertices, faces_pv)

    Path(save_path).parent.mkdir(parents=True, exist_ok= True)
    mesh.save(save_path)
    print(f"Raw mesh saved -> {save_path}  ({mesh.n_faces:,}) faces")

    return mesh 


def load_mesh(stl_path: str) -> pv.PolyData:
    """
    Loads an existing  STL file as PyVista PolyData mesh.
    """
    mesh= pv.read(stl_path)
    print(f"loaded: {stl_path}  ({mesh.n_faces:,}) faces")
    return mesh
