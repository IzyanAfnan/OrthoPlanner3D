# src/mesh_generation.py
# Applies Laplacian smoothing and quadratic decimation to raw bone meshes

import pyvista as pv
from pathlib import Path


def process_mesh(raw_mesh: pv.PolyData,
                 save_path: str,
                 smooth_iter: int= 30,
                 smooth_factor: float= 0.1,
                  decimate_reduction: float = 0.8 ) -> pv.PolyData:

    """
    Smooths and decimates a raw bone mesh for landmark detection and real time rendering.

    Smoothing first removes staircase voxel artifacts.
    Decimation then reduces face count for rendering performance.
    Normal computation prepares the mesh for lighting in the dashboard.

    Args:
        raw_mesh            : Input PyVista PolyData from Marching cubes.
        save_path           : Path to save the processed STL
        smooth_iter         : Laplacian smoothing iterations (default 30)
        smooth_factor       : Relaxation factor 0-1 (default 0.1 = gentle)
        decimation_reduction: Fraction of faces to remove (default 0.8)

        Returns:
            Processed PyVista PolyData mesh
    """

    original_faces= raw_mesh.n_faces

    smoothed= raw_mesh.smooth(n_iter= smooth_iter,
                              relaxation_factor= smooth_factor,
    )

    decimated= smoothed.decimate(target_reduction= decimate_reduction)

    final_mesh = decimated.compute_normals(auto_orient_normals=True)

    kept_pct= 100 * final_faces / original_faces \
              if (final_faces := final_mesh.n_faces) else 0
    print(f"Procedded: {original_faces:,} → {final_mesh.n_faces} faces"
          f"{kept_pct:.0f}% kept")

    Path(save_path).parent.mkdir(parents= True, exist_ok= True)
    final_mesh.save(save_path)
    print(f"Saved → {save_path}")

    return final_mesh