# src/bone_splitter.py

from pathlib import Path
from typing import Optional, Tuple
import pyvista as pv


def split_tibia(
    tibia_both: pv.PolyData,
    femur_mesh: Optional[pv.PolyData] = None,
    save_dir: str = "./meshes/",
) -> Tuple[pv.PolyData, pv.PolyData]:
    """Splits a combined dual-tibia mesh into independent left and right tibia meshes.

    Args:
        tibia_both : PyVista PolyData containing both left and right tibias.
        femur_mesh : Optional reference femur mesh (included for pipeline
          compatibility).
        save_dir   : Directory path where raw STL meshes will be saved.

    Returns:
        (tibia_left, tibia_right) as PyVista PolyData surface meshes.
    """
    # 1. Separate disconnected bodies
    bodies = tibia_both.split_bodies()

    if len(bodies) < 2:
        raise ValueError(
            f"Expected 2 tibia meshes, but found {len(bodies)}. "
            "Check that TotalSegmentator segmentation mask contains both tibias."
        )

    # 2. Filter out floating noise: keep the 2 largest bodies by vertex count
    main_tibias = sorted(bodies, key=lambda b: b.n_points, reverse=True)[:2]

    # 3. Assign Left vs Right using physical X-coordinate
    # In standard patient space (LPS): higher X = Left tibia, lower X = Right tibia
    tibia_a, tibia_b = main_tibias[0], main_tibias[1]

    if tibia_a.center[0] > tibia_b.center[0]:
        tibia_left_raw = tibia_a.extract_surface()
        tibia_right_raw = tibia_b.extract_surface()
    else:
        tibia_left_raw = tibia_b.extract_surface()
        tibia_right_raw = tibia_a.extract_surface()

    # 4. Save raw STL meshes
    output_path = Path(save_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tibia_left_raw.save(str(output_path / "tibia_left_raw.stl"))
    tibia_right_raw.save(str(output_path / "tibia_right_raw.stl"))

    print(f"Successfully split tibias:")
    print(f"  - Left Tibia X-Center:  {tibia_left_raw.center[0]:.2f} mm")
    print(f"  - Right Tibia X-Center: {tibia_right_raw.center[0]:.2f} mm")

    return tibia_left_raw, tibia_right_raw