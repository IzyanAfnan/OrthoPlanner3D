# src/series_merger.py

# Merges two NIfTI CT series covering different anatomical regions into one continuous volume along the Z axis.

#   from src.series_merger import merge_ct_series, inspect_volumes

#  First inspect to understand your data
#   inspect_volumes("./data/raw/patient_01_torso.nii.gz",
#                   "./data/raw/patient_01_lower.nii.gz")

#  Then merge
#   merged_path = merge_ct_series(
#       series_a_path="./data/raw/patient_01_torso.nii.gz",
#       series_b_path="./data/raw/patient_01_lower.nii.gz",
#       output_path="./data/raw/patient_01_merged.nii.gz"
#   )


import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from typing import Tuple, Dict
from pathlib import Path


def inspect_volumes(path_a : str, path_b : str ) -> Dict:

    """
    Inspect two NIfTI volumes and prints a full diagnostic report.

    Call this BEFORE merging to understand your data:
    - Z origin and end positions (physical mm)
    - Voxel spacing
    - Whether they overlap or have a gap
    - Estimated anatomical coverage

    Args:
        path_a: Path to first NIfTI volume
        path_b: Path to second NIfTI volume

        Returns:
            Dictionary with diagnostic information for both volumes
    """

    img_a= nib.load(path_a)
    img_b= nib.load(path_b)

    data_a= img_a.get_fdata()
    data_b= img_b.get_fdata()
    aff_a= img_a.affine
    aff_b= img_b.affine

    spacing_a= np.abs([aff_a[0, 0], aff_a[1, 1], aff_a[2, 2]])
    spacing_b= np.abs([aff_b[0, 0], aff_b[1, 1], aff_b[2, 2]])

    z_origin_a= aff_a[2, 3]
    z_origin_b= aff_b[2, 3]
    z_end_a= z_origin_a + data_a.shape[2] * aff_a[2, 2]
    z_end_b= z_origin_b + data_b.shape[2] * aff_b[2, 2]

    z_min_a= min(z_origin_a, z_end_a)
    z_max_a= max(z_origin_a, z_end_a)
    z_min_b= min(z_origin_b, z_end_b)
    z_max_b= max(z_origin_b, z_end_b)

    coverage_a= data_a.shape[2] * spacing_a[2]
    coverage_b= data_b.shape[2] * spacing_b[2]

    overlap= min(z_max_a, z_max_b) - max(z_min_a, z_min_b)
    gap= max(z_min_a, z_min_b) - min(z_max_a, z_max_b)

    print("=" * 60)
    print("  CT SERIES INSPECTION REPORT")
    print("=" * 60)
    print(f"\nSeries A: {Path(path_a).name}")
    print(f"   Shape:       {data_a.shape}")
    print(f"   Spacing:     {spacing_a.round(3)} mm")
    print(f"   Z range:     {z_min_a:.1f} to {z_max_a:.1f} mm")
    print(f"   Z coverage:  {coverage_a:.0f} to {coverage_a/10:.1f} cm")
    print(f"   HU range:    {data_a.min():.0f} to {data_a.max():.0f}")

    print(f"\nSeries B: {Path(path_b).name}")
    print(f"    Shape:      {data_b.shape}")
    print(f"    Spacing:    {spacing_b.round(3)} mm")
    print(f"    Z range:    {z_min_b:.1f} to {z_max_b:.1f} mm")
    print(f"    Z coverage:  {coverage_b:.0f} to {coverage_b/10:.1f} cm")
    print(f"    HU range: {data_b.min():.0f} to {data_b.max():.0f}")


    print(f"\nRelationship:")
    if spacing_a[2] != spacing_b[2]:
        print(f"  WARNING: Z spacings differ ({spacing_a[2]:.2f} vs "
              f"{spacing_b[2]:.2f} mm) — may cause stitch artifacts")
    else:
        print(f"  Z spacing:    {spacing_a[2]:.2f} mm (MATCH — good)")

    if abs(spacing_a[0] - spacing_b[0]) > 0.01:
        print(f"  X/Y spacing:  DIFFER ({spacing_a[0]:.3f} vs "
              f"{spacing_b[0]:.3f} mm) — resampling required")
    else:
        print(f"  X/Y spacing:  {spacing_a[0]:.3f} mm (MATCH — good)")

    if overlap > 0:
        print(f"  Overlap zone: {overlap:.1f} mm "
              f"({int(overlap/spacing_a[2])} slices)")
    elif gap > 0:
        print(f"  GAP DETECTED: {gap:.1f} mm of missing anatomy. "
              f"Merged volume will have a gap. Consider requesting "
              f"additional NMDID series to fill this region.")
    else:
        print(f"  No overlap, no gap — volumes adjoin exactly.")

    total_coverage = max(z_max_a, z_max_b) - min(z_min_a, z_min_b)
    print(f"\n  Combined coverage: {total_coverage:.0f} mm "
          f"({total_coverage/10:.1f} cm)")
    print(f"  (Hip to ankle typically needs ~700-800 mm)")

    info = {
        "z_min_a": z_min_a, "z_max_a": z_max_a,
        "z_min_b": z_min_b, "z_max_b": z_max_b,
        "spacing_a": spacing_a, "spacing_b": spacing_b,
        "overlap_mm": overlap if overlap > 0 else 0,
        "gap_mm": gap if gap > 0 else 0,
        "total_coverage_mm": total_coverage
    }

    print("=" * 60)
    return info


def merge_ct_series(
        series_a_path: str,
        series_b_path: str,
        output_path: str,
        resample_order: int= 1
) -> str:

    """
    Merge two overlapping NIfTI CT series into one continuous volume.

    The function automatically:
    - Identifies which series is superior (closer to head, higher Z)
    - Identifies which is inferior (closer to feet, lower Z)
    - Resamples inferior series X/Y spacing to match superior if needed
    - Handles X/Y size difference by padding or cropping
    - Finds the overlpa region and uses superior series data there
    - Concatenates into one volume with correct affine

    The superior series is used as spatial reference because it typically contains the femoral head which is the most critical landmark (C_hip). This ensures the femoral head coordinates are in the correct physical space.

    Args:
        series_a_path   : path to first NIfTI series (any order)
        aeries_b_path   : Path to second NIfTI series (any ordfer)
        output_path     : Where to save the merged NIfTI (.nii.gz)
        resample_order  : Interpolation order for scipy.ndimage.zoom 
                          0= nearest, 1= linear (default), 3= cubic
    
    Returns:
        output_path (for chaining)

    Raises:
        ValueError: If a gap is detected between the two series (anatomy would be missing in the merged volume)
    """

    print("=" * 60)
    print(" SERIES MERGER ")
    print("=" * 60)


    # ── Load both volumes ──────────────────────────────────────────
    img_a = nib.load(series_a_path)
    img_b = nib.load(series_b_path)
    data_a = img_a.get_fdata()
    data_b = img_b.get_fdata()
    aff_a  = img_a.affine
    aff_b  = img_b.affine

    spacing_a = np.abs([aff_a[0,0], aff_a[1,1], aff_a[2,2]])
    spacing_b = np.abs([aff_b[0,0], aff_b[1,1], aff_b[2,2]])

    # Z start of each series (physical mm)
    z_origin_a = aff_a[2, 3]
    z_origin_b = aff_b[2, 3]
    z_end_a = z_origin_a + data_a.shape[2] * aff_a[2, 2]
    z_end_b = z_origin_b + data_b.shape[2] * aff_b[2, 2]

    z_min_a = min(z_origin_a, z_end_a)
    z_max_a = max(z_origin_a, z_end_a)
    z_min_b = min(z_origin_b, z_end_b)
    z_max_b = max(z_origin_b, z_end_b)

    # ── Identify superior vs inferior series ───────────────────────
    # Superior = covers the higher Z region (toward head)
    # Inferior = covers the lower Z region (toward feet)
    if z_max_a >= z_max_b:
        sup_data, sup_aff, sup_spacing= data_a, aff_a, spacing_a
        inf_data, inf_aff, inf_spacing= data_b, aff_b, spacing_b
        sup_z_min, sup_z_max= z_min_a, z_max_a
        inf_z_min, inf_z_max= z_min_b, z_max_b
        print(f"\nSuperior (towards head): {Path(series_a_path).name}")
        print(f"Inferior (towards feet): {Path(series_b_path).name}")

    else:
        sup_data, sup_aff, sup_spacing = data_b, aff_b, spacing_b
        inf_data, inf_aff, inf_spacing = data_a, aff_a, spacing_a
        sup_z_min, sup_z_max = z_min_b, z_max_b
        inf_z_min, inf_z_max = z_min_a, z_max_a
        print(f"\nSuperior (toward head): {Path(series_b_path).name}")
        print(f"Inferior (toward feet): {Path(series_a_path).name}")

    print(f"\nSuperior Z: {sup_z_min:.1f} mm → {sup_z_max:.1f} mm") 
    print(f"Inferior Z: {inf_z_min:.1f} mm → {inf_z_max:.1f} mm")

    # ── Check for gap ──────────────────────────────────────────────
    # Gap= inferior series ends below where superior series starts 
    gap_mm = sup_z_min - inf_z_max
    if gap_mm > sup_spacing[2] * 2:
        raise ValueError(
        f"GAP of {gap_mm:.1f} mm detected between series. Anatomy is missing. Request additional series to fill the gap before merging."
        )

    overlap_mm= inf_z_max - sup_z_min
    print(f"Overlap: {max(0, overlap_mm):.1f} "
          f"({max(0, int(overlap_mm/sup_spacing[2]))})"
          f"- superior series takes priority in overlapping zone")

    # ── Resample inferior X/Y to match superior spacing ────────────
    z_step= sup_spacing[2] 

    if abs(sup_spacing[0] - inf_spacing[0]) > 0.001:
        zoom_xy = inf_spacing[0] / sup_spacing[0]
        print(f"\nResampling inferior X/Y by factor {zoom_xy:.4f}"
              f"({inf_spacing[0]:.3f} mm → {sup_spacing[0]:.3f} mm) ...")
        inf_resampled= zoom(inf_data, (zoom_xy, zoom_xy, 1.0),
                            order=resample_order)
        print(f"    Resampled shape: {inf_resampled.shape}")

    else:
        inf_resampled= inf_data
        print(f"\nX/Y spacing match - no resampling needed")

    # ── Match X/Y dimensions ───────────────────────────────────────
    # After resampling, dimensions may not be exactly equal to superior
    target_xy= sup_data.shape[0]
    current_xy= inf_resampled.shape[0]    

    if current_xy != target_xy:
        diff= target_xy - current_xy
        if diff > 0:
            # Pad symmetrically with air HU
            pad = diff // 2
            pad_extra = diff - pad * 2
            inf_resampled = np.pad(
                inf_resampled, 
                ((pad, pad + pad_extra), (pad, pad + pad_extra), (0, 0)),
                mode= "constant", constant_values= -1024
            )
            print(f"Padded inferior to match {target_xy} x {target_xy}"
                  f"(added {diff} pixels)")

        else:
            # Crop from center
            start= (-diff) // 2
            inf_resampled = inf_resampled[
                start: start+target_xy, start: start+target_xy, :
            ]
            print(f"Cropped inferior to {target_xy}x{target_xy}")

    # ── Find how many inferior slices sit BELOW the superior series ─
    # These are the slices we keep from the inferior series
    # (the overlap zone is handled by the superior series)
    if overlap_mm > 0:
        # How many inferior slices ABOVE sup_z_min? Skip those.
        slice_above_sup= int(round(
            (inf_z_max - sup_z_min) / z_step
        ))
        # Keep only inferior slices below the superior series start 
        inf_below_sup= inf_resampled[:, :, :-slice_above_sup] \
                       if slice_above_sup > 0 \
                       else inf_resampled
        print(f"\nInferior slice kept (below superior): {inf_below_sup.shape[2]}")
        print(f"Inferior slice skipped (overlap zone): {slice_above_sup}")

    else:
        inf_below_sup= inf_resampled

    # ── Concatenate: inferior (feet→overlap) + superior (overlap→head)
    merged_data= np.concatenate(
        [inf_below_sup, sup_data], axis=2
    )

    total_coverage_mm= merged_data.shape[2] * z_step
    print(f"\nMerged shape: {merged_data.shape}")
    print(f"Total coverage: {total_coverage_mm:.0f} mm  {total_coverage_mm/10:.1f} cm")

    # ── Build merged affine ────────────────────────────────────────
    # Merged volume starts at inferior series origin (the feet)
    merged_aff= sup_aff.ccopy()
    merged_aff[0, 0]= sup_spacing[0]    # X voxel size
    merged_aff[1, 1]= sup_spacing[1]    # Y voxel size
    merged_aff[2, 2]= z_step            # Z voxel size
    # Z origin= start of inferior series (feet region)
    merged_aff[2, 3]= inf_z_min

    # ── Save merged volume ─────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged_img= nib.Nifti1Image(
        merged_data.astype(np.int16), merged_aff
    )
    nib.save(merged_img, output_path)
    print(f"\nMerged volume saved → {output_path}")

    # ── Quality report ─────────────────────────────────────────────
    print("\n--- QUALITY REPORT ---")
    print(f"  Voxel spacing:      {sup_spacing} mm")
    print(f"  Volume shape:       {merged_data.shape}")
    print(f"  Z origin (feet):    {inf_z_min:.1f} mm")
    print(f"  Z end (head):       {inf_z_min + total_coverage_mm:.1f} mm")
    print(f"  HU range:           {merged_data.min():.0f} "
          f"to {merged_data.max():.0f}")
    print(f"  WARNING: X/Y lateral offset between series is NOT corrected by this merger.")
    print(f"  Expected FMA error: ~arctan(offset_mm / femur_length_mm)")
    print(f"  Acceptable for portfolio. For clinical use, apply")
    print(f"  rigid registration (SimpleITK) before segmentation.")
    print("=" * 60)

    return output_path


def verify_merge(merged_path: str) -> None:
    """
    Loads a merged volume and prints a quick verification summary.
    Run this after merge_ct_series to confirm the output looks right.
    """
    img = nib.load(merged_path)
    data = img.get_fdata()
    spacing = np.abs([img.affine[0,0], img.affine[1,1], img.affine[2,2]])

    print(f"\nVerification: {Path(merged_path).name}")
    print(f"  Shape:    {data.shape}")
    print(f"  Spacing:  {spacing.round(3)} mm")
    print(f"  Coverage: {data.shape[2] * spacing[2] / 10:.1f} cm")
    print(f"  HU range: {data.min():.0f} to {data.max():.0f}")

    if data.shape[2] * spacing[2] < 600:
        print("  WARNING: Coverage under 60cm — may not include "
              "full hip-to-ankle range.")
    else:
        print("  Coverage sufficient for lower extremity analysis.")

