# run_pipeline.py
#
# Master pipeline script for OrthoPlanner3D.
# Runs Stages 2-4 on a single patient's segmentation files.
#
# PREREQUISITES (run manually in terminal before this script):
#   1. Convert DICOM to NIfTI:
#      dcm2niix -o ./data/raw/ -f "patient_XX_torso" -z y <torso_dicom_folder>
#      dcm2niix -o ./data/raw/ -f "patient_XX_lower" -z y <lower_dicom_folder>
#
#   2. Merge series (if two-series acquisition like NMDID):
#      from src.series_merger import merge_ct_series
#      merge_ct_series("./data/raw/patient_XX_torso.nii.gz",
#                      "./data/raw/patient_XX_lower.nii.gz",
#                      "./data/raw/patient_XX_merged.nii.gz")
#
#   3. Run TotalSegmentator — total task (femur):
#      TotalSegmentator -i data/raw/patient_XX_merged.nii.gz
#                       -o data/segmentations/patient_XX_total.nii.gz
#                       --ml --fast --nr_thr_resamp 1 --nr_thr_saving 1
#
#   4. Run TotalSegmentator — appendicular task (tibia, fibula):
#      TotalSegmentator -i data/raw/patient_XX_merged.nii.gz
#                       -o data/segmentations/patient_XX_appendicular.nii.gz
#                       --ml --task appendicular_bones
#                       --nr_thr_resamp 1 --nr_thr_saving 1
#
# USAGE (after prerequisites above):
#   python run_pipeline.py --patient patient_01
#                          --femur_label 75
#                          --tibia_label 2

import argparse
import json
import numpy as np
import pyvista as pv
from pathlib import Path

from src.mesh_generation  import extract_bone_mesh
from src.bone_splitter    import split_tibia, verify_tibia_split
from src.mesh_processor   import process_mesh
from src.landmark_detector import detect_all_landmarks
from src.vector_alignment_solver import run_full_analysis


def run_pipeline(patient_id: str,
                 femur_label: int,
                 tibia_label: int) -> dict:
    """
    Runs the full OrthoPlanner3D pipeline for one patient.

    Args:
        patient_id  : e.g. "patient_01"
        femur_label : TotalSegmentator label ID for femur_left
                      (check np.unique output from verify_segmentation)
        tibia_label : TotalSegmentator label ID for tibia
                      (appendicular task, both tibias combined)

    Returns:
        Dictionary containing all kinematic results
    """
    print(f"\n{'='*60}")
    print(f"  OrthoPlanner3D Pipeline — {patient_id}")
    print(f"{'='*60}\n")

    # Paths
    total_seg_path = f"./data/segmentations/{patient_id}_total.nii.gz"
    app_seg_path   = f"./data/segmentations/{patient_id}_appendicular.nii.gz"
    mesh_dir       = f"./meshes/{patient_id}/"
    results_dir    = f"./results/{patient_id}/"

    Path(mesh_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # ── STAGE 2A: Extract raw bone meshes ─────────────────────────
    print("STAGE 2: Mesh Generation")
    print("-" * 40)

    femur_raw = extract_bone_mesh(
        seg_nifti_path=total_seg_path,
        label_id=femur_label,
        save_path=f"{mesh_dir}/femur_left_raw.stl"
    )

    tibia_both_raw = extract_bone_mesh(
        seg_nifti_path=app_seg_path,
        label_id=tibia_label,
        save_path=f"{mesh_dir}/tibia_both_raw.stl"
    )

    # ── STAGE 2B: Process femur ────────────────────────────────────
    femur_proc = process_mesh(
        femur_raw,
        save_path=f"{mesh_dir}/femur_left_processed.stl"
    )

    # ── STAGE 2C: Split tibias and process left ────────────────────
    tibia_left_raw, tibia_right_raw = split_tibia(
        tibia_both=tibia_both_raw,
        femur_mesh=femur_proc,
        save_dir=mesh_dir
    )

    if not verify_tibia_split(tibia_left_raw, tibia_right_raw,  femur_mesh=femur_proc):
        raise RuntimeError(f"Tibia split verification failed for    {patient_id}")

    tibia_proc = process_mesh(
        tibia_left_raw,
        save_path=f"{mesh_dir}/tibia_left_processed.stl"
    )

    # ── STAGE 3: Landmark Detection ────────────────────────────────
    print("\nSTAGE 3: Landmark Detection")
    print("-" * 40)

    landmarks = detect_all_landmarks(femur_proc, tibia_proc)

    # ── STAGE 4: Kinematic Analysis ────────────────────────────────
    print("\nSTAGE 4: Kinematic Analysis")
    print("-" * 40)

    results = run_full_analysis(landmarks, femur_proc, tibia_proc)

    # ── Save results to JSON ───────────────────────────────────────
    # Convert numpy arrays to lists for JSON serialisation
    serialisable = {}
    for k, v in {**landmarks, **results}.items():
        if isinstance(v, np.ndarray):
            serialisable[k] = v.tolist()
        elif isinstance(v, (np.float64, np.float32)):
            serialisable[k] = float(v)
        else:
            serialisable[k] = v

    results_path = f"{results_dir}/kinematic_results.json"
    with open(results_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nResults saved → {results_path}")

    print(f"\n{'='*60}")
    print(f"  Pipeline complete for {patient_id}")
    print(f"  Meshes:  {mesh_dir}")
    print(f"  Results: {results_path}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OrthoPlanner3D — run pipeline for one patient"
    )
    parser.add_argument(
        "--patient", required=True,
        help="Patient ID, e.g. patient_01"
    )
    parser.add_argument(
        "--femur_label", type=int, required=True,
        help="TotalSegmentator femur_left label ID from total task"
    )
    parser.add_argument(
        "--tibia_label", type=int, required=True,
        help="TotalSegmentator tibia label ID from appendicular task"
    )

    args = parser.parse_args()
    run_pipeline(args.patient, args.femur_label, args.tibia_label)