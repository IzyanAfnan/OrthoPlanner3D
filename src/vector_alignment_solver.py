# src/vector_alignment_solver.py

# Computes the mechanical axes of the lower extermity and clinical alignment angles from anatomical landmark coordinates.

# Outputs:
#     v_FMA       : Femoral Mechanical Axis unit vector
#     v_TMA       : Tibial Mechanical Axis unit vector
#     hka_3d      : Hip-Knee-Ankle angle in 3D (degrees)
#     hka_coronal : HKA projected onto coronal plane (clinical standard)
#     deformity   : Deviation from 180° (+ = varus, - = valgus)
#     mLDFA       : Mechanical Lateral Distal Femoral Angle
#     MPTA        : Medial Proximal Tibial Angle

import numpy as np
import pyvista as pv
from typing import Dict, Tuple


def compute_mechanical_axes(
        landmarks: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Computes normalized femoral and tibial mechanical axis vectors


    Femoral Mechanical Axis (FMA): C_hip → C_knee_f
    Tibial Mechanical Axis (TMA): C_knee_t → C_ankle

    Both axes point superiorly (towards the head) for consistency.

    Args:
        landmarks: dict with keys C_hip, C_knee_f, C_knee_t, C_ankle

    Returns:
        dict with keys v_FMA, v_TMA, femur_length_mm, tibia_length_mm
    """

    C_hip   = landmarks["C_hip"]
    C_knee_f= landmarks["C_knee_f"]
    C_knee_t= landmarks["C_knee_t"]
    C_ankle = landmarks["C_ankle"]

    raw_FMA = C_knee_f - C_hip
    raw_TMA = C_knee_t - C_ankle

    femur_length = np.linalg.norm(raw_FMA)
    tibia_length = np.linalg.norm(raw_TMA)

    if femur_length < 1.0 or tibia_length < 1.0 :
        raise ValueError(
            "Degenerate axis detected (length < 1mm). Check that landmark coordinates are in mm and that C_hip ≠ C_knee_f and C_ankle ≠ C_knee_t"
        )

    v_FMA = raw_FMA / femur_length
    v_TMA = raw_TMA / tibia_length

    print(f"FMA: {v_FMA.round(4)}   |   femur length: {femur_length:.1f} mm")
    print(f"TMA: {v_TMA.round(4)}   |   tibia length: {tibia_length:.1f} mm")

    return {
        "v_FMA"             : v_FMA,
        "v_TMA"             : v_TMA,
        "femur_length_mm"   : float(femur_length),
        "tibia_length_mm"   : float(tibia_length)
    }


def compute_hka_angle(
        v_FMA: np.ndarray,
        v_TMA: np.ndarray
) -> Dict[str, float]:
    """
    Computes the Hip-Knee-Ankle (HKA) angle and deformity classification.

    Computes two values:
    - hka_3d        : full 3D angle between FMA and TMA
    - hka_coronal   : angle projected onto the coronal X-Z plane, simulating a clinical long-leg radiograph.

    Normal alignment: HKA ≈ 180°
    Varus (bow-legged)  : HKA < 178° (deformity > 2°)
    Valgus (knock-kneed): HKA > 182° (deformity < -2°)

    Agrs:
        v_FMA: Normalized femoral mechanical axis vector
        v_TMA: Normalized tibial mechanical axis vector

    Return:
        dict with hka_3d, hka_coronal, deformity, classification
    """

    # 3D angle
    dot_3d = np.dot(v_FMA, v_TMA)
    hka_3d= float(np.degrees(np.arccos(np.clip(dot_3d, -1.0, 1.0))))

    # Coronal projection - simulate long-leg radiograph
    v_FMA_cor= np.array([v_FMA[0], 0, v_FMA[2]])
    v_TMA_cor= np.array([v_TMA[0], 0, v_TMA[2]])
    v_FMA_cor= v_FMA_cor / np.linalg.norm(v_FMA_cor)
    v_TMA_cor= v_TMA_cor / np.linalg.norm(v_TMA_cor)
    dot_cor= np.clip(np.dot(v_FMA_cor, v_TMA_cor), -1.0, 1.0)

    hka_projected= float(np.degrees(np.arccos(dot_cor)))

    deformity= 180.0 - hka_projected

    if abs(deformity) <= 2.0:
        classification= "Normal Alignment"
    elif deformity > 2.0:
        classification= "Varus (bow-legged)"
    else:
        classification= "Valgus (knock-kneed)"

    return {
        "hka_3d"        : hka_3d,
        "hka_coronal"   : hka_projected,
        "deformity"     : deformity,
        "classification": classification
    }


def compute_joint_line_angles(
        femur_mesh: pv.PolyData,
        tibia_mesh: pv.PolyData,
        v_FMA: np.ndarray,
        v_TMA: np.ndarray,
        distal_fraction: float = 0.15,
        proximal_fraction: float = 0.15
) -> Dict[str, float]:
    """
    Computes mLDFA and MTPA - the femoral and tibial contributions to overall limb alignment deformity.

    mLDFA (Mechanical Lateral Distal Femoral Angle):
        Angle between FMA and the femoral joint line.
        Normal: ~87°. > 90° = femoral varus contribution.

    MPTA (Medial Proximal Tibial Angle):
        Angle between TMA and the tibial plateau.
        Normal: ~87°. < 85° = tibial varus contribution.

    Args:
         femur_mesh        : Processed femur PyVista mesh
        tibia_mesh        : Processed tibia PyVista mesh
        v_FMA             : Normalised femoral mechanical axis
        v_TMA             : Normalised tibial mechanical axis
        distal_fraction   : Fraction of femur height for condyle extraction
        proximal_fraction : Fraction of tibia height for plateau extraction

    Returns:
        dict with mLDFA and MPTA in degrees
    """

    # ── mLDFA ─────────────────────────────────────────────────────
    f_pts= np.array(femur_mesh.points)
    z_min_f, z_max_f = f_pts[:, 2].min(), f_pts[:, 2].max()
    z_threshold_f= z_min_f + distal_fraction * (z_max_f - z_min_f)
    distal = f_pts[f_pts[:, 2] <= z_threshold_f]
    lat_c = distal[np.argmax(distal[:, 0])]
    med_c = distal[np.argmin(distal[:, 0])]
    u_fem= (med_c - lat_c) / np.linalg.norm(med_c - lat_c)
    mLDFA= float(np.degrees(np.arccos(np.clip(np.dot(u_fem, v_FMA), -1.0, 1.0))))


    # ── MPTA ──────────────────────────────────────────────────────
    t_pts = np.array(tibia_mesh.points)
    z_min_t, z_max_t = t_pts[:, 2].min(), t_pts[:, 2].max()
    z_threshold_t = z_max_t - proximal_fraction * (z_max_t - z_min_t)
    proximal = t_pts[t_pts[:, 2] >= z_threshold_t]
    lat_p = proximal[np.argmax(proximal[:, 0])]
    med_p = proximal[np.argmin(proximal[:, 0])]
    u_tib = (med_p - lat_p) / np.linalg.norm(med_p - lat_p)
    MPTA = float(np.degrees(np.arccos(np.clip(np.dot(u_tib, v_TMA), -1.0, 1.0))))

    print(f"mLDFA : {mLDFA:.1f}°  (normal ~87°)")
    print(f"MPTA :  {MPTA:.1f}°  (normal ~87°)")

    return {
        "mLDFA" : mLDFA,
        "MPTA"  : MPTA
    }


def run_full_analysis(
        landmarks: Dict[str, np.ndarray],
        femur_mesh: pv.PolyData,
        tibia_mesh: pv.PolyData
) -> Dict:
    """
    Master function - runs full kinematic analysis from landmarks and bone meshes. 

    Args:
        landmarks   : Dict from landmark_detector.detect_all_landamrks()
        femur_mesh  : Processed femur PyVista mesh
        tibia_mesh  : Processed tibia PyVista mesh

    Returns:
        Complete results dictionary containing all axes, anglesand clinical classification 
    """

    print("=" * 55)
    print(" KINEMATIC ANALYSIS ")
    print("=" * 55)

    axes = compute_mechanical_axes(landmarks)
    angles = compute_hka_angle(axes["v_FMA"], axes["v_TMA"])
    joint = compute_joint_line_angles(femur_mesh, tibia_mesh, axes["v_FMA"], axes["v_TMA"])

    results = {**axes, **angles, **joint}
    print(f"\n{'─'*40}")
    print(f"  CLINICAL REPORT")
    print(f"{'─'*40}")
    print(f"  Femur length:    {axes['femur_length_mm']:.1f} mm")
    print(f"  Tibia length:    {axes['tibia_length_mm']:.1f} mm")
    print(f"  HKA (3D):        {angles['hka_3d']:.2f}°")
    print(f"  HKA (coronal):   {angles['hka_coronal']:.2f}°")
    print(f"  Deformity:       {angles['deformity']:+.2f}°")
    print(f"  Classification:  {angles['classification']}")
    print(f"  mLDFA:           {joint['mLDFA']:.2f}°")
    print(f"  MPTA:            {joint['MPTA']:.2f}°")
    print(f"{'─'*40}")

    return results