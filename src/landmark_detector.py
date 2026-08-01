# src_landmark_detector.py
# Automatically localizes the four anatomical landmarks required for Hip-Knee (HKA) Angle calculation from 3D bone meshes.

# Landmarks:
#     C_hip= femoral head center (sphere fitting)
#     C_knee_f= distal knee center (centroid of condylar region)
#     C_knee_t= proximal tibial center (tibial plateau centroid)
#     C_ankle= ankle center (transmalleolor axis center)

import numpy as np
import pyvista as pv
from typing import Tuple, Dict


def fit_sphere_lstsq(vertices: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fits a sphere to a cloud of 3D points using linear least-squares

    Expands (x-xc)² + (y-yc)² + (z-zc)² = R² into linear form Ax=B and solves the overdetermined system via Moore-Penrose puesdo-inverse.

    Args:
        vertices: (N, 3) array of 3D points

    Returns:
        center: (3,) array- sphere center[xc, yc, zc] in mm
        radius: float - sphere radius in mm
    """

    x, y, z= vertices[:,0], vertices[:,1], vertices[:,2]
    A= np.column_stack([2*x, 2*y, 2*z, np.ones(len(x))])
    B= x**2 + y**2 + z**2

    result, _, _, _ = np.linalg.lstsq(A, B, rcond= None)
    center= result[:3]
    radius= float(np.sqrt(result[3] + np.sum(center**2)))

    return center, radius


def find_hip_center(femur_mesh: pv.PolyData,
                    proximal_fraction: float= 0.02) -> Tuple[np.ndarray, float]:

    """
    Locates the femoral head center (C_hip) using sphere fitting.

    Extracts the top "proximal fraction" of femur vertices by Z-coordinste (the femoral head region) and fits a sphere to them.

    Args:
        femur_mesh          : Proximal fraction
        proximal_fraction   : Fraction of bone height to use (default= 0.02)

    Returns:
        (center, radius) : C_hip coordinates and femoral head radius in mm
    """

    pts= np.array(femur_mesh.points)
    z_min, z_max= pts[:, 2].min(), pts[:, 2].max()
    threshold= z_max - proximal_fraction * (z_max - z_min)
    proximal_pts= pts[pts[:, 2] >= threshold]

    center, radius= fit_sphere_lstsq(proximal_pts)

    print(f"C_hip: [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.2f}] mm \n Radius: {radius:.1f} mm | {len(proximal_pts)} points used")

    return center, radius


def find_distal_femoral_center(femur_mesh: pv.PolyData,
                               distal_fraction: float= 0.08) -> np.ndarray:

    """
    Locates the distal femoral center (C_knee_f).
    Centroid of the bottom "distal fraction" of the femur by Z. 
    """

    pts= np.array(femur_mesh.points)
    z_min, z_max= pts[:, 2].min(), pts[:, 2].max()
    threshold= z_min + distal_fraction * (z_max - z_min)
    distal_pts= pts[pts[:, 2] <= threshold]
    center= np.mean(distal_pts, axis=0)     # mean across all the columns (x, y, z)

    print(f"C_knee_f: [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}] mm, \n {len(distal_pts)} points used")

    return center

def find_proximal_tibial_center(tibia_mesh: pv.PolyData,
                                proximal_fraction: float= 0.08) -> np.ndarray:
    """
    Locates the proximal tibial plateau center (C_knee_t).
    Area centroid of the top "proximal fraction" of the tibia by Z.
    """

    pts= np.array(tibia_mesh.points)
    z_min, z_max= pts[:, 2].min(), pts[:, 2].max()
    threshold= z_max - proximal_fraction * (z_max - z_min)
    proximal_pts= pts[pts[:, 2] >= threshold]

    center= np.mean(proximal_pts, axis= 0)

    print(f"C_knee_t: [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}] mm, \n {len(proximal_pts)} points used.")

    return center

def find_ankle_center(tibia_mesh: pv.PolyData,
                         distal_fraction: float= 0.10) -> np.ndarray:
    """
    Locates the ankle center (C_ankle) via the transmalleolar axis midpoint.
    Find the most medial (min x) and lateral (max x) points in the distal tibia region and return their midpoint.
    """

    pts= np.array(tibia_mesh.points)
    z_min, z_max= pts[:, 2].min(), pts[:, 2].max()
    threshold= z_min + distal_fraction * (z_max - z_min)
    distal_pts= pts[pts[:, 2] <= threshold]
    medial= distal_pts[np.argmin(distal_pts[:, 0])]
    lateral= distal_pts[np.argmax(distal_pts[:, 0])]

    center= (lateral + medial) / 2.0 
    print(f"C_ankle: [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}]")

    return center

def detect_all_landmarks(femur_mesh: pv.PolyData,
                         tibia_mesh: pv.PolyData) -> Dict[str, np.ndarray]:
    """
    Master function — detects all four landmarks from femur and tibia meshes.

    Args:
        femur_mesh : Processed left femur PyVista mesh
        tibia_mesh : Processed left tibia PyVista mesh

    Returns:
        Dictionary with keys C_hip, C_knee_f, C_knee_t, C_ankle.
        Each value is a (3,) NumPy array of [x, y, z] in mm.
    """
    print("="*55)
    print("  LANDMARK DETECTION")
    print("="*55)

    C_hip, R_hip = find_hip_center(femur_mesh)
    C_knee_f     = find_distal_femoral_center(femur_mesh)
    C_knee_t     = find_proximal_tibial_center(tibia_mesh)
    C_ankle      = find_ankle_center(tibia_mesh)

    landmarks = {
        "C_hip":    C_hip,
        "C_knee_f": C_knee_f,
        "C_knee_t": C_knee_t,
        "C_ankle":  C_ankle,
        "R_hip":    np.array([R_hip, 0, 0])   # store radius for reference
    }

    print("\nAll landmarks detected.")
    return landmarks


def visualize_landmarks(femur: pv.PolyData,
                        tibia: pv.PolyData,
                        landmark: Dict[str, np.ndarray]) -> None:
    
    """
    Renders 3D meshes, landmark spheres, and mechanical axis lines.
    """

    plotter = pv.Plotter()
    plotter.add_mesh(femur, color="#E8C9A0", opacity=0.35)
    plotter.add_mesh(tibia, color="#C8D8E8", opacity=0.35)
    
    colors = {"C_hip": "red", "C_knee_f": "orange", "C_knee_t": "cyan", "C_ankle": "blue"}

    for name, pt in landmark.items():
        if name != "R_hip":
            plotter.add_mesh(pv.Sphere(radius=8, center=pt), color=colors.get(name, "green"))
            
    plotter.add_mesh(pv.Line(landmark["C_hip"], landmark["C_knee_f"]), color="red", line_width=4)
    plotter.add_mesh(pv.Line(landmark["C_ankle"], landmark["C_knee_t"]), color="blue", line_width=4)
    plotter.add_title("Landmarks & Mechanical Axes")
    
    plotter.show()