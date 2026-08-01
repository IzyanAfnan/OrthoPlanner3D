# OrthoPlanner3D

AI-driven 3D lower-limb segmentation and orthopedic surgical planning pipeline using **TotalSegmentator** and **PyVista**.

---

## Clinical Motivation

Total Knee Arthroplasty (TKA) is one of the most common orthopedic surgeries worldwide. The most important decision in TKA planning is the bone resection angle — an error of even **4°** can cause uneven prosthetic wear and early implant failure.

This pipeline automates the measurement of the **Hip-Knee-Ankle (HKA)** angle from patient CT scans. It takes a raw CT scan and produces an interactive surgical planning dashboard.

---

## Pipeline Overview

```text
[CT Scan (.nii.gz)]
        │
        ▼
[Stage 1] AI Segmentation (TotalSegmentator)
        │
        ▼
[Stage 2] 3D Mesh Reconstruction (Marching Cubes + PyVista)
        │
        ▼
[Stage 3] Anatomical Landmark Detection (Sphere Fitting + Centroid)
        │
        ▼
[Stage 4] Mechanical Axis & HKA Angle Calculation (NumPy)
        │
        ▼
[Stage 5] Interactive Planning Dashboard (Streamlit + stpyvista)

---

## 📊 Sample Output

### Week 4 — 3D Mesh Processing
3D bone meshes reconstructed from patient CT data using Marching Cubes, Laplacian smoothing, and quadric decimation.

| Bone | Raw Faces | Processed Faces | Reduction |
|------|-----------|-----------------|-----------|
| **Femur (left)** | 113,848 | 22,768 | ~80% |
| **Tibia (left)** | 78,848  | 615768 | ~80% |

---

### Week 5 — Landmark Detection

Anatomical landmarks automatically detected from patient_01 
left limb meshes:

| Landmark | X (mm) | Y (mm) | Z (mm) | Method |
|----------|--------|--------|--------|--------|
| C_hip    | 111.7  | 220.0  | 1001.9 | Least-squares sphere fitting |
| C_knee_f | 101.6  | 241.0  | 590.7  | Distal femur centroid |
| C_knee_t | 87.8   | 228.6  | 571.9  | Tibial plateau centroid |
| C_ankle  | 103.4  | 193.5  | 219.2  | Transmalleolar axis midpoint |

* **Femoral head sphere-fitting error:** 0.63 mm (target: <3mm)

#### 3D Mechanical Axes Reconstruction
![All 4 Landmarks and Mechanical Axes](./fma_tma.png)

* **Femoral Mechanical Axis (FMA - Red Line):** Line vector connecting `C_hip` to `C_knee_f`.
* **Tibial Mechanical Axis (TMA - Blue Line):** Line vector connecting `C_ankle` to `C_knee_t`.