# OrthoPlanner3D
AI-driven 3D lower-limb segmentation and orthopedic surgical planning pipeline using TotalSegmentator and PyVista.

---

## Clinical Motivation

Total Knee Arthroplasty (TKA) is one of the most common orthopedic surgeries worldwide. The most important decision in TKA planning is the bone resection angle - an error of even 4° can cause uneven prosthetic wear and early implant faliure.

This pipeline automates the measurement of Hip-Knee-Angle (HKA) angle from patient CT scans. It takes a raw CT scan and produces an interactive surgical planning dashboard.

---

## Pipeline Overview

CT Scan (.nii.gz or .nii)
↓
[Stage 1] AI Segmentation (TotalSegmentator)
↓
[Stage 2] 3D Mesh Reconstruction (Marching Cubes + PyVista)
↓
[Stage 3] Anatomical Landmark Detection (Sphere Fitting + Centroid)
↓
[Stage 4] Mechanical Axis & HKA Angle Calculation (NumPy)
↓
[Stage 5] Interactive Planning Dashboard (Streamlit + stpyvista)