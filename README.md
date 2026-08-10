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
```

---

## Project Structure

```text
OrthoPlanner3D/
│
├── data/
│   ├── raw/                  ← NIfTI CT files
│   └── segmentations/        ← TotalSegmentator outputs
│
├── meshes/
│   └── patient_01/           
│       ├── femur_left_raw.stl
│       ├── femur_left_processed.stl
│       ├── tibia_left_raw.stl
│       └── tibia_left_processed.stl
│
├── results/
│   └── patient_01/
│       └── kinematic_results.json 
│
├── src/
│   ├── series_merger.py
│   ├── mesh_generation.py
│   ├── bone_splitter.py     
│   ├── mesh_processor.py
│   ├── landmark_detector.py
│   ├── vector_alignment_solver.py
│   └── verify_segmentation.py
│
├── run_pipeline.py           
├── app.py                    ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Environment Setup

```bash
conda create -n orthoplanner2 python=3.10 -y
conda activate orthoplanner2
pip install torch==2.3.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## Pipeline Execution

**Step 1 — Convert DICOM to NIfTI** (terminal):
```bash
dcm2niix -o ./data/raw/ -f "patient_02_torso" -z y <path_to_torso_dicom/>
dcm2niix -o ./data/raw/ -f "patient_02_lower" -z y <path_to_lower_dicom/>
```

**Step 2 — Merge series**:
```python
from src.series_merger import merge_ct_series
merge_ct_series("./data/raw/patient_02_torso.nii.gz",
                "./data/raw/patient_02_lower.nii.gz",
                "./data/raw/patient_02_merged.nii.gz")
```

**Step 3 — Run TotalSegmentator** (terminal):
```bash
TotalSegmentator -i data/raw/patient_02_merged.nii.gz \
    -o data/segmentations/patient_02_total.nii.gz \
    --ml --fast --nr_thr_resamp 1 --nr_thr_saving 1

TotalSegmentator -i data/raw/patient_02_merged.nii.gz \
    -o data/segmentations/patient_02_appendicular.nii.gz \
    --ml --task appendicular_bones \
    --nr_thr_resamp 1 --nr_thr_saving 1
```

**Step 4 — Run the pipeline** (terminal):
```bash
python run_pipeline.py --patient patient_02 \
                       --femur_label 75 \
                       --tibia_label 2
```

**Step 5 — Launch the dashboard** (terminal):
```bash
streamlit run app.py
```

## Sample Output

### 3D Mesh Processing
3D bone meshes reconstructed from patient CT data using Marching Cubes, Laplacian smoothing, and quadric decimation.

| Bone | Raw Faces | Processed Faces | Reduction |
|------|-----------|-----------------|-----------|
| **Femur (left)** | 113,848 | 22,768 | ~80% |
| **Tibia (left)** | 78,848  | 15,768 | ~80% |

---

### Landmark Detection

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
![All 4 Landmarks and Mechanical Axes](./assets/fma_tma.png)

* **Femoral Mechanical Axis (FMA - Red Line):** Line vector connecting `C_hip` to `C_knee_f`.
* **Tibial Mechanical Axis (TMA - Blue Line):** Line vector connecting `C_ankle` to `C_knee_t`.
