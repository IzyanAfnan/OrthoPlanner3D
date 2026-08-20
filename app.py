import streamlit as st
import json 
import numpy as np
import pyvista as pv
import plotly.graph_objects as go
from pathlib import Path


# ── PAGE CONFIGURATION ─────────────────────────────────────────────

st.set_page_config(
    page_title= "Orthoplanner3D",
    page_icon= "🦴",
    layout= "wide",
    initial_sidebar_state= "expanded"
)


# ── CUSTOM CSS ─────────────────────────────────────────────────────
# This makes the dashboard look a bit nicer - dark header, coloured metrices, clean typography. Injected as raw CSS via st.markdown.

st.markdown("""
<style>
    /* Main header gradiet */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        text-align: center;
    }

    .main-header h1 {
        color: #e8c9a0;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }

    .main-header p {
        color: #a0b4c8;
        font-size: 1rem;
        margin: 0.3rem 0 0 0;
    }

    /* Metric cards */
    .metric-card{
        background: #1e2a3a;
        border: 1px solid #2d4a6b;
        border-radius: 8px;
        padding: 1rem 1.2 rem;
        margin-bottom: 0.8rem;
    }

    .metric-label {
        color: #7a9bb5;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        color: #e8c9a0;
        font-size: 1.8rem;
        font-weight: 700;
    }

    .metric-sub {
        color: #a0b4c8;
        font-size: 0.85rem;
        margin-top: 0.2rem;
    }
    
    /* Classification badge */
    .badge-normal {background: #1a4731; color:#4ade80;
                    padding: 4px 12px; border-radius: 20px;
                    font-weight: 600; font-size: 0.9rem; }
    .badge-varus  {background: #4a2512; color:#f97316;
                    padding: 4px 12px; border-radius: 20px;
                    font-weight: 600; font-size: 0.9rem; }
    .badge-valgus {background: #1a2a4a; color:#60a5fa;
                    padding: 4px 12px; border-radius: 20px;
                    font-weight: 600; font-size: 0.9rem; }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #0d1a2a;
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {
        color: #a0b4c8 !important;
    }

    /* Section headers */
    .section-header {
        color: #7a9bb5;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        border-bottom: 1px solid #1e3a5f;
        padding-bottom: 0.4rem;
        margin-bottom: 1.2rem 0 0.8rem 0;
    }

    /* Warning note */
    .merge-warning {
        background: #2d1f00;
        border-left: 3px solid #f59e0b;
        padding: 0.6rem 0.8rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.8rem;
        color: #fbbf24;
        margin-top: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ── HEADER ─────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1> OrthoPlanner3D </h1>
    <p> AI-Driven Preoperative Planning · Kinematic Alignment · Robotic Knee Arthroplasty </p>
</div>""", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Patient Selection")

    # Auto-detect available patients from results folder
    results_dir = Path("./results")
    available_patients = sorted([
        d.name for d in results_dir.iterdir()
        if d.is_dir() and (d / "kinematic_results.json").exists()
    ]) if results_dir.exists()  else []

    if not available_patients:
        st.error("No processed patient found. \n Run `python run_pipeline.py` first.")
        st.stop()

    selected_patient = st.selectbox(
        "Select patient",
        available_patients,
        format_func=lambda x: x.replace("_", " ").title()
    )

    st.markdown("---")
    st.markdown("### Virtual Resection Simulation")
    st.caption("Adjust to simulate tibial resection correction")

    resection_angle = st.slider(
        "Tibial plateau correction (°)",
        min_value= -10.0,
        max_value= 10.0,
        value= 0.0,
        step= 0.5,
        help= "Positive = varus correction, Negative = valgus correction"
    )

    st.markdown("---")
    st.markdown("### Pipeline Reference")
    st.caption("This dashboard pre-processed results. To add a new patient,  run the 5-step pipeline described in the README.")

    with st.expander("Quick command reference"):
        st.code(
            "# Run full pipeline for new patient\n"
            "python run_pipeline.py \\\n"
            "  --patient patient_02 \\\n"
            "  --femur_label 75 \\\n"
            "  --tibia_label 2",
            language="bash"
        )

# ── LOAD PATIENT DATA ──────────────────────────────────────────────
@st.cache_data
def load_results(patient_id: str) -> dict:
    """"Load Kinematic results JSON for selected patient."""
    path= Path(f"./results/{patient_id}/kinematic_results.json")
    with open(path) as f:
        return json.load(f)


# ── LOAD MESH  ──────────────────────────────────────────────
@st.cache_resource
def load_meshes(patient_id: str):
    """Load processed STL meshes. Returns None if files not found."""
    mesh_dir   = Path(f"./meshes/{patient_id}")
    femur_path = mesh_dir / "femur_left_processed.stl"
    tibia_path = mesh_dir / "tibia_left_processed.stl"
    femur = pv.read(str(femur_path)) if femur_path.exists() else None
    tibia = pv.read(str(tibia_path)) if tibia_path.exists() else None
    return femur, tibia

# ── BUILD PLOTLY BONE VIEWER  ──────────────────────────────────────────────
def build_plotly_bone_viewer(femur, tibia, landmarks_data,
                              v_FMA, v_TMA_corrected, resection_angle):
    """Builds interactive Plotly 3D figure with bones and axes."""

    fig = go.Figure()

    # Femur mesh
    if femur is not None:
        f_pts   = np.array(femur.points)
        f_faces = femur.faces.reshape(-1, 4)
        fig.add_trace(go.Mesh3d(
            x=f_pts[:,0], y=f_pts[:,1], z=f_pts[:,2],
            i=f_faces[:,1], j=f_faces[:,2], k=f_faces[:,3],
            color='#D4A96A', opacity=0.75,
            name='Femur (left)', showscale=False,
            flatshading=False,
            lighting=dict(ambient=0.5, diffuse=0.8,
                          specular=0.3, roughness=0.6),
            lightposition=dict(x=200, y=-300, z=500)
        ))

    # Tibia mesh
    if tibia is not None:
        t_pts   = np.array(tibia.points)
        t_faces = tibia.faces.reshape(-1, 4)
        fig.add_trace(go.Mesh3d(
            x=t_pts[:,0], y=t_pts[:,1], z=t_pts[:,2],
            i=t_faces[:,1], j=t_faces[:,2], k=t_faces[:,3],
            color='#7EB8D4', opacity=0.75,
            name='Tibia (left)', showscale=False,
            flatshading=False,
            lighting=dict(ambient=0.5, diffuse=0.8,
                          specular=0.3, roughness=0.6),
            lightposition=dict(x=200, y=-300, z=500)
        ))

    # Landmark coordinates
    C_hip    = np.array(landmarks_data["C_hip"])
    C_knee_f = np.array(landmarks_data["C_knee_f"])
    C_knee_t = np.array(landmarks_data["C_knee_t"])
    C_ankle  = np.array(landmarks_data["C_ankle"])

    # Femoral Mechanical Axis — always fixed, never changes
    fig.add_trace(go.Scatter3d(
        x=[C_hip[0],    C_knee_f[0]],
        y=[C_hip[1],    C_knee_f[1]],
        z=[C_hip[2],    C_knee_f[2]],
        mode='lines',
        line=dict(color='#ef4444', width=8),
        name='Femoral Axis (FMA)'
    ))

    # Tibial Mechanical Axis — updates when slider moves
    tibia_length       = float(np.linalg.norm(C_knee_t - C_ankle))
    C_knee_t_corrected = C_ankle + v_TMA_corrected * tibia_length

    tma_label = (f"Tibial Axis (TMA) — {resection_angle:+.1f}° correction"
                 if resection_angle != 0 else "Tibial Axis (TMA)")

    fig.add_trace(go.Scatter3d(
        x=[C_ankle[0], C_knee_t_corrected[0]],
        y=[C_ankle[1], C_knee_t_corrected[1]],
        z=[C_ankle[2], C_knee_t_corrected[2]],
        mode='lines',
        line=dict(color='#60a5fa', width=8,
                  dash='dash' if resection_angle != 0 else 'solid'),
        name=tma_label
    ))

    # Show original TMA as faint reference when correction is active
    if resection_angle != 0:
        fig.add_trace(go.Scatter3d(
            x=[C_ankle[0], C_knee_t[0]],
            y=[C_ankle[1], C_knee_t[1]],
            z=[C_ankle[2], C_knee_t[2]],
            mode='lines',
            line=dict(color='#1e3a5f', width=4, dash='dot'),
            name='TMA (original — reference)',
            opacity=0.5
        ))

    # Landmark spheres
    for pt, color, size, label in [
        (C_hip,    '#ef4444', 10, 'C_hip'),
        (C_knee_f, '#f97316',  8, 'C_knee_f'),
        (C_knee_t, '#22d3ee',  8, 'C_knee_t'),
        (C_ankle,  '#60a5fa',  8, 'C_ankle'),
    ]:
        fig.add_trace(go.Scatter3d(
            x=[pt[0]], y=[pt[1]], z=[pt[2]],
            mode='markers',
            marker=dict(size=size, color=color,
                        line=dict(color='white', width=1)),
            name=label
        ))

    # Camera: coronal (front) view
    bone_height = abs(float(C_hip[2]) - float(C_ankle[2]))
    knee_center = (C_knee_f + C_knee_t) / 2

    fig.update_layout(
        scene=dict(
            bgcolor='#0d1b2a',
            xaxis=dict(showgrid=False, showticklabels=False,
                       showline=False, zeroline=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False,
                       showline=False, zeroline=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False,
                       showline=False, zeroline=False, title=''),
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0, y=-(bone_height/400), z=0)
            )
        ),
        paper_bgcolor='#0d1b2a',
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            font=dict(color='#a0b4c8', size=11),
            bgcolor='rgba(13,27,42,0.8)',
            bordercolor='#1e3a5f',
            borderwidth=1,
            x=0, y=1
        ),
        height=680
    )

    return fig


results = load_results(selected_patient)

# ── APPLY RESECTION CORRECTION ─────────────────────────────────────
# Rotate TMA in coroal plane by resection_angle. This simulates the effect of tilting the tibia cut

v_TMA= np.array(results["v_TMA"])
correction_rad = np.radians(resection_angle)
cos_a, sin_a = np.cos(correction_rad), np.sin(correction_rad)

v_TMA_corrected = v_TMA.copy()
v_TMA_corrected[0] = v_TMA[0] * cos_a - v_TMA[2] * sin_a
v_TMA_corrected[2] = v_TMA[0] * sin_a + v_TMA[2] * cos_a
v_TMA_corrected = v_TMA_corrected / np.linalg.norm(v_TMA_corrected)

# Recompute coronal HKA with corrected TMA
v_FMA = np.array(results["v_FMA"])
v_FMA_cor = np.array([v_FMA[0], 0, v_FMA[2]])
v_TMA_cor = np.array([v_TMA_corrected[0], 0.0, v_TMA_corrected[2]])
v_FMA_cor = v_FMA_cor / np.linalg.norm(v_FMA_cor)
v_TMA_cor = v_TMA_cor / np.linalg.norm(v_TMA_cor)
dot= np.clip(np.dot(v_FMA_cor, v_TMA_cor), -1.0, 1.0)
hka_corrected = float(np.degrees(np.arccos(dot)))
deformity_corrected = 180.0 - hka_corrected

# Classification
original_hka        = results["hka_coronal"]
original_deformity  = results["deformity"]

if abs(deformity_corrected) <= 2.0:
    classification  = "Normal"
    badge_class     = "badge-normal"
elif deformity_corrected > 0:
    classification  = "Varus"
    badge_class     = "badge-varus"
else:
    classification  = "Valgus"
    badge_class     = "badge-valgus"


# ── MAIN CONTENT ───────────────────────────────────────────────────
col_3d, col_report = st.columns([3, 2])

with col_report:
    st.markdown("<div class='section-header'>Clinical Report</div>", unsafe_allow_html=True)

    patient_label = selected_patient.replace("_", " ").title()
    st.markdown(f"**Patient:**  {patient_label}")

    # HKA Metric
    delta = hka_corrected - original_hka
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">HKA Angle (Coronal)</div>
        <div class="metric-value">{hka_corrected:.1f}°</div>
        <div class="metric-sub">Original: {original_hka:.1f}°  
             {"▲" if delta > 0 else "▼"} {abs(delta):.1f}° from correction
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Deformity Metric
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Deformity</div>
        <div class="metric-value">{deformity_corrected:+.1f}°</div>
        <div class="metric-sub">± 2° = normal range</div>
    </div>
    """, unsafe_allow_html=True)

    # Classification Badge
    st.markdown(
        f'<span class="{badge_class}">{classification}</span>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-header">Bone Lengths</div>',
                unsafe_allow_html=True)

    bone_col1, bone_col2 = st.columns(2)
    with bone_col1:
        femur_len = float(np.array(results["femur_length_mm"]).flat[0])
        st.metric("Femur", f"{femur_len:.0f} mm")
    with bone_col2:
        tibia_len = float(np.array(results["tibia_length_mm"]).flat[0])
        st.metric("Tibia", f"{tibia_len:.0f} mm")

    st.markdown('<div class="section-header">Component Angles</div>',
                unsafe_allow_html=True)

    angle_col1, angle_col2 = st.columns(2)
    with angle_col1:
        st.metric("mLDFA", f"{results['mLDFA']:.1f}°",
                  help="Mechanical Lateral Distal Femoral Angle\nNormal: 85 - 90°")
    with angle_col2:
        st.metric("MPTA", f"{results['MPTA']:.1f}°",
                  help="Medial Proximal Tibial Angle\nNormal: 85 - 90°")

    if resection_angle != 0.0:
        st.markdown(
            f'<div class="merge-warning">⚕ Simulating {resection_angle:+.1f}° '
            f'tibial correction → projected HKA: {hka_corrected:.1f}°</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-header">Landmark Coordinates</div>',
                unsafe_allow_html=True)

    landmarks_data = {
        "C_hip":    results.get("C_hip", [0, 0, 0]),
        "C_knee_f": results.get("C_knee_f", [0, 0, 0]),
        "C_knee_t": results.get("C_knee_t", [0, 0, 0]),
        "C_ankle":  results.get("C_ankle", [0, 0, 0])
    }

    for name, coords in landmarks_data.items():
        coords = np.array(coords)
        st.caption(f"{name}: "
                   f"[{coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}] mm")

    st.markdown("---")
    st.caption(
        "⚠️ Research use only. Not validated for clinical deployment.  \n"
        "~1.6° systematic error expected from inter-series merge offset."
    )

with col_3d:
    st.markdown('<div class="section-header">3D Bone Reconstruction '
                '& Mechanical Axes</div>', unsafe_allow_html=True)

    # Load meshes for selected patient
    femur_mesh, tibia_mesh = load_meshes(selected_patient)

    if femur_mesh is None or tibia_mesh is None:
        st.warning(
            f"Mesh files not found in ./meshes/{selected_patient}/  \n"
            "Run `python run_pipeline.py` to generate them."
        )
    else:
        fig = build_plotly_bone_viewer(
            femur_mesh,
            tibia_mesh,
            landmarks_data,
            v_FMA,
            v_TMA_corrected,
            resection_angle
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": True,
                "modeBarButtonsToRemove": [
                    "select2d", "lasso2d", "autoScale2d"
                ],
                "displaylogo": False
            }
        )
        st.markdown("""
        <div style="display:flex; gap:1.5rem; margin-top:0.4rem;
                    font-size:0.82rem; color:#7a9bb5;">
            <span>🔴 Femoral Axis (fixed)</span>
            <span>🔵 Tibial Axis (updates with slider)</span>
        </div>
        """, unsafe_allow_html=True)

        if resection_angle != 0:
            st.caption(
                f"Dashed = corrected TMA at {resection_angle:+.1f}°  |  "
                f"Dotted = original TMA (reference)"
            ) 

# ── RESECTION SIMULATION SUMMARY ──────────────────────────────
if resection_angle != 0.0:
    st.markdown("---")
    st.markdown(
        '<div class="section-header">Virtual Resection Analysis</div>',
        unsafe_allow_html=True,
    )

    sim_col1, sim_col2, sim_col3 = st.columns(3)

    with sim_col1:
        st.metric(
            label="Pre-correction HKA",
            value=f"{original_hka:.1f}°",
            delta=None,
        )

    with sim_col2:
        st.metric(
            label="Post-correction HKA",
            value=f"{hka_corrected:.1f}°",
            delta=f"{hka_corrected - original_hka:+.1f}°",
        )

    with sim_col3:
        residual = abs(deformity_corrected)
        st.metric(
            label="Residual Deformity",
            value=f"{residual:.1f}°",
            delta=(
                "Within normal range"
                if residual <= 2.0
                else "Outside normal range"
            ),
        )

    # Progress bars showing alignment quality
    target = 180.0
    max_deviation = 15.0
    pre_norm = max(0.0, 1.0 - abs(original_hka - target) / max_deviation)
    post_norm = max(0.0, 1.0 - abs(hka_corrected - target) / max_deviation)

    st.caption("Alignment quality (100% = perfect neutral)")

    prog_col1, prog_col2 = st.columns(2)
    with prog_col1:
        st.caption("Before correction:")
        st.progress(pre_norm)
    with prog_col2:
        st.caption("After correction:")
        st.progress(post_norm)

    # Clinical recommendation banner
    if residual <= 2.0:
        st.success(
            f"✅ At **{resection_angle:+.1f}°** tibial correction, "
            f"projected alignment is within normal clinical range "
            f"(HKA = **{hka_corrected:.1f}°**)."
        )
    else:
        direction = "varus" if deformity_corrected > 0 else "valgus"
        st.warning(
            f"⚠️ At **{resection_angle:+.1f}°** correction, "
            f"a residual {direction} deformity of **{residual:.1f}°** remains. "
            f"Consider adjusting the correction angle."
        )

# ── PIPELINE INFORMATION ───────────────────────────────────────────
st.markdown("---")
with st.expander("📋 Pipeline Information — How This Result Was Computed"):
    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown("**Stage 1 — AI Segmentation**")
        st.caption(
            "TotalSegmentator (nnU-Net architecture) automatically "
            "identified the femur and tibia from the CT volume. "
            "No manual annotation was performed."
        )
        st.markdown("**Stage 2 — 3D Reconstruction**")
        st.caption(
            "Marching Cubes algorithm converted bone voxel masks into "
            "triangulated surface meshes. Laplacian smoothing removed "
            "staircase artifacts. Quadric decimation reduced face count "
            "by ~80% for real-time rendering."
        )

    with info_col2:
        st.markdown("**Stage 3 — Landmark Detection**")
        st.caption(
            "Femoral head center located via linear least-squares sphere "
            "fitting (error: <1mm). Knee and ankle centers found by "
            "geometric centroid and transmalleolar axis midpoint methods."
        )
        st.markdown("**Stage 4 — Kinematic Analysis**")
        st.caption(
            "Mechanical axes computed as normalised 3D vectors. "
            "HKA angle computed via dot product and projected onto "
            "the coronal plane to simulate a long-leg radiograph. "
            "mLDFA and MPTA computed from joint line vectors."
        )

    st.markdown("**Data Source**")
    st.caption(
        "CT data from the New Mexico Decedent Image Database (NMDID). "
        "Full-body post-mortem CT. Torso and lower extremity series "
        "merged along Z axis. See Known Limitations in README."
    )

with st.expander("ℹ️ About OrthoPlanner3D"):
    st.markdown("""
    OrthoPlanner3D is an open-source research pipeline for preoperative 
    planning in Total Knee Arthroplasty (TKA). It mirrors the clinical 
    workflow of systems like **Stryker Mako Total Knee 2.0**, where 
    CT-derived 3D bone models and mechanical axis calculations guide 
    bone resection planning.

    **Developed by:** Izyan Afnan  
    **Institution:** UPES Dehradun, B.Tech Biomedical Engineering  
    **GitHub:** github.com/IzyanAfnan/OrthoPlanner3D

    *For research and educational use only. Not validated for clinical deployment.*
    """)