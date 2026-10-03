"""
src/bone_splitter.py

Splits a combined dual-tibia mesh (TotalSegmentator v2's `appendicular_bones`
task uses a single label for both tibias) into anatomically correct
left and right tibia meshes.

ASSIGNMENT LOGIC (in priority order):
  1. PRIMARY  — Nearest-surface-point distance to the femur.
     The distal femoral condyles sit almost directly on the proximal
     tibial plateau (typical joint space gap: a few mm to ~2cm).
     This is orientation-agnostic and robust to translational drift
     between CT acquisitions (e.g. torso/lower series merges), unlike
     comparing whole-bone centroid X-position.
  2. FALLBACK — X-center heuristic, used only when no femur mesh is
     supplied. Flagged as unreliable. Prefer always passing femur_mesh.

SANITY CHECKS built in:
  - Ambiguous-match warning if the two candidate distances are too close
  - Body-size mismatch warning (possible fibula fragment / bad segmentation)
  - Extra-fragment warning if more than 2 significant bodies are found
  - Post-split verification: joint-gap check + contralateral check
"""

from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


# ── Tunable thresholds ─────────────────────────────────────────────
MIN_FRAGMENT_RATIO   = 0.15   # discard bodies smaller than 15% of the largest body's point count
AMBIGUOUS_GAP_MARGIN = 15.0   # mm — if |dist_a - dist_b| < this, flag as ambiguous
JOINT_GAP_OK_MM      = 30.0   # mm — matched tibia should be within this of femur surface
JOINT_GAP_WARN_MM    = 60.0   # mm — beyond this, split is very likely wrong
SIZE_MISMATCH_RATIO  = 0.5    # warn if smaller candidate has <50% the points of the larger


def _get_candidate_bodies(tibia_both: pv.PolyData, verbose: bool = True) -> List[pv.PolyData]:
    """
    Splits a combined mesh into connected components and filters out
    small noise fragments left over from segmentation edges.

    Returns the 2 largest bodies that pass the size filter, sorted
    largest-to-smallest.
    """
    if tibia_both is None or tibia_both.n_points == 0:
        raise ValueError("tibia_both mesh is empty — check upstream mesh extraction.")

    bodies = tibia_both.split_bodies()
    bodies = [b.extract_surface() for b in bodies]
    bodies.sort(key=lambda b: b.n_points, reverse=True)

    if verbose:
        print(f"split_bodies found {len(bodies)} raw component(s):")
        for i, b in enumerate(bodies):
            print(f"  [{i}] {b.n_points} pts, X-center={b.center[0]:.1f}")

    if len(bodies) == 0:
        raise ValueError("No connected components found in tibia_both mesh.")

    largest_n = bodies[0].n_points
    kept = [b for b in bodies if b.n_points >= MIN_FRAGMENT_RATIO * largest_n]

    if len(kept) < len(bodies) and verbose:
        print(f"  Discarded {len(bodies) - len(kept)} fragment(s) below "
              f"{MIN_FRAGMENT_RATIO:.0%} of largest body's size.")

    if len(kept) < 2:
        raise ValueError(
            f"Expected 2 tibia bodies but only {len(kept)} passed the size filter. "
            "Check that the appendicular segmentation label contains both tibias, "
            "and that TotalSegmentator output isn't fragmented."
        )

    if len(kept) > 2 and verbose:
        print(f"  WARNING: {len(kept)} significant bodies found (expected 2). "
              "Keeping the 2 largest — verify segmentation quality; a fibula "
              "fragment may have leaked into this label.")

    return kept[:2]


def _min_surface_distance(mesh_a: pv.PolyData, mesh_b: pv.PolyData) -> float:
    """Minimum point-to-point distance (mm) between two mesh surfaces."""
    tree = cKDTree(mesh_b.points)
    dists, _ = tree.query(mesh_a.points, k=1)
    return float(dists.min())


def split_tibia(
    tibia_both: pv.PolyData,
    femur_mesh: Optional[pv.PolyData] = None,
    reference_is_left: bool = True,
    save_dir: str = "./meshes/",
    swap_sides: bool = False,
    save: bool = True,
    verbose: bool = True,
) -> Tuple[pv.PolyData, pv.PolyData]:
    """
    Splits a combined tibia mesh into left and right tibias.

    Args:
        tibia_both : PyVista PolyData containing both tibias combined
        femur_mesh : Processed femur mesh — used to find the anatomically
                     ipsilateral tibia via joint-surface proximity.
        reference_is_left : Which side femur_mesh actually is. Set False
                     when femur_mesh is the RIGHT femur (label 76), or the
                     ipsilateral tibia will be labeled "left" when it's
                     actually the right one.
        save_dir   : Where to save tibia_left_raw.stl + tibia_right_raw.stl
        swap_sides : Manual override — flips the final assignment. Use
                     only after visually confirming the automatic match
                     got it backwards.
        save       : Whether to write STL files to disk.
        verbose    : Whether to print diagnostic info.

    Returns:
        (tibia_left, tibia_right) as PyVista PolyData objects
    """
    tibia_a, tibia_b = _get_candidate_bodies(tibia_both, verbose=verbose)
    x_a, x_b = tibia_a.center[0], tibia_b.center[0]

    ratio = min(tibia_a.n_points, tibia_b.n_points) / max(tibia_a.n_points, tibia_b.n_points)
    if ratio < SIZE_MISMATCH_RATIO and verbose:
        print(f"  WARNING: candidate bodies differ significantly in size "
              f"({tibia_a.n_points} vs {tibia_b.n_points} pts, ratio={ratio:.2f}). "
              "One may be partial/fragmented rather than a full bone.")

    if femur_mesh is not None and femur_mesh.n_points > 0:
        dist_a = _min_surface_distance(tibia_a, femur_mesh)
        dist_b = _min_surface_distance(tibia_b, femur_mesh)

        if verbose:
            print(f"\n  Femur↔Body A min surface distance: {dist_a:.1f} mm")
            print(f"  Femur↔Body B min surface distance: {dist_b:.1f} mm")

        if abs(dist_a - dist_b) < AMBIGUOUS_GAP_MARGIN and verbose:
            print(f"  WARNING: distances are close (Δ={abs(dist_a - dist_b):.1f} mm < "
                  f"{AMBIGUOUS_GAP_MARGIN} mm) — match may be ambiguous. "
                  "Visually inspect the split before trusting downstream results.")

        ipsilateral_is_a = dist_a <= dist_b
        matched_gap = min(dist_a, dist_b)
    else:
        if verbose:
            print("\n  WARNING: No femur reference provided. Falling back to an "
                  "X-center heuristic — unreliable. Verify visually.")
        ipsilateral_is_a = x_a >= x_b
        matched_gap = None

    if swap_sides:
        ipsilateral_is_a = not ipsilateral_is_a
        if verbose:
            print("  swap_sides=True — flipping the assignment above.")

    tibia_ipsilateral, tibia_contralateral = (
        (tibia_a, tibia_b) if ipsilateral_is_a else (tibia_b, tibia_a)
    )

    if reference_is_left:
        tibia_left_raw, tibia_right_raw = tibia_ipsilateral, tibia_contralateral
    else:
        tibia_left_raw, tibia_right_raw = tibia_contralateral, tibia_ipsilateral

    if verbose:
        side_label = "LEFT" if reference_is_left else "RIGHT"
        print(f"\n  → Ipsilateral body assigned as {side_label} tibia "
              f"(reference femur was {side_label.lower()})")
        if matched_gap is not None:
            print(f"    (joint-gap distance: {matched_gap:.1f} mm)")

    if save:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        tibia_left_raw.save(str(out / "tibia_left_raw.stl"))
        tibia_right_raw.save(str(out / "tibia_right_raw.stl"))
        if verbose:
            print(f"\n  Saved tibia_left_raw.stl  (X-center: {tibia_left_raw.center[0]:.1f})")
            print(f"  Saved tibia_right_raw.stl (X-center: {tibia_right_raw.center[0]:.1f})")

    return tibia_left_raw, tibia_right_raw


def verify_tibia_split(
    tibia_left: pv.PolyData,
    tibia_right: pv.PolyData,
    femur_mesh: Optional[pv.PolyData] = None,
    reference_is_left: bool = True,
) -> bool:
    """
    Verifies that the tibia split assigned sides correctly.

    Args:
        reference_is_left : which side femur_mesh actually is (must match
                             what was passed to split_tibia).
    """
    left_x, right_x = tibia_left.center[0], tibia_right.center[0]

    print("\n=== TIBIA SPLIT VERIFICATION ===")
    print(f"  Left  tibia X-center: {left_x:.1f} mm")
    print(f"  Right tibia X-center: {right_x:.1f} mm")

    passed = True

    if femur_mesh is not None and femur_mesh.n_points > 0:
        ipsilateral_tibia, contralateral_tibia = (
            (tibia_left, tibia_right) if reference_is_left else (tibia_right, tibia_left)
        )
        gap_ipsi = _min_surface_distance(ipsilateral_tibia, femur_mesh)
        gap_contra = _min_surface_distance(contralateral_tibia, femur_mesh)

        print(f"\n  Femur↔Ipsilateral tibia gap:   {gap_ipsi:.1f} mm")
        print(f"  Femur↔Contralateral tibia gap: {gap_contra:.1f} mm")

        if gap_ipsi < JOINT_GAP_OK_MM:
            print("  ✓ Joint-gap check passed (ipsilateral gap < 30mm)")
        elif gap_ipsi < JOINT_GAP_WARN_MM:
            print("  ⚠ Ipsilateral gap is 30-60mm — borderline, verify visually")
        else:
            print("  ✗ Joint-gap check FAILED (ipsilateral gap > 60mm) — sides likely swapped")
            passed = False

        if gap_contra <= gap_ipsi:
            print("  ✗ Contralateral check FAILED — sides are very likely swapped.")
            passed = False
        else:
            print("  ✓ Contralateral check passed")

    if passed:
        print("\n  ✓ Verification PASSED — split looks correct")
    else:
        print("\n  ✗ Verification FAILED — inspect meshes visually, then re-run "
              "split_tibia(..., swap_sides=True) if anatomy confirms a flip")
    print("================================\n")

    return passed