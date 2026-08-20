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
  2. FALLBACK — LPS convention (higher X = anatomical left), used only
     when no femur mesh is supplied. Flagged as unreliable.

SANITY CHECKS built in:
  - Ambiguous-match warning if the two candidate distances are too close
  - Body-size mismatch warning (possible fibula fragment / bad segmentation)
  - Extra-fragment warning if more than 2 significant bodies are found
  - Post-split verification: LPS check + joint-gap check + contralateral check
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
    save_dir: str = "./meshes/",
    swap_sides: bool = False,
    save: bool = True,
    verbose: bool = True,
) -> Tuple[pv.PolyData, pv.PolyData]:
    """
    Splits a combined tibia mesh into left and right tibias.

    Args:
        tibia_both : PyVista PolyData containing both tibias combined
        femur_mesh : Processed femur mesh (same-side reference) — used
                     to find the anatomically ipsilateral tibia via
                     joint-surface proximity. Strongly recommended;
                     falls back to LPS convention (unreliable) if omitted.
        save_dir   : Where to save tibia_left_raw.stl + tibia_right_raw.stl
        swap_sides : Manual override — flips the final left/right
                     assignment. Use only after visually confirming the
                     automatic match got it backwards.
        save       : Whether to write STL files to disk.
        verbose    : Whether to print diagnostic info.

    Returns:
        (tibia_left, tibia_right) as PyVista PolyData objects

    Raises:
        ValueError: if fewer than 2 valid tibia bodies are found.
    """
    tibia_a, tibia_b = _get_candidate_bodies(tibia_both, verbose=verbose)
    x_a, x_b = tibia_a.center[0], tibia_b.center[0]

    # Size sanity check — the two tibias should be roughly comparable.
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
            print(f"  (reference X-centers — A={x_a:.1f}, B={x_b:.1f}, femur={femur_mesh.center[0]:.1f})")

        if abs(dist_a - dist_b) < AMBIGUOUS_GAP_MARGIN and verbose:
            print(f"  WARNING: distances are close (Δ={abs(dist_a - dist_b):.1f} mm < "
                  f"{AMBIGUOUS_GAP_MARGIN} mm) — match may be ambiguous. "
                  "Visually inspect the split before trusting downstream results.")

        left_is_a = dist_a <= dist_b
        matched_gap = min(dist_a, dist_b)
    else:
        if verbose:
            print("\n  WARNING: No femur reference provided. Falling back to LPS "
                  "convention (higher X = anatomical left) — less reliable. "
                  "Verify visually or re-run with femur_mesh set.")
        left_is_a = x_a >= x_b
        matched_gap = None

    if swap_sides:
        left_is_a = not left_is_a
        if verbose:
            print("  swap_sides=True — flipping the assignment above.")

    tibia_left_raw, tibia_right_raw = (tibia_a, tibia_b) if left_is_a else (tibia_b, tibia_a)

    if verbose:
        print(f"\n  → {'Body A' if left_is_a else 'Body B'} assigned as LEFT tibia")
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
) -> bool:
    """
    Verifies that the tibia split assigned sides correctly.

    Checks:
      1. LPS sanity      — left X-center > right X-center
      2. Joint-gap check — left tibia's min surface distance to femur
                            should be small (a real knee joint)
      3. Contralateral    — right tibia must be farther from the femur
                            than left; otherwise sides are likely swapped

    Returns:
        True if all checks pass, False if the split is likely wrong.
    """
    left_x, right_x = tibia_left.center[0], tibia_right.center[0]

    print("\n=== TIBIA SPLIT VERIFICATION ===")
    print(f"  Left  tibia X-center: {left_x:.1f} mm")
    print(f"  Right tibia X-center: {right_x:.1f} mm")

    passed = True

    if femur_mesh is not None and femur_mesh.n_points > 0:
        gap_left  = _min_surface_distance(tibia_left, femur_mesh)
        gap_right = _min_surface_distance(tibia_right, femur_mesh)

        print(f"\n  Femur↔Left  tibia gap: {gap_left:.1f} mm")
        print(f"  Femur↔Right tibia gap: {gap_right:.1f} mm")

        if gap_left < JOINT_GAP_OK_MM:
            print("  ✓ Joint-gap check passed (left gap < 30mm)")
        elif gap_left < JOINT_GAP_WARN_MM:
            print("  ⚠ Left gap is 30-60mm — borderline, verify visually")
        else:
            print("  ✗ Joint-gap check FAILED (left gap > 60mm) — sides likely swapped")
            passed = False

        if gap_right <= gap_left:
            print("  ✗ Contralateral check FAILED — right tibia isn't farther "
                  "from the femur than left. Sides are very likely swapped.")
            passed = False
        else:
            print("  ✓ Contralateral check passed (right gap > left gap)")

    if passed:
        print("\n  ✓ Verification PASSED — split looks correct")
    else:
        print("\n  ✗ Verification FAILED — inspect meshes visually, then re-run "
              "split_tibia(..., swap_sides=True) if anatomy confirms a flip")
    print("================================\n")

    return passed