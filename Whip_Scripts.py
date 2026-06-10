"""
Whip_Scripts.py
===============
QTM automation scripts for semi-automated labeling and quality control of
motion-capture data from the bullwhip target-striking paradigm.

Developed for Qualisys Track Manager (QTM 2023.3+, 32-bit Python 3.10).
Third-party packages were installed from pre-built wheels; see README.

Menu items and keyboard shortcuts are registered by add_menu() at the bottom,
which QTM calls automatically on script load.

Authors: Aleksei Krotov, Northeastern University Action Lab, 2023-2025.
"""

# ── standard library ─────────────────────────────────────────────────────────
import os
import sys
import csv
import math
import inspect
import subprocess
from datetime import datetime

# ── third-party ──────────────────────────────────────────────────────────────
import numpy as np

# ── QTM API ───────────────────────────────────────────────────────────────────
import qtm
import importlib
import helpers.menu_tools
import helpers.traj
importlib.reload(helpers.menu_tools)
importlib.reload(helpers.traj)
from helpers.menu_tools import add_menu_item, add_command

# Ensure the script's own directory is on sys.path so helpers can be found
_THIS_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)


# =============================================================================
# CONFIGURATION — edit these paths and constants for your project
# =============================================================================

# Path to the inter-marker distance threshold file.
# An example file is provided in the repository (example_VerifyDist_Thresholds.txt).
# Copy it to your project directory and adjust values as needed.
THRESHOLD_FILE_PATH = r"C:\QTM_Projects\MyProject\VerifyDist_Thresholds.txt"

# Path to the NPZ model file used by LabelAllInFrame_Train / _Label.
# Created automatically on first call to LabelAllInFrame_Train.
LABEL_MODEL_PATH = r"C:\QTM_Projects\MyProject\LabelFirstFrameModel.npz"

# Path to the marker-set XML file (exported from QTM's markerset editor).
MARKERSET_XML_PATH = r"C:\QTM_Projects\WholeBody&Whip&Target.txt"

# Maximum number of trajectory parts added per call to ConnectTrajPartsUsingSupport.
# Increase (or set to 0 for unlimited) when gaps are many and densely fragmented.
MAX_PARTS_ADDED = 30

# Timeout limits (seconds) for the gap-search outer loop and the per-part inner loop.
TIMEOUT_GAP_SEARCH = 30   # seconds for the outer gap-iteration loop
TIMEOUT_PART_ADD   = 2    # seconds for the per-candidate part-adding loop

# Minimum gap length (frames) to attempt filling. Short cosmetic gaps are skipped.
MIN_GAP_FRAMES = 4

# Distance tolerances used only for the inline geometry checks (handle colinearity,
# head L/R orientation). Sequential whip-marker and body-marker tolerances are
# stored in THRESHOLD_FILE_PATH so they can be edited without touching code.
HANDLE_COLINEARITY_TOL_MM = 55   # max point-to-line distance for HDR/HDL/w10/HD0

# Whip-marker sequential distance upper bounds (mm), indexed w1→w2, w2→w3, …, w9→w10.
DIST_MAX_WHIP_SEQ = [95, 135, 180, 225, 225, 225, 225, 225, 195, 80]

# Number of nearest neighbours used to build body-marker distance constraints.
N_BODY_CONSTRAINTS = 4

# Number of frames averaged around the current frame for the Train/Label model.
LABEL_FRAME_HALF_WIN = 40


# =============================================================================
# MARKER SETS
# =============================================================================

WHIP_NAMES   = ['w1','w2','w3','w4','w5','w6','w7','w8','w9','w10',
                 'HDR','HDL','HD0']
R_ARM_NAMES  = ['W_RWristIn','W_RWristOut','W_RHandOut',
                 'W_RElbowOut','W_RElbowIn','W_RArm',
                 'W_RShoulderTop','W_RShoulderBack']
L_ARM_NAMES  = ['W_LWristIn','W_LWristOut','W_LHandOut',
                 'W_LElbowOut','W_LElbowIn','W_LArm',
                 'W_LShoulderTop','W_LShoulderBack']
TORSO_NAMES  = ['W_Chest','W_SpineTop','W_BackL','W_BackR',
                 'W_WaistLFront','W_WaistLBack','W_WaistRBack','W_WaistRFront']
TARGET_NAMES = ['Target-1','Target-2','Target-3','Target-R','Target-L']
HEAD_NAMES   = ['W_HeadTop','W_HeadFront','W_HeadL','W_HeadR']
L_LEG_NAMES  = ['W_LThigh','W_LKneeOut','W_LKneeIn','W_LShin',
                 'W_LAnkleOut','W_LAnkleIn','W_LHeelBack',
                 'W_LForefootOut','W_LForefootIn','W_LToeTip']
R_LEG_NAMES  = ['W_RThigh','W_RKneeOut','W_RKneeIn','W_RShin',
                 'W_RAnkleOut','W_RAnkleIn','W_RHeelBack',
                 'W_RForefootOut','W_RForefootIn','W_RToeTip']

# Sequential distance thresholds for body-segment groups (0 = no check).
# Length must equal len(GROUP_NAMES) - 1 for each group.
DIST_MAX_R_ARM_SEQ   = [0, 0, 0, 130, 0, 0, 0]
DIST_MAX_L_ARM_SEQ   = [0, 0, 0, 120, 0, 0, 0]
DIST_MAX_TORSO_SEQ   = [300, 250, 230, 520, 260, 125, 250]
DIST_MAX_TARGET_SEQ  = [190, 235, 235, 0]
DIST_MAX_HEAD_SEQ    = [150, 160, 210]
DIST_MAX_L_LEG_SEQ   = [0, 0, 0, 0, 0, 0, 0, 0, 0]
DIST_MAX_R_LEG_SEQ   = [0, 0, 0, 0, 0, 0, 0, 0, 0]

# Non-adjacent pairs that need an explicit upper-bound (marker_index_A, marker_index_B, mm).
# Indices refer to the combined ALL_NAMES list built in Whip_CheckWhipM2MDistances.
DIST_MAX_NONADJACENT = [
    (37, 39, 400),   # Target-3 to Target-1
]


# =============================================================================
# SIMPLE UTILITY COMMANDS
# =============================================================================

def gtf(N):
    """Go to frame N (1-based QTM index)."""
    n_frames = qtm.gui.timeline.get_frame_count()
    N = int(N)
    if N < 1 or N > n_frames:
        print(f"Frame {N} out of range [1, {n_frames}]. Aborting.")
        return
    qtm.gui.timeline.set_current_frame(N - 1)

# Short alias for interactive console use
def f(N):
    gtf(N)


def Whip_Names_to_W_Names():
    """Prefix all non-Target, non-short-name trajectories with 'W_'."""
    series_ids = qtm.data.series._3d.get_series_ids()
    count = 0
    for sid in series_ids:
        name = qtm.data.object.trajectory.get_label(sid)
        if name and len(name) > 3 and "Target" not in name:
            qtm.data.object.trajectory.set_label(sid, "W_" + name)
            count += 1
    if count:
        print(f"Added prefix W_ to {count} marker names.")


def Whip_W_Names_to_Names():
    """Remove the 'W_' prefix from trajectory labels."""
    series_ids = qtm.data.series._3d.get_series_ids()
    count = 0
    for sid in series_ids:
        name = qtm.data.object.trajectory.get_label(sid)
        if name and len(name) > 3 and "Target" not in name:
            qtm.data.object.trajectory.set_label(sid, name.split("_", 1)[-1])
            count += 1
    if count:
        print(f"Removed prefix W_ from {count} marker names.")


def MarkerRemoveVirtualParts():
    """Delete all virtual (auto-filled) parts from the selected trajectory."""
    s = qtm.gui.selection.get_selections()
    if not s or len(s) == 0:
        print("Nothing is selected.")
        return
    if len(s) > 1:
        print("Select exactly one trajectory. Aborting.")
        return
    if "trajectory" not in s[0]["type"]:
        print("Selection is not a trajectory. Aborting.")
        return
    sid = s[0]["id"]
    name = qtm.data.object.trajectory.get_label(sid)
    parts = qtm.data.object.trajectory.get_parts(sid)
    virt_idx = [i for i, p in enumerate(parts) if p["type"] == "virtual"]
    if not virt_idx:
        print("No virtual parts found.")
        return
    try:
        qtm.data.object.trajectory.delete_parts(sid, virt_idx)
        print(f"Removed virtual parts {virt_idx} from {name}.")
    except Exception as e:
        print(f"Could not remove virtual parts from {name}: {e}")


# =============================================================================
# MARKER DATA ACCESS
# =============================================================================

def GetMarkerDataAll(sid):
    """
    Return all frames of a trajectory as an (N_frames, 3) float array.
    Frames with no data are NaN rows.
    Positions are rounded to the nearest mm (integer mm is the native QTM precision).
    """
    n_frames = qtm.gui.timeline.get_frame_count()
    pos = np.full((n_frames, 3), np.nan)
    raw = qtm.data.series._3d.get_samples(sid, {"start": 0, "end": n_frames - 1})
    for i, frame in enumerate(raw):
        if frame is not None:
            pos[i, :] = [round(v, 0) for v in frame["position"]]
    return pos


def getData_forNameList(name_list, id_by_name, n_frames):
    """
    Fetch trajectory data for a list of marker names.

    Returns
    -------
    data : ndarray, shape (n_frames, 3, len(name_list))
        Axis 2 corresponds to markers in name_list order. Missing markers → NaN slice.
    """
    data = np.full((n_frames, 3, len(name_list)), np.nan)
    for i, name in enumerate(name_list):
        if name not in id_by_name:
            print(f"  ID not found for '{name}'.")
            continue
        try:
            data[:, :, i] = GetMarkerDataAll(id_by_name[name])
        except Exception as e:
            print(f"  Could not fetch data for '{name}': {e}")
    return data


def _build_id_by_name():
    """Return {label: series_id} for all labeled trajectories in the current file."""
    id_by_name = {}
    for sid in qtm.data.series._3d.get_series_ids():
        name = qtm.data.object.trajectory.get_label(sid)
        if name is not None:
            id_by_name[name] = sid
    return id_by_name


# =============================================================================
# THRESHOLD FILE I/O  (plain-text CSV, human-editable)
# =============================================================================

def _load_threshold_file(path, n_markers, marker_names):
    """
    Load a threshold matrix from the plain-text threshold file.

    File format (see example_VerifyDist_Thresholds.txt in the repository):
        Header row
        "Marker Names and Indices" row
        One row per marker:  "index - name"
        "***Thresholds***" row
        N×N comma-separated integer matrix

    Returns
    -------
    matrix : ndarray (n_markers, n_markers) int
        Zero entries mean no threshold is set for that pair.
    names_in_file : list of str
    """
    matrix = np.zeros((n_markers, n_markers), dtype=int)
    names_in_file = []
    if not os.path.exists(path):
        return matrix, names_in_file
    with open(path, "r") as fh:
        reader = csv.reader(fh)
        reading_names = False
        last_row = 0
        for i, row in enumerate(reader):
            if not row:
                continue
            if "Marker Names" in row[0]:
                reading_names = True
                continue
            if "***Thresholds***" in row[0]:
                reading_names = False
                last_row = i
                break
            if reading_names:
                names_in_file.append(row[0].split(" - ")[1].strip())
    n_cols = len(names_in_file) if names_in_file else n_markers
    matrix = np.genfromtxt(path, delimiter=",", dtype=int,
                           skip_header=last_row + 1,
                           usecols=np.arange(n_cols))
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    # Pad or trim to (n_markers, n_markers) if sizes differ
    out = np.zeros((n_markers, n_markers), dtype=int)
    mn = min(n_markers, matrix.shape[0], matrix.shape[1])
    out[:mn, :mn] = matrix[:mn, :mn]
    return out, names_in_file


def _save_threshold_file(path, marker_names, matrix):
    """Write the threshold matrix and marker list to the plain-text file."""
    dt = datetime.now()
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"QTM VerifyDist thresholds. Last edited: {dt}"])
        w.writerow(["Marker Names and Indices"])
        for i, name in enumerate(marker_names):
            w.writerow([f"{i} - {name}"])
        w.writerow(["***Thresholds***"])
        for row in matrix:
            w.writerow(row)


# =============================================================================
# DISTANCE VERIFICATION  (VerifyDist / Whip_CheckWhipM2MDistances)
# =============================================================================

def VerifyDist():
    """Run full inter-marker distance verification and open the report."""
    Whip_CheckWhipM2MDistances()


def VerifyDist_Get():
    """Print current per-pair thresholds stored in the threshold file."""
    all_names = (WHIP_NAMES + R_ARM_NAMES + L_ARM_NAMES + TORSO_NAMES +
                 TARGET_NAMES + HEAD_NAMES + L_LEG_NAMES + R_LEG_NAMES)
    matrix, names_in_file = _load_threshold_file(THRESHOLD_FILE_PATH,
                                                   len(all_names), all_names)
    if not names_in_file:
        print(f"Threshold file not found: {THRESHOLD_FILE_PATH}")
        return
    print(f"Thresholds from {THRESHOLD_FILE_PATH}:")
    for i in range(len(names_in_file)):
        for j in range(i + 1, len(names_in_file)):
            v = matrix[i, j]
            if v:
                print(f"  {names_in_file[i]} — {names_in_file[j]}: {v} mm")


def VerifyDist_Set(mark_pair, new_val):
    """
    Update one threshold entry.

    Parameters
    ----------
    mark_pair : str  e.g. "w1-w2" or "W_RElbowOut-W_RHandOut"
    new_val   : int  New upper-bound distance in mm.
    """
    all_names = (WHIP_NAMES + R_ARM_NAMES + L_ARM_NAMES + TORSO_NAMES +
                 TARGET_NAMES + HEAD_NAMES + L_LEG_NAMES + R_LEG_NAMES)
    matrix, names_in_file = _load_threshold_file(THRESHOLD_FILE_PATH,
                                                   len(all_names), all_names)
    if not names_in_file:
        print(f"Threshold file not found. Run VerifyDist() first.")
        return

    parts = mark_pair.split("-")
    # Reconstruct "Target-N" names which contain a hyphen
    if len(parts) == 3:
        if parts[0] == "Target":
            m1, m2 = f"Target-{parts[1]}", parts[2]
        else:
            m1, m2 = parts[0], f"Target-{parts[2]}"
    elif len(parts) == 4:
        m1, m2 = f"Target-{parts[1]}", f"Target-{parts[3]}"
    else:
        m1, m2 = parts[0], parts[1]

    try:
        i1, i2 = names_in_file.index(m1), names_in_file.index(m2)
    except ValueError:
        print(f"Marker name not found. Available: {names_in_file}")
        return

    if i1 > i2:
        i1, i2 = i2, i1
    old_val = matrix[i1, i2]
    matrix[i1, i2] = new_val
    matrix[i2, i1] = new_val

    _save_threshold_file(THRESHOLD_FILE_PATH, names_in_file, matrix)
    print(f"Threshold {mark_pair}: {old_val} → {new_val} mm")


def Whip_CheckWhipM2MDistances():
    """
    Compute inter-marker distances for all marker groups, check against
    thresholds, flag violations, and write a timestamped report.
    """
    proj_dir = qtm.settings.directory.get_project_directory()
    id_by_name = _build_id_by_name()
    n_frames = qtm.gui.timeline.get_frame_count()
    print(f"\n{n_frames} frames. Fetching marker data...")

    # ── fetch data ────────────────────────────────────────────────────────────
    groups = [WHIP_NAMES, R_ARM_NAMES, L_ARM_NAMES, TORSO_NAMES,
              TARGET_NAMES, HEAD_NAMES, L_LEG_NAMES, R_LEG_NAMES]
    seq_thresholds = [DIST_MAX_WHIP_SEQ, DIST_MAX_R_ARM_SEQ, DIST_MAX_L_ARM_SEQ,
                      DIST_MAX_TORSO_SEQ, DIST_MAX_TARGET_SEQ, DIST_MAX_HEAD_SEQ,
                      DIST_MAX_L_LEG_SEQ, DIST_MAX_R_LEG_SEQ]

    all_names = []
    all_seq   = []
    for grp, seq in zip(groups, seq_thresholds):
        all_names += grp
        all_seq   += seq + [0]   # sentinel 0 between groups

    mark_data = np.full((n_frames, 3, len(all_names)), np.nan)
    for i, name in enumerate(all_names):
        if name in id_by_name:
            try:
                mark_data[:, :, i] = GetMarkerDataAll(id_by_name[name])
            except Exception as e:
                print(f"  Could not fetch '{name}': {e}")
    print("Done.")

    # ── build distance-threshold matrix ──────────────────────────────────────
    # Symmetric matrix; entry [i, j] = upper-bound distance (mm) for the pair,
    # 0 meaning "no threshold". Sequential thresholds are placed on the first
    # super-diagonal (pair i, i+1); explicit non-adjacent overrides elsewhere.
    dist_max = np.zeros((len(all_names), len(all_names)), dtype=int)
    for i, thr in enumerate(all_seq):
        if thr > 0 and i + 1 < len(all_names):
            dist_max[i, i + 1] = thr
            dist_max[i + 1, i] = thr
    for a, b, v in DIST_MAX_NONADJACENT:
        dist_max[a, b] = v
        dist_max[b, a] = v

    # Load / create customised thresholds from file
    matrix_file, names_in_file = _load_threshold_file(THRESHOLD_FILE_PATH,
                                                        len(all_names), all_names)
    if names_in_file and names_in_file == all_names and matrix_file.shape == dist_max.shape:
        print(f"Using thresholds from {THRESHOLD_FILE_PATH}.")
        dist_max = matrix_file
    else:
        print(f"Writing default thresholds to {THRESHOLD_FILE_PATH}.")
        _save_threshold_file(THRESHOLD_FILE_PATH, all_names, dist_max)

    # ── evaluate distances ────────────────────────────────────────────────────
    # Only store distances for the small set of markers used in relational checks.
    relations = [
        "w10;HDL>w10;HDR",
        "W_RElbowOut;W_RHandOut>W_RElbowOut;W_RWristOut",
    ]
    reduced_names = _extract_relation_markers(relations)
    dist_reduced  = np.full((len(reduced_names), len(reduced_names), n_frames),
                            999, dtype=np.int16)
    nan_reduced   = np.zeros((len(reduced_names), len(reduced_names), n_frames),
                             dtype=bool)

    lines_to_file  = []
    review_flags   = np.zeros(n_frames + 1, dtype=bool)
    review_markers = {}

    # Pairs whose distance must be stored for the relational checks even when
    # they carry no threshold (otherwise the threshold-skip below would drop them).
    relation_pairs = set()
    for rel in relations:
        for side in rel.split(">"):
            a, b = side.split(";")
            relation_pairs.add(frozenset((a, b)))

    skip_marks = set()
    for i1, m1 in enumerate(all_names):
        if i1 in skip_marks:
            continue
        for i2 in range(i1 + 1, len(all_names)):
            if i2 in skip_marks:
                continue
            m2 = all_names[i2]
            has_threshold = dist_max[i1, i2] > 0
            needs_storage = frozenset((m1, m2)) in relation_pairs
            if not has_threshold and not needs_storage:
                continue   # nothing to compute for this pair

            m1d = mark_data[:, :, i1]
            m2d = mark_data[:, :, i2]

            # Whole-marker NaN check
            if np.isnan(m1d).all():
                print(f"  Marker {m1} has no data.")
                skip_marks.add(i1)
                _mark_nan_in_reduced(m1, reduced_names, nan_reduced)
                break   # m1 invalid — no further pairs for this i1
            if np.isnan(m2d).all():
                print(f"  Marker {m2} has no data.")
                skip_marks.add(i2)
                _mark_nan_in_reduced(m2, reduced_names, nan_reduced)
                continue

            dist = np.linalg.norm(m2d - m1d, axis=1)
            if needs_storage:
                _store_reduced_dist(m1, m2, dist, reduced_names,
                                    dist_reduced, nan_reduced)

            if not has_threshold:
                continue

            _report_distance_pair(m1, m2, dist, dist_max[i1, i2],
                                  lines_to_file, review_flags, review_markers)

    # ── geometry checks ───────────────────────────────────────────────────────
    lines_to_file.append("***")
    _check_handle_colinearity(all_names, mark_data, lines_to_file,
                              review_flags, review_markers)
    lines_to_file.append("***")
    _check_head_orientation(all_names, mark_data, lines_to_file,
                            review_flags, review_markers)

    # ── relational checks ─────────────────────────────────────────────────────
    lines_to_file.append("***")
    lines_to_file.append("*** Relations (Med(IQR 5–95) for both sides) ***")
    _check_relations(relations, all_names, dist_reduced, nan_reduced,
                     reduced_names, lines_to_file, review_flags, review_markers)

    # ── unidentified long trajectories ────────────────────────────────────────
    unid_summary = _find_long_unidentified(n_frames)

    # ── overall review summary ────────────────────────────────────────────────
    lines_to_file.append("***")
    _append_review_summary(review_flags, review_markers, lines_to_file)
    lines_to_file.append("***")
    if unid_summary:
        lines_to_file.append("*** Long unidentified/discarded trajectories (≥10 frames): ***")
        lines_to_file.extend(unid_summary)
    else:
        lines_to_file.append("*** No unidentified trajectories ≥10 frames. ***")

    # ── save distances and report ─────────────────────────────────────────────
    np.savez(os.path.join(proj_dir, "!TempWhipRecordingValidationDistances.npz"),
             var1=dist_reduced, var2=nan_reduced, var3=reduced_names)

    report_path = os.path.join(proj_dir, "!TempWhipRecordingValidationReport.txt")
    dt = datetime.now()
    with open(report_path, "w") as fh:
        fh.write(dt.strftime("%b %d  %H:%M:%S") + "\n")
        for row in lines_to_file:
            fh.write(str(row) + "\n")
    subprocess.Popen(["start", "notepad.exe", report_path], shell=True)
    print("Validation finished.")


# ── internal helpers for Whip_CheckWhipM2MDistances ──────────────────────────

def _report_distance_pair(m1, m2, dist, threshold, lines, review_flags,
                          review_markers):
    """Compute statistics for a distance timeseries and append report lines."""
    finite = dist[np.isfinite(dist)]
    if len(finite) == 0:
        return
    d_min  = int(finite.min())
    d_max  = int(finite.max())
    d_med  = int(np.median(finite))
    d_iqr  = int(np.diff(np.percentile(finite, [5, 95]))) if len(finite) >= 4 else 0
    tag    = f"*** {m1}-{m2}:"
    summary = f"{tag} {d_min} - {d_med}({d_iqr}) - {d_max} - n={len(finite)}"

    exceed = np.where(dist > threshold)[0]
    if len(exceed):
        qtm_frames = (exceed + 1).tolist()
        grouped    = GroupFramesInRanges(qtm_frames)
        s = f"{summary} *** Exceeded {threshold} mm at:"
        lines.append(s); print(s)
        lines.append(grouped); print(grouped)
        review_flags[qtm_frames] = True
        for fr in qtm_frames:
            review_markers[fr] = (review_markers.get(fr, "") +
                                  f";{m1};{m2}").lstrip(";")
    elif d_max < 0.8 * threshold:
        s = f"{summary} *** Within {threshold} mm OK (consider lowering threshold)"
        lines.append(s); print(s)
    else:
        s = f"{summary} *** Within {threshold} mm OK"
        lines.append(s); print(s)


def _extract_relation_markers(relations):
    """Return a deduplicated list of marker names referenced in relation strings."""
    names = []
    for rel in relations:
        for side in rel.split(">"):
            names += side.split(";")
    return list(np.unique(names))


def _store_reduced_dist(m1, m2, dist, reduced_names, dist_reduced, nan_reduced):
    """Store distance timeseries into the reduced arrays used for relational checks."""
    if m1 not in reduced_names or m2 not in reduced_names:
        return
    i1 = reduced_names.index(m1)
    i2 = reduced_names.index(m2)
    valid = ~np.isnan(dist)
    dist_reduced[i1, i2, valid] = dist[valid].astype(np.int16)
    dist_reduced[i2, i1, valid] = dist[valid].astype(np.int16)
    nan_reduced[i1, i2, :] = ~valid
    nan_reduced[i2, i1, :] = ~valid


def _mark_nan_in_reduced(name, reduced_names, nan_reduced):
    if name in reduced_names:
        i = reduced_names.index(name)
        nan_reduced[i, :, :] = True
        nan_reduced[:, i, :] = True


def _check_handle_colinearity(all_names, mark_data, lines, review_flags,
                               review_markers):
    """
    Verify that HD0 lies close to the line defined by HDR–HDL and w10.
    Uses the point-to-line distance: |n·(A-C)| / |n|
    where n = (HDR-HDL) × (w10-HD0).
    """
    idx = [all_names.index(n) for n in ["w10", "HDR", "HDL", "HD0"]
           if n in all_names]
    if len(idx) < 4:
        lines.append("*** Handle colinearity check skipped (markers missing)")
        return
    w10, HDR, HDL, HD0 = [mark_data[:, :, i] for i in idx]
    e1 = HDR - HDL
    e2 = HD0 - w10
    n  = np.cross(e1, e2, axis=1)
    nn = np.linalg.norm(n, axis=1)
    dist = np.abs(np.sum(n * (HDR - w10), axis=1)) / np.where(nn > 0, nn, 1)

    d_med  = int(np.nanmedian(dist))
    d_iqr  = int(np.diff(np.nanpercentile(dist, [25, 75])))
    d_min  = int(np.nanmin(dist))
    d_max  = int(np.nanmax(dist))
    tol    = HANDLE_COLINEARITY_TOL_MM
    label  = f"Handle colinearity (HDR-HDL)×(w10-HD0): {d_min}-{d_med}({d_iqr})-{d_max}"

    exceed = np.where(dist > tol)[0]
    if len(exceed):
        qtm_frames = (exceed + 1).tolist()
        s = f"*** {label} *** Exceeded {tol} mm at:"
        lines.append(s); print(s)
        g = GroupFramesInRanges(qtm_frames)
        lines.append(g); print(g)
        review_flags[qtm_frames] = True
        for fr in qtm_frames:
            review_markers[fr] = (review_markers.get(fr, "") + ";Handle").lstrip(";")
    else:
        s = f"*** {label} *** Within {tol} mm OK"
        lines.append(s); print(s)


def _check_head_orientation(all_names, mark_data, lines, review_flags,
                             review_markers):
    """
    Verify that W_HeadL is to the right (+X) of the midpoint of
    W_HeadTop and W_HeadFront, and W_HeadR is to the left.
    (Lab X axis points leftward relative to a standing subject.)
    """
    needed = ["W_HeadTop", "W_HeadFront", "W_HeadL", "W_HeadR"]
    if not all(n in all_names for n in needed):
        lines.append("*** Head orientation check skipped (markers missing)")
        return
    HT, HF, HL, HR = [mark_data[:, :, all_names.index(n)] for n in needed]
    mid = 0.5 * (HT + HF)
    bad = ((HL[:, 0] < mid[:, 0]) & np.isfinite(HL[:, 0])) | \
          ((HR[:, 0] > mid[:, 0]) & np.isfinite(HR[:, 0]))
    exceed = np.where(bad)[0]
    if len(exceed):
        qtm_frames = (exceed + 1).tolist()
        s = "*** Head L/R markers may be swapped at:"
        lines.append(s); print(s)
        g = GroupFramesInRanges(qtm_frames)
        lines.append(g); print(g)
        review_flags[qtm_frames] = True
        for fr in qtm_frames:
            review_markers[fr] = (review_markers.get(fr, "") +
                                  ";W_HeadR;W_HeadL").lstrip(";")
    else:
        lines.append("*** Head L/R orientation: correct everywhere OK")
        print("*** Head L/R orientation: correct everywhere OK")


def _check_relations(relations, all_names, dist_reduced, nan_reduced,
                     reduced_names, lines, review_flags, review_markers):
    """Check that d(m1,m2) > d(m3,m4) for each relation string 'm1;m2>m3;m4'."""
    for rel in relations:
        left, right = rel.split(">")
        m1, m2 = left.split(";")
        m3, m4 = right.split(";")
        if not all(n in reduced_names for n in [m1, m2, m3, m4]):
            lines.append(f"*** Relation '{rel}' skipped (marker missing)")
            continue
        i1, i2 = reduced_names.index(m1), reduced_names.index(m2)
        i3, i4 = reduced_names.index(m3), reduced_names.index(m4)

        d12 = dist_reduced[i1, i2, :].astype(float)
        d12[nan_reduced[i1, i2, :]] = np.nan
        d34 = dist_reduced[i3, i4, :].astype(float)
        d34[nan_reduced[i3, i4, :]] = np.nan

        if np.all(d12 == 0) or np.all(d34 == 0):
            lines.append(f"*** Relation '{rel}' skipped (all-zero distances)")
            continue

        d12m = int(np.nanmedian(d12)); d12q = int(np.diff(np.nanpercentile(d12, [5, 95])))
        d34m = int(np.nanmedian(d34)); d34q = int(np.diff(np.nanpercentile(d34, [5, 95])))
        label = f"({m1}-{m2}) > ({m3}-{m4}) as {d12m}({d12q}) > {d34m}({d34q})"
        exceed = np.where(d12 < d34)[0]
        if len(exceed):
            qtm_frames = (exceed + 1).tolist()
            s = f"*** {label}: *** Violated at:"
            lines.append(s); print(s)
            g = GroupFramesInRanges(qtm_frames)
            lines.append(g); print(g)
            review_flags[qtm_frames] = True
            for fr in qtm_frames:
                review_markers[fr] = (review_markers.get(fr, "") +
                                      f";{m1};{m2};{m3};{m4}").lstrip(";")
        else:
            s = f"*** {label}: *** Satisfied everywhere OK"
            lines.append(s); print(s)


def _find_long_unidentified(n_frames, min_len=10, top_n=10):
    """Find the longest unidentified trajectories (≥ min_len valid frames)."""
    result = []
    lengths = {}
    for sid in qtm.data.series._3d.get_series_ids():
        if qtm.data.object.trajectory.get_label(sid) is not None:
            continue
        data = GetMarkerDataAll(sid)
        valid = np.where(~np.isnan(data).any(axis=1))[0]
        if len(valid) >= min_len:
            lengths[sid] = (valid[0] + 1, valid[-1] + 1, len(valid))
    for sid, (f0, f1, n) in sorted(lengths.items(), key=lambda x: -x[1][2])[:top_n]:
        result.append(f"{f0}-{f1}  (len={n})  Traj ID {sid}")
    return result


def _append_review_summary(review_flags, review_markers, lines):
    """Group flagged frames and append a final review section to lines."""
    if not np.any(review_flags):
        s = "*** No problematic frames found ***"
        lines.append(s); print(s)
        return
    s = "*** Overall, review the following frames: ***"
    lines.append(s); print(s)
    frames = np.where(review_flags)[0].tolist()
    grouped_with_markers = _group_frames_with_markers(frames, review_markers)
    for row in grouped_with_markers:
        lines.append(row); print(row)


def _group_frames_with_markers(frames, review_markers):
    """Group consecutive frames and collect unique marker names per range."""
    if not frames:
        return []
    result = []
    seg_start = frames[0]
    seg_end   = frames[0]
    for fr in frames[1:]:
        if fr - seg_end > 1:
            result.append(_format_frame_range(seg_start, seg_end, review_markers))
            seg_start = fr
        seg_end = fr
    result.append(_format_frame_range(seg_start, seg_end, review_markers))
    return result


def _format_frame_range(f_lo, f_hi, review_markers):
    dur  = f_hi - f_lo + 1
    marks = set()
    for fr in range(f_lo, f_hi + 1):
        if fr in review_markers:
            marks.update(review_markers[fr].split(";"))
    label = f"{f_lo}-{f_hi} ({dur})" if dur > 1 else str(f_lo)
    return f"{label.ljust(15)}: {', '.join(sorted(marks))}"


# =============================================================================
# NAVIGATE TO NEXT PROBLEMATIC INTERVAL
# =============================================================================

def Verify_GoToNext():
    """
    Jump to the frame just before the next flagged interval in the most
    recent validation report.  Reports intervals of ≥5 frames only.
    """
    proj_dir  = qtm.settings.directory.get_project_directory()
    report    = os.path.join(proj_dir, "!TempWhipRecordingValidationReport.txt")
    id_by_name = _build_id_by_name()
    n_frames   = qtm.gui.timeline.get_frame_count()

    if not os.path.exists(report):
        print("Validation report not found.")
        return

    # Warn if report is stale
    age_min = (datetime.now() - datetime.fromtimestamp(
        os.path.getmtime(report))).total_seconds() / 60
    if age_min > 15:
        print(f"Warning: report is {age_min:.0f} min old — re-run VerifyDist().")

    all_names = (WHIP_NAMES + R_ARM_NAMES + L_ARM_NAMES + TORSO_NAMES +
                 TARGET_NAMES + HEAD_NAMES + L_LEG_NAMES + R_LEG_NAMES)
    threshold_matrix, threshold_names = _load_threshold_file(
        THRESHOLD_FILE_PATH, len(all_names), all_names)

    # Parse report for flagged intervals
    prob_first, prob_dur, prob_lines = [], [], []
    with open(report, "r") as fh:
        reader  = csv.reader(fh)
        reading = False
        for row in reader:
            if not row:
                continue
            if "*** Overall" in row[0]:
                reading = True
                continue
            if reading and "***" in row[0]:
                reading = False
            if not reading:
                continue
            if "(" not in row[0]:
                continue
            dur = int(row[0].split("(")[1].split(")")[0])
            if dur <= 4:
                continue
            f0 = int(row[0].split("-")[0])
            prob_first.append(f0)
            prob_dur.append(dur)
            # Enrich two-marker lines with actual peak distance
            if len(row) == 2:
                m1 = row[0].split(": ")[1].strip()
                m2 = row[1].strip()
                names = [m1, m2]
                md = getData_forNameList(names, id_by_name, n_frames)
                dist = np.sqrt(np.sum((md[:, :, 0] - md[:, :, 1]) ** 2, axis=1))
                rng  = np.arange(f0, f0 + dur)
                d_max = np.round(np.nanmax(dist[rng]), 0)
                try:
                    i1 = threshold_names.index(m1)
                    i2 = threshold_names.index(m2)
                    thr = max(threshold_matrix[i1, i2],
                              threshold_matrix[i2, i1])
                except (ValueError, IndexError):
                    thr = "?"
                prob_lines.append(
                    f"{row[0].split(': ')[0]}: {m1}-{m2}: {d_max} mm exceeded {thr} mm")
            else:
                prob_lines.append("".join(row))

    if not prob_first:
        print("No intervals ≥5 frames found in the report.")
        return

    i_cur = qtm.gui.timeline.get_current_frame()
    arr   = np.array(prob_first)
    arr[arr - i_cur < 0] = i_cur   # mask past intervals
    idx   = next((i for i, d in enumerate(arr - i_cur) if d != 0), None)
    if idx is not None:
        gtf(arr[idx] - 1)
        print(prob_lines[idx])
    else:
        print("No more problematic intervals after the current frame.")


# =============================================================================
# ONLINE POSITION MODEL FOR FIRST-FRAME LABELING
# =============================================================================

def _welford_update(n_old, mean_old, var_old, new_sample):
    """
    Online mean and variance update (Welford's algorithm).
    Handles NaN in new_sample by returning the existing statistics unchanged.
    """
    if np.any(np.isnan(new_sample)):
        return n_old, mean_old, var_old
    n_new  = n_old + 1
    mean_new = (mean_old * n_old + new_sample) / n_new if not np.any(np.isnan(mean_old)) \
               else new_sample
    delta  = new_sample - mean_old if not np.any(np.isnan(mean_old)) else np.zeros_like(new_sample)
    delta2 = new_sample - mean_new
    var_new = (var_old * (n_old - 1) + delta * delta2) / max(n_new - 1, 1) \
              if not np.any(np.isnan(var_old)) else np.zeros_like(new_sample)
    return n_new, np.round(mean_new, 0), np.round(np.maximum(var_new, 0), 0)


def _read_markerset_names():
    """Parse marker names from the QTM markerset XML file."""
    names = []
    if not os.path.exists(MARKERSET_XML_PATH):
        print(f"Markerset file not found: {MARKERSET_XML_PATH}")
        return names
    with open(MARKERSET_XML_PATH, "r") as fh:
        for line in fh:
            if "<Name>" in line:
                names.append(line.split("<Name>")[1].split("</Name>")[0])
    return names


def LabelAllInFrame_Train():
    """
    Update the position model from data near the current frame.

    The model stores for each marker: mean position, variance, and N
    (number of training recordings contributing to that estimate).
    Also stores a symmetric N×N matrix of inter-marker distance means.
    All data are saved to LABEL_MODEL_PATH (.npz).
    """
    marker_names = _read_markerset_names()
    if not marker_names:
        return
    n   = len(marker_names)
    i_cur = qtm.gui.timeline.get_current_frame()
    n_frames = qtm.gui.timeline.get_frame_count()
    half = LABEL_FRAME_HALF_WIN
    frame_range = np.clip(np.arange(i_cur - half, i_cur + half), 0, n_frames - 1)

    # Collect data for this training recording
    pos_mean = {}   # name → (3,) mean position
    pos_data = {}   # name → (window, 3) array
    for name in marker_names:
        sid = qtm.data.object.trajectory.find_trajectory(name)
        if sid is None:
            continue
        raw = qtm.data.series._3d.get_samples(
            sid, {"start": frame_range[0].item(), "end": frame_range[-1].item()})
        arr = np.full((len(raw), 3), np.nan)
        for i, fr in enumerate(raw):
            if fr is not None:
                arr[i, :] = fr["position"]
        pos_data[name] = arr
        pos_mean[name] = np.nanmean(arr, axis=0)

    # Pairwise mean distances in this recording
    dist_pose = np.full((n, n), np.nan)
    for i1, n1 in enumerate(marker_names):
        for i2, n2 in enumerate(marker_names):
            if i2 <= i1 or n1 not in pos_data or n2 not in pos_data:
                continue
            d = np.linalg.norm(pos_data[n1] - pos_data[n2], axis=1)
            dist_pose[i1, i2] = np.nanmean(d)
            dist_pose[i2, i1] = dist_pose[i1, i2]

    # Load existing model or initialise
    if os.path.exists(LABEL_MODEL_PATH):
        m = np.load(LABEL_MODEL_PATH, allow_pickle=True)
        pos_N   = m["pos_N"];   pos_M = m["pos_M"];   pos_S = m["pos_S"]
        dist_N  = m["dist_N"];  dist_M = m["dist_M"]
    else:
        pos_N  = np.zeros((n, 1), dtype=int)
        pos_M  = np.full((n, 3), np.nan)
        pos_S  = np.full((n, 3), np.nan)
        dist_N = np.zeros((n, n), dtype=int)
        dist_M = np.full((n, n), np.nan)

    # Update model
    for i, name in enumerate(marker_names):
        if name not in pos_mean or np.any(np.isnan(pos_mean[name])):
            continue
        pos_N[i], pos_M[i], pos_S[i] = _welford_update(
            pos_N[i, 0], pos_M[i], pos_S[i], pos_mean[name])
        for i2, n2 in enumerate(marker_names):
            if i2 <= i or np.isnan(dist_pose[i, i2]):
                continue
            dist_N[i, i2], dist_M[i, i2], _ = _welford_update(
                dist_N[i, i2], dist_M[i, i2], np.array([0.0]), dist_pose[i, i2])
            dist_M[i2, i] = dist_M[i, i2]
            dist_N[i2, i] = dist_N[i, i2]

    np.savez(LABEL_MODEL_PATH, pos_N=pos_N, pos_M=pos_M, pos_S=pos_S,
             dist_N=dist_N, dist_M=dist_M, marker_names=marker_names)
    print("Model updated and saved to", LABEL_MODEL_PATH)


def LabelAllInFrame_Label():
    """
    Assign unidentified trajectories near the current frame to known markers.

    Uses a diagonal Gaussian probability density evaluated at the observed
    position of each unidentified trajectory; assigns each candidate to
    its highest-scoring marker and calls qtm.data.object.trajectory.move_parts.
    """
    if not os.path.exists(LABEL_MODEL_PATH):
        print(f"Model file not found: {LABEL_MODEL_PATH}. Run LabelAllInFrame_Train() first.")
        return
    m = np.load(LABEL_MODEL_PATH, allow_pickle=True)
    pos_M = m["pos_M"]; pos_S = m["pos_S"]
    marker_names = list(m["marker_names"])
    print(f"Model loaded from {LABEL_MODEL_PATH}")

    i_cur    = qtm.gui.timeline.get_current_frame()
    n_frames = qtm.gui.timeline.get_frame_count()
    half     = LABEL_FRAME_HALF_WIN
    frame_range = np.clip(np.arange(i_cur - half, i_cur + half), 0, n_frames - 1)

    # Gather unidentified trajectories present in the window
    unid_ids, unid_pos = [], []
    for sid in qtm.data.series._3d.get_series_ids():
        if qtm.data.object.trajectory.get_label(sid) is not None:
            continue
        raw = qtm.data.series._3d.get_samples(
            sid, {"start": frame_range[0].item(), "end": frame_range[-1].item()})
        arr = np.full((len(raw), 3), np.nan)
        for i, fr in enumerate(raw):
            if fr is not None:
                arr[i, :] = fr["position"]
        if not np.all(np.isnan(arr)):
            unid_ids.append(sid)
            unid_pos.append(np.nanmean(arr, axis=0))
    print(f"Unidentified trajectories in window: {len(unid_ids)}")

    # Build lookup for known-label trajectory IDs
    label_to_id = {n: qtm.data.object.trajectory.find_trajectory(n)
                   for n in marker_names}

    # Score and assign
    for sid, obs in zip(unid_ids, unid_pos):
        scores = np.array([_mvn_score(obs, pos_M[i], pos_S[i])
                           for i in range(len(marker_names))])
        best = marker_names[int(np.argmax(scores))]
        dest = label_to_id.get(best)
        if dest is None:
            continue
        try:
            qtm.data.object.trajectory.move_parts(sid, dest)
        except Exception as e:
            print(f"  Could not assign traj {sid} → {best}: {e}")


def _mvn_score(x, mean, sd):
    """
    Diagonal multivariate Gaussian log-probability (unnormalised).
    A minimum SD of 10% of the largest dimension is applied for stability.
    """
    if np.any(np.isnan(mean)):
        return -np.inf
    s = np.maximum(np.abs(sd), 0.1 * np.nanmax(np.abs(sd)) + 1e-6) * 10
    return -0.5 * np.sum(((x - mean) / s) ** 2)


# =============================================================================
# GAP FILLING  (ConnectTrajPartsUsingSupport / AddJoin*)
# =============================================================================

def AddJoinWhip():
    """Fill gaps in all whip markers (w1–w10) in reverse distal-to-proximal order."""
    for name in WHIP_NAMES[:10][::-1]:
        print(f"\n*** Auto-filling {name} ***")
        sid = qtm.data.object.trajectory.find_trajectory(name)
        if sid is not None:
            ConnectTrajPartsUsingSupport(sid)


def AddJoinTraj(max_parts=0):
    """Fill gaps in a trajectory selected via the console."""
    ConnectTrajPartsUsingSupport([], max_parts)


def AddJoinTrajButton(sid=[], max_parts=[], debug=False):
    """Fill gaps in the currently selected trajectory (menu / hotkey entry point)."""
    ConnectTrajPartsUsingSupport(sid, max_parts, debug)


def ConnectTrajPartsUsingSupport(sid=[], max_parts=[], debug=False):
    """
    Automatically fill gaps in a labeled trajectory by recruiting
    unidentified trajectory segments that satisfy biomechanical
    distance constraints.

    Algorithm (per gap, in order of gap start frame):
      1. For each unidentified trajectory that has data in the gap,
         check whether its median distance to the anchor marker's
         neighbors is within learned bounds.
      2. Attempt to move all qualifying parts at once; fall back to
         part-by-part; use SwapPartsFixOverlap when an overlap prevents
         a direct move.
    """
    n_frames = qtm.gui.timeline.get_frame_count()
    i_cur    = qtm.gui.timeline.get_current_frame()

    # Resolve the trajectory to work on
    if not sid:
        s = qtm.gui.selection.get_selections()
        if not s or len(s) == 0:
            print("Nothing selected."); return
        if len(s) > 1:
            print("Select exactly one trajectory."); return
        if "trajectory" not in s[0]["type"]:
            print("Selection is not a trajectory."); return
        sid = s[0]["id"]

    id_name = qtm.data.object.trajectory.get_label(sid)

    # Determine max parts
    if isinstance(max_parts, int) and max_parts == 0:
        n_max = 999999
    elif not max_parts:
        n_max = MAX_PARTS_ADDED
    else:
        n_max = int(max_parts)
    print(f"Filling gaps in {id_name} (max {n_max} parts).")

    # Decide constraint type and load reference data
    is_whip = id_name in WHIP_NAMES[:10]
    all_traj = WHIP_NAMES if is_whip else (
        WHIP_NAMES + R_ARM_NAMES + L_ARM_NAMES + TORSO_NAMES +
        L_LEG_NAMES + R_LEG_NAMES + TARGET_NAMES + HEAD_NAMES)

    data_by_name = {}
    for name in all_traj:
        t = qtm.data.object.trajectory.find_trajectory(name)
        if t is not None:
            data_by_name[name] = GetMarkerDataAll(t)
            if t == sid:
                data_self = data_by_name[name]

    fill_before = np.count_nonzero(~np.isnan(data_self).any(axis=1)) / n_frames * 100

    if np.isnan(data_self[i_cur, 0]):
        print("Move cursor to a frame with data, just before the gap. Aborting.")
        return

    # Build constraints
    if is_whip:
        prox_data, prox2_data, dist_data, ub_prox, ub_prox2, ub_dist = \
            AutoAdd_Whip_GetProxDist(data_by_name, WHIP_NAMES[:10],
                                     WHIP_NAMES.index(id_name), DIST_MAX_WHIP_SEQ)
    else:
        dist_med, dist_iqr, dist_min, dist_max = \
            AutoAdd_Body_ClosestConstraints(data_by_name, all_traj,
                                           data_self, id_name, N_BODY_CONSTRAINTS)

    # Pre-fetch all unidentified trajectory data (avoids repeated API calls)
    unid_ids = [s for s in qtm.data.series._3d.get_series_ids()
                if qtm.data.object.trajectory.get_label(s) is None]
    print(f"Caching {len(unid_ids)} unidentified trajectories...")
    unid_data    = np.full((n_frames, 3, len(unid_ids)), 9999, dtype=int)
    unid_nan     = np.full((n_frames, len(unid_ids)), True, dtype=bool)
    for k, u_id in enumerate(unid_ids):
        raw = GetMarkerDataAll(u_id)
        valid = ~np.isnan(raw).any(axis=1)
        unid_nan[:, k]          = ~valid
        unid_data[valid, :, k]  = raw[valid, :].astype(int)

    # Gap-filling loop
    n_added = 0
    gap_ranges = [g for g in qtm.data.series._3d.get_gap_ranges(sid)
                  if g["start"] >= i_cur and
                  (g["end"] - g["start"] + 1) >= MIN_GAP_FRAMES]

    t_start = datetime.now()
    for ig, gap in enumerate(gap_ranges):
        if (datetime.now() - t_start).total_seconds() > TIMEOUT_GAP_SEARCH:
            print(f"Gap search timed out ({TIMEOUT_GAP_SEARCH} s).")
            break
        gap_ind = np.arange(gap["start"], gap["end"] + 1)
        print(f"Gap {ig + 1}/{len(gap_ranges)}: frames {gap['start']}–{gap['end']}")

        # Find candidate unidentified trajectories for this gap
        candidates = {}
        for k, u_id in enumerate(unid_ids):
            d_gap = np.full((len(gap_ind), 3), np.nan)
            valid = ~unid_nan[gap_ind, k]
            if not valid.any():
                continue
            d_gap[valid, :] = unid_data[gap_ind[valid], :, k]
            if np.isnan(d_gap).all():
                continue

            if is_whip:
                skip = AutoAdd_Whip_VerifyConstraints(
                    ub_prox, ub_prox2, ub_dist,
                    gap_ind, d_gap, prox_data, prox2_data, dist_data)
            else:
                skip = AutoAdd_Body_VerifyConstraints(
                    dist_med, gap_ind, data_by_name, d_gap,
                    N_BODY_CONSTRAINTS, dist_iqr, dist_min, dist_max, id_name)
            if skip:
                continue

            first_valid = np.where(~np.isnan(d_gap).any(axis=1))[0]
            if len(first_valid):
                candidates[u_id] = first_valid[0]

        if not candidates:
            print("  No suitable candidates found.")
            continue

        # Process candidates in order of earliest valid frame
        for u_id in sorted(candidates, key=candidates.get):
            if n_added >= n_max:
                print(f"Reached limit of {n_max} parts added.")
                break
            n_added += _try_add_candidate(u_id, sid, id_name, gap_ind,
                                          data_self, n_added, n_max, debug)

    data_self = GetMarkerDataAll(sid)
    fill_after = np.count_nonzero(~np.isnan(data_self).any(axis=1)) / n_frames * 100
    if fill_after > fill_before:
        print(f"Fill ratio: {fill_before:.1f}% → {fill_after:.1f}%")
    else:
        print("No segments added.")


def _try_add_candidate(u_id, sid, id_name, gap_ind, data_self,
                       n_added, n_max, debug):
    """
    Attempt to move parts from u_id to sid for the given gap_ind.
    Returns the number of parts successfully added.
    """
    try:
        parts = [p for p in qtm.data.object.trajectory.get_parts(u_id)
                 if p["type"] == "measured"]
    except RuntimeError as e:
        print(f"  Cannot get parts for traj {u_id}: {e}")
        return 0

    added = 0
    # Try all parts at once first
    if len(parts) > 1:
        try:
            qtm.data.object.trajectory.move_parts(u_id, sid)
            print(f"  Traj {u_id}: moved all {len(parts)} parts to {id_name}.")
            return 1
        except Exception:
            pass   # Fall through to non-overlapping subset
        # Try the subset of parts entirely within the gap
        subset = [i for i, p in enumerate(parts)
                  if np.all(np.isin(
                      np.arange(p["range"]["start"], p["range"]["end"] + 1), gap_ind))]
        if subset:
            try:
                qtm.data.object.trajectory.move_parts(u_id, sid, subset)
                print(f"  Traj {u_id}: moved {len(subset)} non-overlapping parts.")
                added += 1
            except Exception:
                pass

    # Part-by-part with overlap repair
    parts = [p for p in qtm.data.object.trajectory.get_parts(u_id)
             if p["type"] == "measured"]
    parts = SelectPartsWithMinimumOverlap(parts, gap_ind, 1000)
    t0 = datetime.now()
    ii = 0
    while ii < len(parts):
        if (datetime.now() - t0).total_seconds() > TIMEOUT_PART_ADD:
            print("  Part-adding loop timed out.")
            break
        if n_added + added >= n_max:
            break
        try:
            qtm.data.object.trajectory.move_parts(u_id, sid, [ii])
            added += 1
            parts = [p for p in qtm.data.object.trajectory.get_parts(u_id)
                     if p["type"] == "measured"]
            parts = SelectPartsWithMinimumOverlap(parts, gap_ind, 1000)
        except Exception:
            # Overlap repair
            sub_gaps = [g for g in qtm.data.series._3d.get_gap_ranges(sid)
                        if g["start"] in gap_ind and g["end"] in gap_ind and
                        (g["end"] - g["start"] + 1) >= MIN_GAP_FRAMES]
            for sg in sub_gaps:
                try:
                    SwapPartsFixOverlap(u_id, sid,
                                        np.arange(sg["start"], sg["end"] + 1))
                except Exception as e2:
                    print(f"  SwapPartsFixOverlap failed: {e2}")
            ii += 1
    return added


# ── constraint helpers ────────────────────────────────────────────────────────

def AutoAdd_Whip_GetProxDist(data_by_name, whip_names, idx, dist_max_seq):
    """Return proximal/distal reference data and upper-bound distances for a whip marker."""
    prox  = data_by_name[whip_names[idx + 1]]
    prox2 = data_by_name.get(whip_names[idx + 2]) if idx + 2 < len(whip_names) else None
    ub_prox  = dist_max_seq[idx]
    ub_prox2 = (dist_max_seq[idx] + dist_max_seq[idx + 1]) if prox2 is not None else []
    dist_ref = data_by_name.get(whip_names[idx - 1]) if idx > 0 else None
    ub_dist  = dist_max_seq[idx - 1] if idx > 0 else []
    print(f"  Prox bound ({whip_names[idx + 1]}): {ub_prox} mm" +
          (f"  Dist bound ({whip_names[idx - 1]}): {ub_dist} mm" if idx > 0 else ""))
    return prox, prox2, dist_ref, ub_prox, ub_prox2, ub_dist


def AutoAdd_Whip_VerifyConstraints(ub_prox, ub_prox2, ub_dist,
                                    gap_ind, data_gap,
                                    data_prox, data_prox2, data_dist):
    """Return True (skip) if the candidate violates whip distance constraints."""
    if np.nanmedian(np.linalg.norm(data_prox[gap_ind] - data_gap, axis=1)) > ub_prox:
        return True
    if data_prox2 is not None and np.array(ub_prox2).size:
        if np.nanmedian(np.linalg.norm(data_prox2[gap_ind] - data_gap, axis=1)) > ub_prox2:
            return True
    if data_dist is not None and np.array(ub_dist).size:
        if np.nanmedian(np.linalg.norm(data_dist[gap_ind] - data_gap, axis=1)) > ub_dist:
            return True
    return False


def AutoAdd_Body_ClosestConstraints(data_by_name, all_names, data_self,
                                     id_name, n_constraints):
    """
    Find the n_constraints nearest neighbors of id_name and compute their
    1st–99th percentile distance bounds (used to screen candidates).
    """
    dist_med = {n: 9999 for n in all_names}
    dist_raw = {}
    for name in all_names:
        if name in data_by_name:
            d = np.linalg.norm(data_by_name[name] - data_self, axis=1)
            if not np.isnan(d).all():
                dist_med[name] = int(np.nanmedian(d))
                dist_raw[name] = d
    sorted_names = dict(sorted(dist_med.items(), key=lambda x: x[1]))
    dist_iqr, dist_min, dist_max = {}, {}, {}
    print(f"  {n_constraints} nearest neighbors of {id_name}:")
    for i, name in enumerate(sorted_names):
        if i == 0 or i > n_constraints:
            continue
        dist_min[name], dist_max[name] = np.nanpercentile(dist_raw[name], [1, 99])
        dist_iqr[name] = np.diff(np.nanpercentile(dist_raw[name], [25, 75]))[0]
        print(f"    {name}: median={dist_med[name]} mm, "
              f"bounds=[{int(dist_min[name])}, {int(dist_max[name])}]")
    return sorted_names, dist_iqr, dist_min, dist_max


def AutoAdd_Body_VerifyConstraints(dist_med_sorted, gap_ind, data_by_name,
                                    data_gap, n_constraints,
                                    dist_iqr, dist_min, dist_max, id_name):
    """Return True (skip) if the candidate violates body-marker distance constraints."""
    for i, name in enumerate(dist_med_sorted):
        if i == 0 or i > n_constraints:
            continue
        d = np.linalg.norm(data_by_name[name][gap_ind] - data_gap, axis=1)
        if (np.nanpercentile(d, 5)  < 0.9 * dist_min[name] or
                np.nanpercentile(d, 95) > 1.1 * dist_max[name]):
            return True
        # Extra check: wrist/hand cannot be further from elbow than the other one
        if id_name in ("W_RWristOut", "W_RHandOut"):
            elb = data_by_name.get("W_RElbowOut")
            wr  = data_by_name.get("W_RWristOut")
            hd  = data_by_name.get("W_RHandOut")
            if elb is not None and wr is not None and hd is not None:
                d_ew = np.linalg.norm(elb[gap_ind] - wr[gap_ind], axis=1)
                d_eh = np.linalg.norm(elb[gap_ind] - hd[gap_ind], axis=1)
                if np.any(d_ew > d_eh):
                    return True
        # Head L/R cannot cross the sagittal midplane
        if id_name in ("W_HeadL", "W_HeadR"):
            HL = data_by_name.get("W_HeadL")
            HR = data_by_name.get("W_HeadR")
            HT = data_by_name.get("W_HeadTop")
            HF = data_by_name.get("W_HeadFront")
            if all(v is not None for v in [HL, HR, HT, HF]):
                mid = 0.5 * (HT[gap_ind, :] + HF[gap_ind, :])
                if (np.any(HL[gap_ind, 0] < mid[:, 0]) or
                        np.any(HR[gap_ind, 0] > mid[:, 0])):
                    return True
    return False


# =============================================================================
# OVERLAP REPAIR  (SwapPartsFixOverlap)
# =============================================================================

def SelectPartsWithMinimumOverlap(parts, gap_ind, max_overlap):
    """
    Return only the parts of a trajectory whose frame range does not
    extend more than max_overlap frames outside gap_ind.
    Attaches the original part index as parts[i]["indorig"].
    """
    out = []
    for i, p in enumerate(parts):
        if (p["range"]["start"] > gap_ind[0] - max_overlap and
                p["range"]["end"] < gap_ind[-1] + max_overlap):
            p["indorig"] = i
            out.append(p)
    return out


def SwapPartsFixOverlap(id1=[], id2=[], gap_range=[], debug=False):
    """
    Move trajectory parts from id1 (donor) to id2 (acceptor), splitting
    at overlap boundaries when necessary.

    If id1/id2 are not provided, the two currently selected trajectories
    are used (the unlabeled one becomes the donor).
    gap_range : ndarray of frame indices defining the target gap.
                Inferred from the current cursor position when omitted.
    """
    n_frames = qtm.gui.timeline.get_frame_count()
    i_cur    = qtm.gui.timeline.get_current_frame()

    # Resolve IDs from selection if not supplied
    if not id1 or not id2:
        sel = qtm.gui.selection.get_selections()
        if not sel or len(sel) < 2:
            print("FIXSWAP: Select exactly two trajectories."); return
        ids   = [s["id"] for s in sel]
        names = [qtm.data.object.trajectory.get_label(s["id"]) for s in sel]
        if names[0] is None:               # ensure donor is index 1
            ids.reverse(); names.reverse()
    else:
        ids   = [id1, id2]
        names = [qtm.data.object.trajectory.get_label(id1),
                 qtm.data.object.trajectory.get_label(id2)]

    if len(ids) < 2:
        print("FIXSWAP: Need two trajectories."); return
    print(f"FIXSWAP: {names[1]}({ids[1]}) → {names[0]}({ids[0]})")

    data0 = GetMarkerDataAll(ids[0])
    data1 = GetMarkerDataAll(ids[1])
    valid0 = ~np.isnan(data0).any(axis=1)
    valid1 = ~np.isnan(data1).any(axis=1)
    ind0   = np.where(valid0)[0]
    ind_nan0 = np.where(~valid0)[0]

    # Infer gap_range from cursor if not supplied
    if len(gap_range) == 0:
        if not valid0[i_cur]:
            l = max(ind0[ind0 < i_cur], default=0)
            r = min(ind0[ind0 > i_cur], default=n_frames - 1)
            gap_range = np.arange(l + 1, r)
        else:
            l = min(ind_nan0[ind_nan0 > i_cur], default=None)
            if l is None:
                print("FIXSWAP: Cursor not near a gap."); return
            r = min(ind0[ind0 > l], default=n_frames - 1)
            gap_range = np.arange(l, r)
    print(f"FIXSWAP: Target gap {gap_range[0]}–{gap_range[-1]}")

    parts1 = [p for p in qtm.data.object.trajectory.get_parts(ids[1])
              if p["type"] == "measured"]
    parts1 = SelectPartsWithMinimumOverlap(parts1, gap_range, 1000)
    if not parts1:
        print("FIXSWAP: All parts exceed max overlap. Aborting."); return

    max_overlap_bool = not (valid0 * valid1).any()
    if max_overlap_bool:
        # No actual overlap — move all parts directly
        _move_parts_robust(ids[1], ids[0], parts1, gap_range)
        return

    # Overlap exists — split at overlap boundaries and move
    t0 = datetime.now()
    ip = 0
    while ip < len(parts1):
        if (datetime.now() - t0).total_seconds() > TIMEOUT_PART_ADD * 10:
            print("FIXSWAP: Timed out."); break
        p = parts1[ip]
        try:
            qtm.data.object.trajectory.move_parts(ids[1], ids[0], [ip])
            print(f"FIXSWAP: Moved part {ip + 1}/{len(parts1)}.")
            parts1 = [x for x in qtm.data.object.trajectory.get_parts(ids[1])
                      if x["type"] == "measured"]
            parts1 = SelectPartsWithMinimumOverlap(parts1, gap_range, 1000)
        except Exception:
            # Split around existing data in the acceptor
            rng    = np.arange(p["range"]["start"], p["range"]["end"] + 1)
            overlaps = np.where(valid0[rng])[0] + p["range"]["start"]
            splits   = np.unique(np.concatenate([overlaps, overlaps - 1, overlaps + 1]))
            splits   = splits[(splits >= gap_range[0] - 1) &
                               (splits <= gap_range[-1] + 1)]
            if len(splits) > 10:
                print("FIXSWAP: Too many overlaps. Aborting this part."); ip += 1; continue
            for sp in splits:
                try:
                    qtm.data.object.trajectory.split_part(ids[1], int(sp))
                except Exception:
                    pass
            parts1 = [x for x in qtm.data.object.trajectory.get_parts(ids[1])
                      if x["type"] == "measured"]
            parts1 = SelectPartsWithMinimumOverlap(parts1, gap_range, 1)
            _move_parts_robust(ids[1], ids[0], parts1, gap_range)
            ip += 1


def _move_parts_robust(src, dst, parts, gap_range):
    """Attempt to move each part individually, reporting results."""
    offset = 0
    for orig_i, p in enumerate(parts):
        rng = np.arange(p["range"]["start"], p["range"]["end"] + 1).tolist()
        try:
            qtm.data.object.trajectory.move_parts(src, dst,
                                                   [p["indorig"] - offset])
            offset += 1
            print(f"  Moved part {orig_i + 1}/{len(parts)}: "
                  f"frames {rng[0] + 1}–{rng[-1] + 1}")
        except Exception as e:
            print(f"  Could not move part {orig_i + 1}: {e}")


# =============================================================================
# REPORT / FRAME-RANGE UTILITIES
# =============================================================================

def GroupFramesInRanges(frames):
    """
    Convert a sorted list of (1-based) frame numbers into a compact
    range-string list, e.g. [1,2,3,7,8] → ['1-3(3)', '7-8(2)'].
    """
    if not frames:
        return []
    result = []
    seg_start = frames[0]
    seg_end   = frames[0]
    for fr in frames[1:]:
        if fr - seg_end > 1:
            result.append(_fmt_range(seg_start, seg_end))
            seg_start = fr
        seg_end = fr
    result.append(_fmt_range(seg_start, seg_end))
    return result


def _fmt_range(lo, hi):
    return f"{lo}-{hi}({hi - lo + 1})" if hi > lo else str(lo)


def ListReviewMarkersInRange(review_markers, frame_lo, frame_hi=-1):
    """
    Collect unique marker names logged in review_markers for frames
    [frame_lo, frame_hi] (or just frame_lo when frame_hi == -1).
    """
    marks = set()
    rng = range(frame_lo, frame_hi + 1) if frame_hi > -1 else [frame_lo]
    for fr in rng:
        if fr in review_markers:
            marks.update(review_markers[fr].split(";"))
    return ", ".join(sorted(marks))


# =============================================================================
# HELPER UTILITIES
# =============================================================================

def FindClosestDataRanges(data, i_cur):
    """
    Given an (N, 3) position array and a current frame index, return a
    (3, 2) array describing the three data-filled segments nearest to i_cur:
      row 0: segment to the left of the gap to the left of i_cur
      row 1: current segment containing i_cur
      row 2: segment to the right of the gap to the right of i_cur
    """
    n = len(data)
    ranges = np.array([[-1, -1], [-1, n], [n, n]], dtype=int)
    # Scan left
    state = 0
    for i in range(i_cur, -1, -1):
        is_nan = np.isnan(data[i]).any()
        if state == 0 and is_nan:
            ranges[1, 0] = i + 1; state = 1
        elif state == 1 and not is_nan:
            ranges[0, 1] = i; state = 2
        elif state == 2 and is_nan:
            ranges[0, 0] = i + 1; break
    # Scan right
    state = 0
    for i in range(i_cur, n):
        is_nan = np.isnan(data[i]).any()
        if state == 0 and is_nan:
            ranges[1, 1] = i - 1; state = 1
        elif state == 1 and not is_nan:
            ranges[2, 0] = i; state = 2
        elif state == 2 and is_nan:
            ranges[2, 1] = i - 1; break
    return ranges


def find_name_by_id(name_id_dict, target_id):
    for name, sid in name_id_dict.items():
        if sid == target_id:
            return name
    return None


def find_id_by_name(name_id_dict, target_name):
    return name_id_dict.get(target_name)


# =============================================================================
# QTM MENU REGISTRATION
# =============================================================================

def add_menu():
    """Register menu items and keyboard shortcuts in QTM."""
    menu_id = qtm.gui.insert_menu_submenu(None, "Whip Scripts")

    add_command("_GoToFrame",               gtf)
    add_command("_Whip_Names_to_W_Names",   Whip_Names_to_W_Names)
    add_command("_Whip_W_Names_to_Names",   Whip_W_Names_to_Names)
    add_command("_VerifyDist",              Whip_CheckWhipM2MDistances)
    add_command("_SwapPartsFixOverlap",     SwapPartsFixOverlap)
    add_command("_AddJoinTraj",             AddJoinTrajButton)
    add_command("_MarkerRemoveVirtual",     MarkerRemoveVirtualParts)
    add_command("_Verify_GoToNext",         Verify_GoToNext)

    add_menu_item(menu_id, "Go to Frame (terminal)",
                  "_GoToFrame")
    add_menu_item(menu_id, "Whip – add W_ prefix to body markers",
                  "_Whip_Names_to_W_Names")
    add_menu_item(menu_id, "Whip – remove W_ prefix from body markers",
                  "_Whip_W_Names_to_Names")
    qtm.gui.insert_menu_separator(menu_id)
    add_menu_item(menu_id, "Verify inter-marker distances  (>> VerifyDist)",
                  "_VerifyDist")
    add_menu_item(menu_id, "Go to next verify-problematic interval",
                  "_Verify_GoToNext")
    add_menu_item(menu_id, "Swap parts / fix overlap",
                  "_SwapPartsFixOverlap")
    add_menu_item(menu_id, "Fill gaps using support trajectories  (>> AddJoinTraj)",
                  "_AddJoinTraj")
    add_menu_item(menu_id, "Remove virtual parts from selected marker",
                  "_MarkerRemoveVirtual")

    qtm.gui.set_accelerator({"ctrl": True,  "alt": False, "shift": True,  "key": "w"},
                             "_Whip_Names_to_W_Names")
    qtm.gui.set_accelerator({"ctrl": True,  "alt": False, "shift": True,  "key": "d"},
                             "_VerifyDist")
    qtm.gui.set_accelerator({"ctrl": False, "alt": False, "shift": False, "key": "e"},
                             "_SwapPartsFixOverlap")
    qtm.gui.set_accelerator({"ctrl": True,  "alt": False, "shift": False, "key": "g"},
                             "_AddJoinTraj")
    qtm.gui.set_accelerator({"ctrl": False, "alt": False, "shift": False, "key": "n"},
                             "_Verify_GoToNext")


if __name__ == "__main__":
    add_menu()
