# Whip_Scripts — QTM Automation for Bullwhip Motion-Capture Labeling

Semi-automated trajectory labeling and quality-control scripts for
Qualisys Track Manager (QTM), developed for the bullwhip target-striking paradigm
described in Krotov et al. (2022, *Royal Society Open Science*) and the
in-preparation multi-manuscript series on human motor control of a bullwhip.

The inspiration developed after half of my M.S. was manually fixing marker labels, 
which jumped between the markers as subjects were trying to hit a target with a whip.

Empirical logic developed through sweat, tears, and mixed-language cursing while
identifying markers in 45 40s-long (667 Hz) recordings in each of 17 subjects. 
Specific constant values are of little interest for potential re-use, but I hope 
the logic of two main workhorse scripts would save someone's time, like it once saved mine:
1. Verifying inter-marker distances (to detect potential issues) before exporting a mocap
   file for postprocessing.
2. Using a greedy-like approach to automatically add marker labels to a trajectory,
   under physical constraints and 

**Environment:** QTM 2023.3 build 12577, 32-bit Python 3.10.  
Third-party packages (numpy, scipy, pandas) were installed from pre-built
wheels from https://www.lfd.uci.edu/~gohlke/pythonlibs/ because the
32-bit interpreter supplied with 2023.3 (and up to 2024.1) versions of QTM
could not compile them natively.

---

## Repository contents

```
Whip_Scripts.py                        main script (loaded by QTM)
example_VerifyDist_Thresholds.txt      example threshold file — copy and customise
helpers/
    menu_tools.py                      QTM menu-registration wrappers
    traj.py                            QTM trajectory helpers
```

---

## Setup

1. Copy `example_VerifyDist_Thresholds.txt` to your QTM project folder.
2. Open `Whip_Scripts.py` and edit the six path/constant variables at the
   top of the **CONFIGURATION** section to match your project layout.
3. Load the script in QTM (*Scripts → Load*). The **Whip Scripts** menu and
   keyboard shortcuts are registered automatically.

---

## Functions and logic

### Simple utilities

| Function | Shortcut | Description |
|---|---|---|
| `gtf(N)` / `f(N)` | – | Jump to frame N (1-based QTM index). |
| `Whip_Names_to_W_Names` | Ctrl+Shift+W | Add `W_` prefix to all non-whip, non-target trajectories. |
| `Whip_W_Names_to_Names` | – | Remove `W_` prefix. |
| `MarkerRemoveVirtualParts` | – | Delete auto-filled virtual parts from the selected trajectory. |

---

### Distance verification — `VerifyDist` (Ctrl+Shift+D)

**Purpose:** scan the entire recording for physically and/or biomechanically implausible
inter-marker distances, flag suspect frames, unify them into ranges, and open a timestamped
plain-text report.

**Algorithm:**

1. Build a symmetric *N × N* distance-threshold matrix from:
   - Sequential thresholds (`DIST_MAX_*_SEQ`): the *k*-th value of a group's
     sequence becomes the upper-bound distance (mm) for the adjacent pair
     (marker *k*, marker *k+1*) within that anatomical group, stored at the
     symmetric off-diagonal entries [k, k+1] and [k+1, k].
   - Non-adjacent overrides (`DIST_MAX_NONADJACENT`) for specific pairs
     (e.g., Target-3 ↔ Target-1).
   - User-edited values from `THRESHOLD_FILE_PATH` (overwrites defaults if
     the marker list matches). Slight changes may be introduced for different
     subjects / different sessions based on manual inspection of the data.

2. For every marker pair (*m1*, *m2*) with a positive threshold:
   - Compute the Euclidean distance time-series $d(t) = \|p_{m1}(t) - p_{m2}(t)\|$.
   - Report: min, median(5th–95th-percentile IQR), max, and N valid frames.
   - Flag all frames where $d(t)$ exceeds the threshold.
    This prevented the swap of any markers of the thong, most crucially,
    but also of some body markers.

3. Two additional geometry checks are run unconditionally:

   **Handle colinearity.** The four handle markers (HDR, HDL, w10, HD0)
   should be approximately coplanar in a line. The scalar residual is the
   point-to-line distance of HD0 from the line defined by the cross-product
   of (HDR − HDL) and (HD0 − w10):
   $$r = \frac{|\mathbf{n} \cdot (\mathbf{A} - \mathbf{C})|}{|\mathbf{n}|},
   \quad \mathbf{n} = (\text{HDR}-\text{HDL}) \times (\text{w10}-\text{HD0})$$
   Frames where $r > 55$ mm are flagged. This prevented the swap of either of
   those markers with the hand or the whip thong markers.

   **Head L/R swap.** W_HeadL should always have a larger *x*-coordinate
   (rightward in the lab frame) than the midpoint of W_HeadTop and W_HeadFront,
   and W_HeadR the smaller. Any frame violating this is flagged.

5. **Relational checks.** Directional inequalities of the form *d(m1,m2) > d(m3,m4)*
   are evaluated frame-by-frame. The default relations are:
   - `w10–HDL > w10–HDR` (the two handle markers must not swap)
   - `W_RElbowOut–W_RHandOut > W_RElbowOut–W_RWristOut` (hand_out and wrist_out markers must not swap)

6. All flagged frames are collated into a final review table showing
   consecutive ranges and the involved markers.

**Threshold file format** (`THRESHOLD_FILE_PATH`, plain text):
```
QTM VerifyDist thresholds. Last edited: ...
Marker Names and Indices
0 - w1
1 - w2
...
***Thresholds***
0,95,0,...
95,0,135,...
```
Entry [i, j] (and its mirror [j, i]) stores the upper-bound distance between
markers *i* and *j*; 0 means no threshold is checked for that pair. The diagonal
is always 0. Use `VerifyDist_Set("w1-w2", 110)` to update a single value without
opening the file.

**Navigate to the next flagged interval:** `Verify_GoToNext` (N key) reads the
saved report, skips intervals shorter than 5 frames, and jumps to the frame just
before the next flagged range. For two-marker violations it fetches the actual
peak distance and compares it to the stored threshold.

---

### Automatic gap filling — `AddJoinTraj` (Ctrl+G)

**Purpose:** fill gaps in a labeled trajectory by recruiting unidentified
trajectory segments that satisfy learned distance constraints.

**Algorithm:**

1. **Cache** all unidentified trajectories as integer arrays before the gap
   loop (avoids repeated API calls — critical for performance on long recordings).

2. **Constraint learning.** For whip markers, the proximal-neighbor and
   distal-neighbor distance bounds come directly from `DIST_MAX_WHIP_SEQ`.
   For body markers, the *N* nearest labeled neighbors are found by sorting
   median distances, and their 1st–99th percentile distance bounds are
   computed from the current recording.

3. **Gap loop.** For each gap of ≥ `MIN_GAP_FRAMES` frames (starting from
   the current cursor position):
   - For each unidentified trajectory with data in the gap window:
     - **Whip constraint:** median distance to the proximal neighbor must be
       ≤ `ub_prox`; to the second-proximal neighbor ≤ `ub_prox + ub_prox2`; to
       the distal neighbor (when it exists) ≤ `ub_dist`.
     - **Body constraint:** the 5th–95th percentile of the candidate's distance
       to each of the *N* reference markers must lie within [0.9·P1, 1.1·P99]
       of the learned distribution, where P1 and P99 are the 1st and 99th
       percentiles observed in the current recording. Additional anatomical
       checks are applied for wrist/hand and head markers.
   - Qualifying candidates are sorted by earliest valid frame and processed
     in that order.

4. **Part addition.** For each qualifying candidate:
   - Try to move all parts at once.
   - Fall back to moving only the non-overlapping subset.
   - Fall back to part-by-part; call `SwapPartsFixOverlap` when an overlap
     prevents a direct move.

5. Hard stops: `MAX_PARTS_ADDED` parts per call; `TIMEOUT_GAP_SEARCH` seconds
   for the outer loop; `TIMEOUT_PART_ADD` seconds for the inner part-adding loop.

**`SwapPartsFixOverlap` (E key):** resolves overlaps between two selected trajectories
by splitting the donor at every frame where it overlaps with existing data on
the acceptor, then moving the non-overlapping fragments.

---

### First-frame auto-labeling — `LabelAllInFrame_Train` / `LabelAllInFrame_Label`

These two functions support the common workflow where the analyst navigates to
a standardised "whip forward" posture and calls *Train* once per recording to
update a positional model, then calls *Label* to auto-assign unidentified
trajectories near that frame.

**Model (stored as `LABEL_MODEL_PATH`, `.npz`):**
- For each marker: mean position μ (3-vector, mm) and variance σ² (3-vector),
  updated online using Welford's algorithm.
- For each marker pair: mean inter-marker distance (scalar, mm), also updated
  online.

**Welford update** for a running mean and variance (numerically stable):
$$n' = n + 1, \quad \mu' = \frac{n\,\mu + x}{n'}$$
$$\sigma'^2 = \frac{(n-1)\,\sigma^2 + (x - \mu)(x - \mu')}{n' - 1}$$

**Scoring (Label step):** each unidentified trajectory's mean position over the
window is scored against every known marker using an unnormalised diagonal
multivariate Gaussian:
$$\text{score}(x \mid \mu, \sigma) = -\frac{1}{2} \sum_{k=1}^{3}
\left(\frac{x_k - \mu_k}{10\,\sigma_k + \epsilon}\right)^2$$
The trajectory is assigned to the marker with the highest score.

---

## Notes on the QTM Python environment

- The 32-bit Python 3.10 bundled with QTM 2023.3 cannot natively compile
  C-extension packages. Wheels from https://www.lfd.uci.edu/~gohlke/pythonlibs/
  were used for numpy, scipy, and pandas.
- Some QTM API methods documented in the recent SDK are not yet implemented in
  this build (e.g., `timeline.set_measured_range`). Several workarounds
  were required; see inline comments.
- `qtm.data.series._3d.get_samples` returns a list of dicts (or `None`
  for empty frames); it does not support NumPy-style slicing. All array
  construction is done by looping over the returned list.

---

## Citation

If you use these scripts, please cite the associated manuscript:

> 
> *Hitting a target with a whip: Cognitive, athletic, and task-centered
determinants.* (in preparation)

and the earlier dataset paper:

> Krotov A, Russo M, Nah M, Hogan N, Sternad D (2022).
> Motor control beyond reach — how humans hit a target with a whip.
> *Royal Society Open Science*, 9, 220525.
> https://doi.org/10.1098/rsos.220525
