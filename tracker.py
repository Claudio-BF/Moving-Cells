#!/usr/bin/env python3
"""Stabilize a CZI timelapse, segment neutrophils in 3D by Cellpose-SAM stitching,
track with Ultrack, inspect in napari.

The 2D->3D step is plain Cellpose-style stitching, using the Cellpose-SAM
("cpsam") model:

    1. Keep the full Z-stack (no projection).
    2. Estimate XY camera shake and apply the SAME shift to every Z slice.
    3. Segment each Z slice independently in 2D with Cellpose-SAM.
    4. Stitch the per-slice 2D masks into 3D by matching them along Z on
       intersection-over-union (IoU) -- the same idea as Cellpose's
       ``stitch_threshold``, with one addition: a label may be reconnected
       across a short Z gap (see STITCH_MAX_Z_GAP).
    5. Convert the 3D labels to Ultrack foreground/contours and track in 3D.

Because labels are propagated by 2D-mask overlap (not by 3D connectivity),
horizontal separation and Z-gap bridging are independent knobs:

  * Horizontal separation is decided entirely by the 2D Cellpose masks. Two cells
    that touch in XY but are separate 2D objects stay separate all the way up the
    stack. Tune this with the CELLPOSE_* knobs (cellprob, flow).
  * Z-gap bridging is decided by STITCH_MAX_Z_GAP, which lets a label survive a
    few empty slices and re-match by IoU. Matching is per-object, so bridging a
    gap never merges neighbours in XY.

So: if cells merge side-to-side, fix it in 2D (Cellpose); if a cell breaks into
two along Z, raise STITCH_MAX_Z_GAP.

Edit the USER SETTINGS block, then run:

    python tracker.py

List one or more movies in CZI_PATHS: each is processed in turn (its outputs
written to its own folder), then all are opened together in a SINGLE napari
window with a "Movie" dropdown to switch between them -- no extra windows.
Command-line parameters are not accepted.

Set TIME_SUBSET to a single frame index to segment just one frame (fast) and skip
tracking, or to a (start, stop) pair to run the full pipeline on only that
inclusive range of frames.
"""

from __future__ import annotations

import inspect
import json
import logging
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any

import dask.array as da
import napari
import numpy as np
import pandas as pd
from bioio import BioImage
from scipy import ndimage as ndi
from skimage.measure import regionprops_table
from skimage.registration import phase_cross_correlation
from tqdm.auto import tqdm
from ultrack import MainConfig, Tracker
from ultrack.utils import labels_to_contours
from ultrack.utils.array import create_zarr


# Silence two scikit-image >= 0.26 FutureWarnings raised from inside Ultrack
# (remove_small_objects(min_size=...) and RegionProperties.intensity_image, the
# pre-0.26 spellings). They come from the installed Ultrack package, not this
# script; drop this block once Ultrack adopts the >= 0.26 API.
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=r".*min_size.*deprecat",
)
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=r".*intensity_image.*deprecat",
)


# =============================================================================
# USER SETTINGS — edit these values before running the script
# =============================================================================

# Input and output.
# List one or more movies here. Each is processed independently and then all are
# shown together in one napari window (a "Movie" dropdown switches between them).
CZI_PATHS: list[Path] = [
    Path("/mnt/d/JerisonServer/Eric/Lightfield demo/2026-03-27-mpx-r848-dss-7dpf/processed/Z-subsets/Bath-5.czi"),
    Path("/mnt/d/JerisonServer/Eric/Lightfield demo/2026-03-27-mpx-r848-dss-7dpf/processed/Z-subsets/Bath-2.czi"),
    Path("/mnt/d/JerisonServer/Eric/Lightfield demo/2026-03-27-mpx-r848-dss-7dpf/processed/Z-subsets/Bath-8.czi"),
    Path("/mnt/d/JerisonServer/Eric/Lightfield demo/2026-03-27-mpx-r848-dss-7dpf/processed/Z-subsets/11.czi"),
    Path("/mnt/d/JerisonServer/Eric/Lightfield demo/2026-03-27-mpx-r848-dss-7dpf/processed/Z-subsets/20.czi"),
    Path("/mnt/d/JerisonServer/Eric/Lightfield demo/2026-03-27-mpx-r848-dss-7dpf/processed/Z-subsets/22.czi"),
    # add more movies here, e.g.:
    # Path("/mnt/d/JerisonServer/Eric/Lightfield demo/2026-03-27-mpx-r848-dss-7dpf/processed/Z-subsets/Bath-6.czi"),
]
# Each movie writes to its OWN folder so several movies never clobber each other:
#   OUTPUT_DIR set   -> <OUTPUT_DIR>/<CZI filename>/
#   OUTPUT_DIR None  -> <CZI filename>_ultrack beside each CZI
OUTPUT_DIR: Path | None = Path("data")

# CZI selection and processing mode
SCENE = 0
TRACK_CHANNEL = 0
REGISTRATION_CHANNEL: int | None = None  # None -> use TRACK_CHANNEL
NO_STABILIZATION = False

# Time-point selection / quick-test mode.
#   None             -> process the whole movie and track.
#   an int N         -> process ONLY frame N (stabilize, segment, stitch, show in
#                       napari) and skip tracking. Use this to tune Cellpose +
#                       stitching quickly on a single frame.
#   a (start, stop)  -> process only frames start..stop INCLUSIVE and run the full
#                       pipeline (tracking included) on that slice. For example
#                       (10, 60) processes the 51 frames 10 through 60. A length-1
#                       slice behaves exactly like the single-frame case above.
# Reported frame numbers (napari titles, run_parameters.json) use the original
# absolute indices; the arrays are re-indexed from 0 within the slice.
TIME_SUBSET: int | tuple[int, int] | None = [10,50]

# Registration settings (simple global XY translation).
REGISTRATION_DOWNSAMPLE = 2
REGISTRATION_SIGMA = 4.0
REGISTRATION_UPSAMPLE = 10
MAX_REGISTRATION_SHIFT = 50.0

# -- 2D per-slice segmentation (Cellpose-SAM / "cpsam") --
# Cellpose-SAM runs once per Z slice. It is a channel-agnostic, largely
# size-invariant generalist trained on ROI diameters from ~7.5 to 120 px
# (mean 30), so it needs no channel selection and no diameter estimate to
# segment these small neutrophils out of the box.
#
# >>> To reduce HORIZONTAL (XY) merging of touching cells, in rough order of
#     effectiveness:
#       - raise CELLPOSE_CELLPROB_THRESHOLD (e.g. 0.5–3.0): shrinks masks, so
#         touching cells separate;
#       - lower CELLPOSE_FLOW_THRESHOLD (e.g. 0.3): drops poorly-formed masks;
#       - raise NORM_HIGH_PERCENTILE toward 99.999.
CELLPOSE_MODEL = "cpsam"           # built-in Cellpose-SAM, or a path to a fine-tuned model
CELLPOSE_GPU = True
CELLPOSE_FLOW_THRESHOLD = 1.3
CELLPOSE_CELLPROB_THRESHOLD = 1.3  # raise (e.g. 1.0–3.0) to drop weaker detections
# Number of image patches run on the GPU at once. cpsam uses the memory-heavy
# SAM-ViT-L backbone, so its default is small: raise it on a large-VRAM GPU for
# speed, or lower it if you hit a CUDA out-of-memory error.
CELLPOSE_BATCH_SIZE = 8
# Optional size hint. None keeps native resolution and is recommended here: these
# ~7 px cells sit right at the floor of cpsam's trained range (7.5–120 px), so it
# should handle them without rescaling. Setting a value rescales toward cpsam's
# 30 px mean: below 30 upsamples (recovers missed tiny cells, but is much slower);
# above 30 downsamples big cells (faster).
CELLPOSE_DIAMETER: float | None = 5

# -- 2D -> 3D stitching --
# Two adjacent-slice 2D masks are joined into one 3D object when their IoU is at
# least STITCH_IOU_THRESHOLD (this is Cellpose's `stitch_threshold`). Lower it if
# a cell that clearly continues up the stack is being split into separate labels;
# raise it if two distinct cells stacked along Z are being fused.
STITCH_IOU_THRESHOLD = 0.25
# Maximum number of CONSECUTIVE empty Z slices a label may jump across and still
# be reconnected to its earlier appearance. 0 = Cellpose's behaviour (adjacent
# slices only). 1 bridges a single-slice dropout, which is what you asked for.
# This bridges gaps WITHOUT any XY dilation, so it cannot cause horizontal merges.
STITCH_MAX_Z_GAP = 2

# -- Brightness handling (separates bright neutrophils from the dim fish) --
#   - Raise NORM_LOW_PERCENTILE to suppress more of the fish (more aggressive).
#   - Lower it if real neutrophils are being missed.
#   - Raise NORM_HIGH_PERCENTILE toward 99.999 if touching neutrophils merge.
NORM_LOW_PERCENTILE = 99.0
NORM_HIGH_PERCENTILE = 99.9999
NORM_SAMPLE_FRAMES = 8  # frames sampled across time to estimate the global window
# Discard any 2D object whose mean (globally-normalized) brightness is below this
# floor, removing dim fish regions that slipped through. 0 disables the gate.
MIN_CELL_NORM_INTENSITY = 0.1

# -- Per-slice size + shape filter (runs on each Z slice's 2D Cellpose masks) --
# After the dim-object gate above, every 2D object on a slice is checked and
# dropped before stitching if it is the wrong SIZE or not roughly circular, so a
# malformed detection never reaches the 3D objects or the tracker.
#   * SIZE: an object is removed when its 2D pixel area is < MIN_AREA or
#     > MAX_AREA -- the SAME bounds Ultrack uses for its 3D segments (defined in
#     the "ultrack.segmentation" block below). They are REUSED here as 2D pixel
#     areas: for these ~6-7 px neutrophils a mid-cell cross-section is ~30-40 px,
#     so the [MIN_AREA, MAX_AREA] band keeps whole cells while dropping single-
#     pixel slivers and over-large merged/hazy blobs. (This also trims the tiny
#     top/bottom cross-sections of a cell, but STITCH_MAX_Z_GAP can bridge that.)
#   * SHAPE: circularity is judged by ECCENTRICITY of the equivalent ellipse:
#     0 = a perfect circle, ->1 = a straight line. It is derived from image
#     moments, so it stays reliable on the small pixelated masks here -- unlike
#     the perimeter-based form factor 4*pi*area/perimeter^2, whose perimeter is
#     badly estimated at this object size. An object with eccentricity above
#     SLICE_MAX_ECCENTRICITY is dropped; 0.85 rejects anything longer than ~1.9:1,
#     which mostly catches two cells merged side by side. Set it to 1.0 to skip
#     the shape check entirely.
#   * OPTIONAL extra shape gate: SOLIDITY (area / convex-hull area; 1.0 = convex)
#     catches ragged or pinched masks that are not elongated. It is OFF by default
#     (0.0) because rasterization pushes small but legitimate cross-sections down
#     toward ~0.85; raise it (e.g. 0.9) to also drop concave/lumpy blobs.
# Set SLICE_FILTER_ENABLED = False to disable the whole per-slice filter.
SLICE_FILTER_ENABLED = True
SLICE_MAX_ECCENTRICITY = 0.88  # 0 = circle, ->1 = line; 1.0 disables the check
SLICE_MIN_SOLIDITY = 0.0       # area/convex area; 0.0 disables; try 0.9 to enable

# -- Ultrack contour generation from the 3D labels --
CONTOUR_SIGMA = 1.0  # Gaussian smoothing of label boundaries in labels_to_contours

# Ultrack settings.

# -- ultrack.data --
DATA_WORKERS = 4

# -- ultrack.segmentation --
MIN_AREA = 12        # 3D voxels
MAX_AREA = 60       # 3D voxels
MIN_AREA_FACTOR = 2.0
MIN_FRONTIER = 0.08
SEG_THRESHOLD = 0.3
MAX_NOISE = 0.0
ANISOTROPY_PENALIZATION = 0.0  # >0 penalizes hypotheses that grow across Z
SEG_RANDOM_SEED: int | str = "frame"
SEG_WORKERS = 8

# -- ultrack.linking --
# >>> THESE TWO SETTINGS GOVERN "HUGE JUMPS" <<<
#   MAX_DISTANCE is a HARD CAP: two detections more than this many (x-scaled)
#     pixels apart in consecutive frames are NEVER linked. This is the knob that
#     stops Ultrack from stitching a far-away cell onto a track.
#   DISTANCE_WEIGHT is the SOFT PENALTY: each candidate link's cost grows by
#     DISTANCE_WEIGHT * distance, so among allowed links the solver prefers the
#     shortest. Raise it to penalize long-but-legal jumps harder.
# Cells here move ~3-5 px/frame, so the cap is set to ~3x that (15). If huge jumps
# persist, lower MAX_DISTANCE toward 10-12 and/or raise DISTANCE_WEIGHT further.
MAX_DISTANCE = 10.0     # hard cap on frame-to-frame movement (x-pixel units)
MAX_NEIGHBORS = 8
DISTANCE_WEIGHT = 0.01  # penalize longer links (soft cost ~ weight * distance)
LINK_Z_SCORE_THRESHOLD = 7.0
LINK_WORKERS = 8

# -- ultrack.tracking (ILP solver) --
SOLVER = "GUROBI"  # One of: "auto", "GUROBI", or "CBC"
APPEAR_WEIGHT = -0.0015
DISAPPEAR_WEIGHT = -0.0015
DIVISION_WEIGHT = -0.05
TRACKING_THREADS = 4
SOLUTION_GAP = 0.001
TIME_LIMIT = 36000  # seconds
TRACKING_METHOD = 0
LINK_FUNCTION = "power"  # One of: "power", "identity"
POWER = 4.0
BIAS = 0.0

# Temporal windowing. WINDOW_SIZE = 0 solves the whole movie at once.
WINDOW_SIZE = 0
OVERLAP_SIZE = 5

# -- Post-tracking track filtering (applied after solving, before export) --
# Remove finished tracks AND erase their cells from the segmentation when a
# track is either too short or barely moves:
#   * present in fewer than MIN_TRACK_LENGTH frames, or
#   * mean per-frame centroid step below MIN_MEAN_SPEED voxels/frame (raw z,y,x
#     voxel distance). Cells here crawl ~3-5 voxels/frame, so 2 keeps movers and
#     drops near-stationary debris.
MIN_TRACK_LENGTH = 5    # frames
MIN_MEAN_SPEED = 0.0    # voxels per frame

# napari trajectory tail length (display only).
TAIL_LENGTH = 15

# =============================================================================
# END USER SETTINGS
# =============================================================================


# ---------------------------------------------------------------------------
# Small array helpers
# ---------------------------------------------------------------------------
def _compute(array: Any) -> np.ndarray:
    """Convert a NumPy/Dask/Zarr slice to a NumPy array."""
    if hasattr(array, "compute"):
        array = array.compute()
    return np.asarray(array)


def _positive_labels(array: np.ndarray) -> np.ndarray:
    """Return the sorted non-zero (foreground) label ids in ``array``."""
    ids = np.unique(array)
    return ids[ids > 0]


def _positive_float(value: Any, fallback: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    return value if np.isfinite(value) and value > 0 else fallback


def _as_dask(array: Any) -> da.Array:
    """Wrap a zarr/NumPy array as a dask array (avoids da.from_zarr quirks)."""
    if isinstance(array, da.Array):
        return array
    chunks = getattr(array, "chunks", None) or "auto"
    return da.from_array(array, chunks=chunks)


def _label_layer(array: Any) -> Any:
    """Return a napari-friendly integer array for an add_labels layer."""
    layer = _as_dask(array)
    return layer.astype(np.uint8) if layer.dtype == bool else layer


def _new_zarr(shape: Any, dtype: Any, path: Path) -> Any:
    """Create a disk-backed (over)writable Zarr of the given shape and dtype."""
    return create_zarr(
        shape=tuple(int(v) for v in shape),
        dtype=dtype, store_or_path=str(path), overwrite=True,
    )


# ---------------------------------------------------------------------------
# Environment / logging helpers
# ---------------------------------------------------------------------------
def _silence_cellpose_seed_warnings() -> None:
    """Quiet Cellpose's per-slice "no seeds found ... no masks found" logging.

    Cellpose logs it at WARNING from ``cellpose.dynamics`` once per empty Z slice
    -- tens of thousands of identical lines on a big stack. Raising only that
    logger to ERROR silences them without touching any other Cellpose message.
    Call this AFTER the model is constructed so Cellpose's own logging setup does
    not reset it.
    """
    logging.getLogger("cellpose.dynamics").setLevel(logging.ERROR)


def _check_gpu_imgproc_stack() -> None:
    """Fail fast when Ultrack will use the GPU for image processing but cuCIM is absent.

    Ultrack's imgproc helpers (notably ``labels_to_contours``) move each frame to
    the GPU whenever CuPy is importable and then call ``cucim.skimage``. With CuPy
    but no cuCIM they hand a CuPy array to CPU scikit-image and crash -- and only
    AFTER segmentation finishes. Checking here turns a multi-hour-then-crash into
    an instant, actionable error. No-ops when CuPy is missing (Ultrack stays on
    the CPU).
    """
    import importlib.util

    if importlib.util.find_spec("cupy") is None:
        return  # No CuPy -> Ultrack runs imgproc on CPU -> cuCIM not required.

    try:
        import cucim.skimage  # noqa: F401  # the exact module Ultrack uses on GPU
    except Exception as exc:  # missing, or a broken cuCIM/CuPy/CUDA install
        raise ImportError(
            "CuPy is installed but cuCIM is not importable, so Ultrack's "
            "labels_to_contours step will move data to the GPU and then crash "
            "(TypeError: Implicit conversion to a NumPy array is not allowed). "
            "Install the cuCIM build matching your CuPy/CUDA major version into "
            "this environment, e.g. `pip install cucim-cu12` (CUDA 12.x) or "
            "`pip install cucim-cu11` (CUDA 11.x). To run on the CPU instead, "
            "uninstall CuPy or delete this check."
        ) from exc


# ---------------------------------------------------------------------------
# XY stabilization (camera shake only; biology is untouched)
# ---------------------------------------------------------------------------
def _project_to_2d(tzyx: da.Array) -> da.Array:
    """Collapse Z to a 2D (TYX) movie for registration estimation only."""
    return tzyx[:, 0] if int(tzyx.shape[1]) == 1 else tzyx.max(axis=1)


def _prepare_registration_frame(
    frame_yx: np.ndarray, *, downsample: int, sigma: float
) -> np.ndarray:
    """Normalize, blur, crop, and downsample one 2D registration frame."""
    frame = np.nan_to_num(np.asarray(frame_yx, dtype=np.float32), copy=False)

    low, high = np.percentile(frame, (1.0, 99.8))
    if high <= low:
        high = low + 1.0
    frame = np.clip((frame - low) / (high - low), 0.0, 1.0)

    if sigma > 0:
        frame = ndi.gaussian_filter(frame, sigma=sigma)

    # Ignore a narrow border, which is less stable and causes wrap ambiguity.
    by, bx = (int(round(s * 0.04)) for s in frame.shape)
    if by > 0 and bx > 0:
        cropped = frame[by:-by, bx:-bx]
        if min(cropped.shape) >= 32:
            frame = cropped

    downsample = max(1, int(downsample))
    frame = frame[::downsample, ::downsample]

    # A soft window reduces edge artifacts in phase correlation.
    if min(frame.shape) >= 8:
        window = np.outer(np.hanning(frame.shape[0]), np.hanning(frame.shape[1]))
        frame = frame * window.astype(np.float32, copy=False)

    return frame


def estimate_xy_shifts(
    registration_tyx: da.Array,
    *,
    downsample: int,
    sigma: float,
    upsample_factor: int,
    max_shift_px: float,
) -> np.ndarray:
    """Estimate the absolute XY shift required to stabilize every frame.

    Each raw frame is aligned to the previously stabilized frame by phase
    correlation. Only translation is modeled.
    """
    n_time = int(registration_tyx.shape[0])
    shifts = np.zeros((n_time, 2), dtype=np.float64)
    ds = max(1, int(downsample))

    previous = _prepare_registration_frame(
        _compute(registration_tyx[0]), downsample=ds, sigma=sigma
    )

    for t in tqdm(range(1, n_time), desc="Estimating XY stabilization"):
        current = _prepare_registration_frame(
            _compute(registration_tyx[t]), downsample=ds, sigma=sigma
        )
        try:
            shift_small, error, _ = phase_cross_correlation(
                previous, current, upsample_factor=max(1, int(upsample_factor)),
                normalization=None,
            )
            shift_full = np.asarray(shift_small[-2:], dtype=float) * ds
            if not np.all(np.isfinite(shift_full)):
                raise ValueError("non-finite phase-correlation shift")
            if np.linalg.norm(shift_full) > max_shift_px:
                raise ValueError(f"shift {shift_full} exceeds MAX_REGISTRATION_SHIFT")
            if not np.isfinite(error):
                warnings.warn(f"Frame {t}: phase-correlation error is non-finite")
        except Exception as exc:  # keep the pipeline usable on a low-contrast frame
            warnings.warn(f"Frame {t}: registration failed ({exc}); reusing last shift")
            shift_full = shifts[t - 1]

        shifts[t] = shift_full
        previous = ndi.shift(
            current, shift=tuple(shift_full / ds), order=1,
            mode="constant", cval=0.0, prefilter=False,
        )

    return shifts


def write_stabilized_volume(
    volume_tzyx: da.Array,
    shifts_yx: np.ndarray,
    output_path: Path,
    *,
    description: str = "Writing stabilized volume",
) -> Any:
    """Apply the per-frame XY shift to every Z slice and write a disk-backed Zarr."""
    output = _new_zarr(volume_tzyx.shape, np.float32, output_path)
    for t in tqdm(range(int(volume_tzyx.shape[0])), desc=description):
        volume = _compute(volume_tzyx[t]).astype(np.float32, copy=False)  # (Z, Y, X)
        dy, dx = shifts_yx[t]
        finite = volume[np.isfinite(volume)]
        cval = float(np.percentile(finite, 1.0)) if finite.size else 0.0
        output[t] = ndi.shift(
            volume, shift=(0.0, dy, dx), order=1,
            mode="constant", cval=cval, prefilter=False,
        )
    return output


# ---------------------------------------------------------------------------
# Global intensity window (keeps the dim fish dark)
# ---------------------------------------------------------------------------
def estimate_intensity_bounds(
    volume_tzyx: Any, *, low_percentile: float, high_percentile: float,
    sample_frames: int,
) -> tuple[float, float]:
    """Estimate one global intensity window from frames sampled across time."""
    n_time = int(volume_tzyx.shape[0])
    count = max(1, min(n_time, int(sample_frames)))
    frame_idx = np.unique(np.linspace(0, n_time - 1, count).round().astype(int))

    rng = np.random.default_rng(0)
    pooled: list[np.ndarray] = []
    for t in frame_idx:
        values = _compute(volume_tzyx[t]).astype(np.float32, copy=False).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        if values.size > 1_000_000:  # cap memory; a sample is enough for percentiles
            values = rng.choice(values, 1_000_000, replace=False)
        pooled.append(values)

    if not pooled:
        return 0.0, 1.0

    everything = np.concatenate(pooled)
    low = float(np.percentile(everything, low_percentile))
    high = float(np.percentile(everything, high_percentile))
    if not np.isfinite(low):
        low = 0.0
    if not np.isfinite(high) or high <= low:
        high = low + 1.0
    return low, high


def _drop_dim_objects(
    mask: np.ndarray, intensity: np.ndarray, min_mean: float
) -> np.ndarray:
    """Zero out labels whose mean normalized intensity is below ``min_mean``."""
    ids = _positive_labels(mask)
    if ids.size == 0:
        return mask
    means = np.asarray(ndi.mean(intensity, labels=mask, index=ids), dtype=float)
    dim = ids[means < min_mean]
    if dim.size:
        mask = mask.copy()
        mask[np.isin(mask, dim)] = 0
    return mask


def _filter_slice_objects(
    mask: np.ndarray,
    *,
    min_area: float,
    max_area: float,
    max_eccentricity: float,
    min_solidity: float,
) -> np.ndarray:
    """Zero out 2D objects on one Z slice that are the wrong size or not circular.

    Runs on a single slice's label image (after the dim-object gate) before
    stitching. An object is removed when ANY of these hold:

      * pixel ``area`` < ``min_area`` or > ``max_area`` -- Ultrack's MIN_AREA /
        MAX_AREA reused as 2D pixel-area bounds;
      * ``eccentricity`` > ``max_eccentricity`` (0 = circle, ->1 = line), which
        drops elongated masks such as two cells merged side by side. Skipped when
        ``max_eccentricity`` >= 1;
      * ``solidity`` < ``min_solidity`` (area / convex-hull area; 1 = convex),
        which drops ragged or pinched masks. Skipped when ``min_solidity`` <= 0.

    The size gate is applied first with a fast bincount, and the moment-/hull-based
    shape metrics are measured only on the objects that pass it -- both to save
    work and to keep the convex hull off degenerate slivers. ``eccentricity`` and
    ``solidity`` come from image moments and the convex hull, which stay meaningful
    on the small pixelated blobs here; the perimeter-based circularity
    ``4*pi*area/perimeter**2`` is deliberately avoided because its perimeter term
    is unreliable at this object size.
    """
    if mask.max() == 0:
        return mask

    # --- size gate: reuse MIN_AREA / MAX_AREA as 2D pixel-area bounds ---
    counts = np.bincount(mask.ravel())
    counts[0] = 0  # ignore background
    bad_size = np.nonzero((counts > 0) & ((counts < min_area) | (counts > max_area)))[0]

    out = mask
    if bad_size.size:
        out = mask.copy()
        out[np.isin(mask, bad_size)] = 0
        if out.max() == 0:
            return out

    # --- shape gate: measured only on the size-passing objects ---
    check_ecc = max_eccentricity < 1.0
    check_sol = min_solidity > 0.0
    if not (check_ecc or check_sol):
        return out

    props: tuple[str, ...] = ("label",)
    if check_ecc:
        props += ("eccentricity",)
    if check_sol:
        props += ("solidity",)
    table = regionprops_table(out, properties=props)

    bad = np.zeros(table["label"].shape, dtype=bool)
    if check_ecc:
        bad |= np.asarray(table["eccentricity"], dtype=float) > max_eccentricity
    if check_sol:
        bad |= np.asarray(table["solidity"], dtype=float) < min_solidity

    drop = table["label"][bad]
    if drop.size:
        out = out if out is not mask else mask.copy()
        out[np.isin(out, drop)] = 0
    return out


# ---------------------------------------------------------------------------
# 2D -> 3D stitching (Cellpose-style IoU, with a Z-gap tolerance)
# ---------------------------------------------------------------------------
def _iou(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IoU between every labelled object in ``a`` and every object in ``b``.

    Returns ``(iou, a_ids, b_ids)`` where ``iou[i, j]`` is the IoU of label
    ``a_ids[i]`` with ``b_ids[j]``. Computed from a joint histogram, so cost is
    O(pixels), not O(objects**2 * pixels).
    """
    a_ids = _positive_labels(a)
    b_ids = _positive_labels(b)
    if a_ids.size == 0 or b_ids.size == 0:
        return np.zeros((a_ids.size, b_ids.size), np.float64), a_ids, b_ids

    both = (a > 0) & (b > 0)
    ai = np.searchsorted(a_ids, a[both])
    bi = np.searchsorted(b_ids, b[both])
    inter = np.zeros((a_ids.size, b_ids.size), dtype=np.int64)
    np.add.at(inter, (ai, bi), 1)

    area_a = np.bincount(a.ravel(), minlength=int(a_ids.max()) + 1)[a_ids]
    area_b = np.bincount(b.ravel(), minlength=int(b_ids.max()) + 1)[b_ids]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1), a_ids, b_ids


def stitch_gap(
    masks_zyx: np.ndarray, *, iou_threshold: float, max_gap: int
) -> np.ndarray:
    """Stitch a stack of 2D label images into 3D, bridging up to ``max_gap`` gaps.

    Walking up Z, each slice's 2D objects are matched against the most recent
    appearance of every still-active label by IoU. A matched object inherits the
    label; an unmatched object starts a new one. A label whose last appearance was
    up to ``max_gap`` empty slices ago is still eligible, which reconnects a cell
    that dropped out on an intermediate slice.

    Because matching is per-object on the 2D masks (never on a dilated/merged
    foreground), two cells that touch in XY but are distinct 2D objects are kept
    separate, and each reference label can be claimed by at most one object per
    slice -- so gap bridging cannot cause horizontal merging. With ``max_gap == 0``
    this reduces to Cellpose's ``stitch3D``.
    """
    masks_zyx = np.asarray(masks_zyx)
    n_z = masks_zyx.shape[0]
    out = np.zeros_like(masks_zyx, dtype=np.int32)

    ref = np.zeros(masks_zyx.shape[1:], dtype=np.int32)  # carried-forward labels
    age: dict[int, int] = {}                             # label -> slices since last seen
    next_label = 1

    for z in range(n_z):
        cur = masks_zyx[z]
        cur_ids = _positive_labels(cur)
        seen: set[int] = set()

        if cur_ids.size:
            assigned: dict[int, int] = {}

            if age:  # try to inherit a label from a recent slice
                iou, _, ref_present = _iou(cur, ref)
                if ref_present.size:
                    best = iou.argmax(axis=1)
                    best_iou = iou.max(axis=1)
                    holder: set[int] = set()
                    # Greedy: strongest overlaps first; each ref label taken once.
                    for k in np.argsort(-best_iou):
                        if float(best_iou[k]) < iou_threshold:
                            break
                        rlab = int(ref_present[best[k]])
                        if rlab in holder:
                            continue
                        assigned[int(cur_ids[k])] = rlab
                        holder.add(rlab)

            lut = np.zeros(int(cur.max()) + 1, dtype=np.int32)
            for cid in cur_ids.tolist():
                if cid not in assigned:
                    assigned[cid] = next_label
                    next_label += 1
                lut[cid] = assigned[cid]
            out[z] = lut[cur]
            seen = set(assigned.values())

        # Carry forward labels that are absent this slice but still young enough.
        keep = np.array(
            [lab for lab, a in age.items() if lab not in seen and a + 1 <= max_gap],
            dtype=np.int64,
        )
        new_ref = np.where(np.isin(ref, keep), ref, 0) if keep.size else np.zeros_like(ref)
        present = out[z] > 0
        new_ref[present] = out[z][present]  # most-recent appearance wins on overlap

        age = {int(lab): age[int(lab)] + 1 for lab in keep.tolist()}
        age.update(dict.fromkeys(seen, 0))
        ref = new_ref

    return out


def segment_and_stitch(
    stabilized: Any,
    *,
    output_path: Path,
    pretrained_model: str,
    diameter: float | None,
    gpu: bool,
    batch_size: int,
    flow_threshold: float,
    cellprob_threshold: float,
    norm_low: float,
    norm_high: float,
    min_cell_norm_intensity: float,
    slice_filter: bool,
    slice_min_area: float,
    slice_max_area: float,
    slice_max_eccentricity: float,
    slice_min_solidity: float,
    iou_threshold: float,
    max_gap: int,
) -> Any:
    """Segment every Z slice in 2D with Cellpose-SAM, then stitch into 3D labels.

    Each slice is normalized with the SAME global window so dim fish stays dark
    (Cellpose's own normalization is disabled). On every slice, objects dimmer
    than ``min_cell_norm_intensity`` are dropped and then -- when ``slice_filter``
    is set -- objects outside ``[slice_min_area, slice_max_area]`` pixels or less
    circular than the ``slice_max_eccentricity`` / ``slice_min_solidity`` bounds
    are removed (see ``_filter_slice_objects``), all before stitching. Produces a
    (T, Z, Y, X) Zarr of 3D instance labels.
    """
    try:
        from cellpose import models
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Cellpose-SAM is required: `pip install 'cellpose>=4'`."
        ) from exc

    # Cellpose 4 removed the models.Cellpose wrapper; CellposeModel loads the
    # Cellpose-SAM ("cpsam") weights by default, or a fine-tuned model by path.
    model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)
    _silence_cellpose_seed_warnings()  # after model construction; see the helper

    n_time = int(stabilized.shape[0])
    n_z = int(stabilized.shape[1])
    span = float(norm_high - norm_low) or 1.0
    labels = _new_zarr(stabilized.shape, np.int32, output_path)

    for t in tqdm(range(n_time), desc="Cellpose 2D + Z-stitch"):
        volume = _compute(stabilized[t]).astype(np.float32, copy=False)  # (Z, Y, X)
        normalized = np.clip((volume - norm_low) / span, 0.0, 1.0)

        # One eval over the whole stack (Cellpose batches the slices internally).
        # cpsam takes no channels argument, and with diameter=None it segments at
        # native resolution -- both are handled inside the model.
        masks = model.eval(
            [normalized[z] for z in range(n_z)],
            diameter=diameter,
            flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
            normalize=False,  # already globally normalized above
            batch_size=batch_size,
        )[0]

        slices = []
        for z in range(n_z):
            mask = np.asarray(masks[z], dtype=np.int32)
            if min_cell_norm_intensity > 0 and mask.max() > 0:
                mask = _drop_dim_objects(mask, normalized[z], min_cell_norm_intensity)
            if slice_filter and mask.max() > 0:
                mask = _filter_slice_objects(
                    mask,
                    min_area=slice_min_area, max_area=slice_max_area,
                    max_eccentricity=slice_max_eccentricity,
                    min_solidity=slice_min_solidity,
                )
            slices.append(mask)

        labels[t] = stitch_gap(
            np.stack(slices, axis=0),
            iou_threshold=iou_threshold, max_gap=max_gap,
        )

    return labels


# ---------------------------------------------------------------------------
# Ultrack configuration
# ---------------------------------------------------------------------------
def make_ultrack_config(*, working_dir: Path, n_time: int) -> MainConfig:
    """Build an Ultrack MainConfig from the USER SETTINGS above."""
    config = MainConfig()

    config.data_config.working_dir = working_dir
    config.data_config.n_workers = DATA_WORKERS

    seg = config.segmentation_config
    seg.min_area = MIN_AREA
    seg.max_area = MAX_AREA
    seg.min_area_factor = MIN_AREA_FACTOR
    seg.min_frontier = MIN_FRONTIER
    seg.threshold = SEG_THRESHOLD
    seg.max_noise = MAX_NOISE
    seg.anisotropy_penalization = ANISOTROPY_PENALIZATION
    seg.random_seed = SEG_RANDOM_SEED
    seg.n_workers = SEG_WORKERS

    link = config.linking_config
    link.max_distance = MAX_DISTANCE
    link.max_neighbors = MAX_NEIGHBORS
    link.distance_weight = DISTANCE_WEIGHT
    link.z_score_threshold = LINK_Z_SCORE_THRESHOLD
    link.n_workers = LINK_WORKERS

    track = config.tracking_config
    track.solver_name = "" if SOLVER == "auto" else SOLVER
    track.appear_weight = APPEAR_WEIGHT
    track.disappear_weight = DISAPPEAR_WEIGHT
    track.division_weight = DIVISION_WEIGHT
    track.n_threads = TRACKING_THREADS
    track.overlap_size = OVERLAP_SIZE
    track.solution_gap = SOLUTION_GAP
    track.time_limit = TIME_LIMIT
    track.method = TRACKING_METHOD
    track.link_function = LINK_FUNCTION
    track.power = POWER
    track.bias = BIAS
    if WINDOW_SIZE > 0 and n_time > WINDOW_SIZE:
        track.window_size = WINDOW_SIZE

    return config


# ---------------------------------------------------------------------------
# Post-tracking filtering
# ---------------------------------------------------------------------------
def _normalize_graph(graph: Any) -> dict[int, list[int]]:
    """Coerce an Ultrack/napari lineage graph to ``{child: [parent, ...]}``.

    ``Tracker.to_tracks_layer`` returns the lineage as ``{child_id: parent(s)}``,
    but depending on the installed Ultrack version each value may be a single
    scalar parent id OR a list of them. napari's Tracks layer accepts either, but
    the rest of this script iterates the parents, so every value is normalized to
    a list of Python ints here (called once, right after tracking). A scalar
    becomes a one-element list; ``None`` / missing becomes an empty list. Without
    this, a scalar-valued graph raises ``TypeError: 'int' object is not iterable``
    the first time the parents are iterated.
    """
    normalized: dict[int, list[int]] = {}
    for child, parents in (graph or {}).items():
        if parents is None:
            parent_list: list[int] = []
        else:
            try:  # already an iterable of parent ids
                parent_list = [int(p) for p in parents]
            except TypeError:  # a single scalar parent id (int / np.integer)
                parent_list = [int(parents)]
        normalized[int(child)] = parent_list
    return normalized


def _subgraph(graph: dict[int, list[int]] | None, ids: set[int]) -> dict[int, list[int]]:
    """Restrict a ``{child: [parents]}`` lineage graph to children/parents in ``ids``.

    Children not in ``ids`` are dropped, parents not in ``ids`` are removed from
    each remaining child, and children left with no parents are dropped too.
    """
    restricted = {
        int(c): [int(p) for p in parents if int(p) in ids]
        for c, parents in (graph or {}).items()
        if int(c) in ids
    }
    return {child: parents for child, parents in restricted.items() if parents}


def _duplicate_segmentation(source: Any, output_path: Path) -> Any:
    """Copy a (T, Z, Y, X) label Zarr to a new store, one frame at a time.

    ``filter_tracks`` rewrites the tracked segmentation in place, so the unfiltered
    volume is snapshotted here first (one frame in RAM at a time) to keep both the
    before- and after-filter versions for the napari view and for export.
    """
    duplicate = _new_zarr(source.shape, source.dtype, output_path)
    for t in tqdm(
        range(int(source.shape[0])), desc="Snapshotting unfiltered segmentation"
    ):
        duplicate[t] = source[t]
    return duplicate


def filter_tracks(
    tracks_df: pd.DataFrame,
    graph: dict[int, list[int]],
    tracked_segments: Any,
    *,
    min_frames: int,
    min_mean_speed: float,
) -> tuple[pd.DataFrame, dict[int, list[int]]]:
    """Drop finished tracks that are too short or barely move, and erase the
    removed cells from ``tracked_segments`` (whose label values equal track ids).

    A track is kept only if it spans at least ``min_frames`` frames AND its mean
    per-frame centroid step (raw z,y,x voxel distance) is at least
    ``min_mean_speed`` voxels/frame. The segmentation Zarr is rewritten in place,
    so the napari view and the exported ``tracked_segments.zarr`` both reflect the
    filtering.
    """
    if tracks_df.empty:
        return tracks_df, graph

    spatial = [c for c in ("z", "y", "x") if c in tracks_df.columns]
    keep: set[int] = set()
    for tid, group in tracks_df.groupby("track_id"):
        group = group.sort_values("t")
        steps = np.linalg.norm(np.diff(group[spatial].to_numpy(float), axis=0), axis=1)
        dt = np.diff(group["t"].to_numpy(float))
        speed = float(np.mean(steps / np.maximum(dt, 1.0))) if steps.size else 0.0
        if len(group) >= min_frames and speed >= min_mean_speed:
            keep.add(int(tid))

    all_ids = {int(i) for i in tracks_df["track_id"].unique()}
    removed = all_ids - keep
    print(
        f"Track filter: kept {len(keep)}/{len(all_ids)} tracks "
        f"(removed {len(removed)} shorter than {min_frames} frames "
        f"or slower than {min_mean_speed} voxels/frame)."
    )

    if removed:
        # Relabel via a lookup table: kept ids map to themselves, the rest to 0.
        lut = np.zeros(max(all_ids) + 1, dtype=tracked_segments.dtype)
        for tid in keep:
            lut[tid] = tid
        for t in tqdm(range(int(tracked_segments.shape[0])), desc="Filtering segmentation"):
            tracked_segments[t] = lut[tracked_segments[t]]

    tracks_df = tracks_df[tracks_df["track_id"].isin(keep)].reset_index(drop=True)
    graph = _subgraph(graph, keep)
    return tracks_df, graph


# ---------------------------------------------------------------------------
# napari display
# ---------------------------------------------------------------------------
def _add_movie_layers(
    viewer: Any, *, name: str, prefix: str, bundle: dict[str, Any]
) -> None:
    """Add every layer for one movie to a shared viewer.

    Each layer is tagged with ``metadata["movie"] = name`` plus its intended
    ("base") visibility and is added hidden; ``_show_movie`` later reveals the
    active movie's layers (honouring each layer's base visibility). Layer names
    are ``prefix``-ed so a multi-movie layer list stays readable.

    The tracked segmentation and trajectories are shown as just two toggleable
    sets -- the cells/tracks ``filter_tracks`` KEPT and the ones it REMOVED. There
    is no separate "all" set: turn both on together to see the pre-filter picture.
    """
    scale = bundle["napari_scale"]
    axes = bundle["axis_labels"]
    tail = bundle["tail_length"]

    def meta(base_visible: bool) -> dict[str, Any]:
        return {"movie": name, "base_visible": base_visible}

    def add_image(array: Any, layer: str, *, base_visible: bool = True, **kw: Any) -> None:
        viewer.add_image(
            _as_dask(array), name=prefix + layer, visible=False,
            scale=scale, axis_labels=axes, metadata=meta(base_visible), **kw,
        )

    def add_labels(array: Any, layer: str, *, base_visible: bool = True, **kw: Any) -> None:
        viewer.add_labels(
            _label_layer(array), name=prefix + layer, visible=False,
            scale=scale, axis_labels=axes, metadata=meta(base_visible), **kw,
        )

    def add_tracks(
        df: pd.DataFrame | None, graph: dict[int, list[int]] | None, layer: str
    ) -> None:
        if df is None or df.empty:
            return
        cols = [c for c in ("track_id", "t", "z", "y", "x") if c in df.columns]
        viewer.add_tracks(
            df[cols].to_numpy(), graph=graph or {}, name=prefix + layer,
            tail_length=tail, tail_width=3, visible=False,
            scale=scale, axis_labels=axes, metadata=meta(True),
        )

    # Raw stabilized image (plus an optional separate registration/context channel).
    if bundle["stabilized_context"] is not None:
        add_image(bundle["stabilized_context"], "stabilized context", colormap="gray")
        add_image(
            bundle["stabilized"], "stabilized track channel",
            colormap="magenta", blending="additive",
        )
    else:
        add_image(bundle["stabilized"], "stabilized track channel", colormap="gray")

    # Intermediate layers (3D labels are shown only in the no-tracking seg. test).
    if bundle["labels_3d"] is not None:
        add_labels(
            bundle["labels_3d"], "3D labels (Cellpose stitched)",
            opacity=0.4, base_visible=bundle["tracked_segments"] is None,
        )
    if bundle["foreground"] is not None:
        add_labels(
            bundle["foreground"], "foreground used by Ultrack",
            opacity=0.25, base_visible=False,
        )
    if bundle["contours"] is not None:
        add_image(
            bundle["contours"], "contours used by Ultrack",
            opacity=0.55, blending="additive", base_visible=False,
        )

    # Tracked segmentation + trajectories: KEPT (post-filter) and REMOVED only.
    kept_seg = bundle["tracked_segments"]
    kept_df = bundle["tracks_df"]
    unf_seg = bundle["tracked_segments_unfiltered"]
    unf_df = bundle["tracks_df_unfiltered"]

    if kept_seg is not None:
        add_labels(kept_seg, "segments — kept", opacity=0.28)
    add_tracks(kept_df, bundle["graph"], "trajectories — kept")

    if unf_df is not None and not unf_df.empty:
        kept_ids = (
            set(kept_df["track_id"].astype(int))
            if kept_df is not None and not kept_df.empty else set()
        )
        removed_ids = set(unf_df["track_id"].astype(int)) - kept_ids
        if removed_ids:
            # Removed voxels = unfiltered cells that are background in the kept
            # volume; stays lazy (napari computes one frame at a time).
            if unf_seg is not None:
                removed_seg = (
                    da.where(_as_dask(kept_seg) == 0, _as_dask(unf_seg), 0)
                    if kept_seg is not None else unf_seg
                )
                add_labels(removed_seg, "segments — REMOVED by filter", opacity=0.5)
            removed_df = unf_df[unf_df["track_id"].isin(removed_ids)].reset_index(drop=True)
            add_tracks(
                removed_df, _subgraph(bundle["graph_unfiltered"], removed_ids),
                "trajectories — REMOVED by filter",
            )

    if kept_seg is not None and (kept_df is None or kept_df.empty):
        warnings.warn(f"{name}: Ultrack returned no tracks; inspect the foreground layer.")


def _show_movie(viewer: Any, active: str) -> None:
    """Make only ``active``'s layers visible, each at its own base visibility."""
    for layer in viewer.layers:
        movie = layer.metadata.get("movie")
        if movie is not None:
            layer.visible = movie == active and layer.metadata.get("base_visible", True)


def view_movies(bundles: list[dict[str, Any]]) -> None:
    """Open ONE napari window showing every processed movie.

    All movies' layers live in a single viewer. With more than one movie a
    "Movie" dropdown selects which movie is visible (no extra windows); with a
    single movie the dropdown is omitted. Layers are added hidden and revealed by
    ``_show_movie`` so each keeps its intended default visibility.
    """
    if not bundles:
        return

    multi = len(bundles) > 1
    title = f"Ultrack — {len(bundles)} movies" if multi else bundles[0]["title"]
    viewer = napari.Viewer(title=title)

    for bundle in bundles:
        name = bundle["name"]
        _add_movie_layers(
            viewer, name=name, prefix=f"{name}: " if multi else "", bundle=bundle,
        )

    _show_movie(viewer, bundles[0]["name"])

    if multi:
        from magicgui.widgets import ComboBox

        names = [bundle["name"] for bundle in bundles]
        switcher = ComboBox(label="Movie", choices=names, value=names[0])
        switcher.changed.connect(lambda active: _show_movie(viewer, active))
        viewer.window.add_dock_widget(switcher, area="left", name="Movie")

    napari.run()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _resolve_time_subset(
    subset: int | tuple[int, int] | None, n_total: int
) -> tuple[int, int]:
    """Resolve TIME_SUBSET into a half-open ``[start, stop)`` frame range.

    ``None``           -> the whole movie, ``[0, n_total)``.
    an ``int`` N       -> the single frame N, i.e. ``[N, N + 1)``.
    a ``(lo, hi)`` pair-> the INCLUSIVE range lo..hi, i.e. ``[lo, hi + 1)``.

    Raises ``IndexError`` if the request falls outside ``0..n_total - 1`` or is
    reversed (``stop < start``).
    """
    if subset is None:
        return 0, n_total
    lo, hi = (subset, subset) if isinstance(subset, int) else subset
    lo, hi = int(lo), int(hi)
    if not 0 <= lo <= hi < n_total:
        raise IndexError(
            f"TIME_SUBSET {subset!r} is outside 0..{n_total - 1} "
            f"(need 0 <= start <= stop)."
        )
    return lo, hi + 1


def process_movie(czi_path: Path, *, output_dir: Path, name: str) -> dict[str, Any]:
    """Run the full pipeline on one CZI and return its napari layer bundle.

    ``name`` labels the movie in napari (dropdown + layer-name prefix). Every
    output is written under ``output_dir``; nothing is displayed here -- the
    returned bundle is handed to ``view_movies`` so all movies share one window.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "ultrack_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    registration_channel = (
        TRACK_CHANNEL if REGISTRATION_CHANNEL is None else REGISTRATION_CHANNEL
    )

    image = BioImage(str(czi_path), reconstruct_mosaic=True)
    image.set_scene(SCENE)
    print(f"Scenes: {image.scenes}")
    print(f"Selected scene: {image.current_scene}")
    print(f"Scene dimensions: {image.dims}")
    print(f"Channels: {image.channel_names}")

    n_channels = int(image.dims.C)
    # NB: the loop variable here must NOT be ``name`` -- that is this function's
    # movie-name parameter (used for the napari "Movie" dropdown, the layer-name
    # prefix, and the returned bundle). Letting the loop rebind it labelled every
    # movie "registration", which silently broke multi-movie runs.
    for role, channel in (("track", TRACK_CHANNEL), ("registration", registration_channel)):
        if not 0 <= channel < n_channels:
            raise IndexError(f"{role} channel {channel} is outside 0..{n_channels - 1}")

    track_tzyx = image.get_image_dask_data("TZYX", C=TRACK_CHANNEL)
    registration_tzyx = image.get_image_dask_data("TZYX", C=registration_channel)

    # Select the requested slice of time points (TIME_SUBSET). The single-frame
    # quick test is just the length-1 case: any slice shorter than two frames
    # cannot be linked, so we segment it and show the result without tracking.
    n_total = int(track_tzyx.shape[0])
    start, stop = _resolve_time_subset(TIME_SUBSET, n_total)
    track_tzyx = track_tzyx[start:stop]
    registration_tzyx = registration_tzyx[start:stop]
    n_frames = stop - start
    scope = f"frame {start}" if n_frames == 1 else f"frames {start}-{stop - 1}"
    subset_label = "" if TIME_SUBSET is None else f" — {scope}"

    segmentation_only = n_frames < 2
    if TIME_SUBSET is not None:
        skip = "; tracking will be skipped" if segmentation_only else ""
        print(f"TIME SUBSET: processing {n_frames} frame(s) ({scope}){skip}.")
    elif segmentation_only:
        raise ValueError(
            "Tracking requires at least two time points (set TIME_SUBSET to a "
            "single frame or a (start, stop) range to test on a subset)."
        )

    # Fail fast before the multi-hour segmentation: a tracking run ends in
    # labels_to_contours, which Ultrack executes on the GPU when CuPy is present
    # and which then requires cuCIM. Segmentation-only runs skip tracking (and
    # thus labels_to_contours), so they are exempt. See _check_gpu_imgproc_stack.
    if not segmentation_only:
        _check_gpu_imgproc_stack()

    if int(track_tzyx.shape[1]) < 2:
        warnings.warn("The selected scene has a single Z plane; stitching is a no-op.")

    # Estimate the global XY shift from the registration channel's 2D projection.
    if NO_STABILIZATION:
        shifts_yx = np.zeros((int(track_tzyx.shape[0]), 2), dtype=float)
    else:
        shifts_yx = estimate_xy_shifts(
            _project_to_2d(registration_tzyx),
            downsample=REGISTRATION_DOWNSAMPLE, sigma=REGISTRATION_SIGMA,
            upsample_factor=REGISTRATION_UPSAMPLE, max_shift_px=MAX_REGISTRATION_SHIFT,
        )

    pd.DataFrame(
        {
            "t": np.arange(len(shifts_yx)),
            "shift_y_pixels": shifts_yx[:, 0],
            "shift_x_pixels": shifts_yx[:, 1],
        }
    ).to_csv(output_dir / "stabilization_shifts.csv", index=False)

    # Apply the same XY shift to every Z slice of the 3D volume.
    stabilized_path = output_dir / "stabilized.zarr"
    stabilized = write_stabilized_volume(
        track_tzyx, shifts_yx, stabilized_path,
        description="Writing stabilized track volume",
    )

    stabilized_context = None
    if registration_channel != TRACK_CHANNEL:
        stabilized_context = write_stabilized_volume(
            registration_tzyx, shifts_yx, output_dir / "stabilized_context.zarr",
            description="Writing stabilized context volume",
        )

    pixel_sizes = image.physical_pixel_sizes
    pixel_x = _positive_float(pixel_sizes.X, 1.0)
    pixel_y = _positive_float(pixel_sizes.Y, pixel_x)
    pixel_z = _positive_float(pixel_sizes.Z, pixel_x)

    # Scales normalized by X so MAX_DISTANCE stays in ~X-pixel units (order z, y, x).
    spatial_scale = (pixel_z / pixel_x, pixel_y / pixel_x, 1.0)
    napari_scale = (1.0, pixel_z, pixel_y, pixel_x)
    axis_labels = ("t", "z", "y", "x")

    # Everything downstream of stabilization operates on sqrt(intensity). The
    # compression is applied here, once, and every intensity consumer below (the
    # global window and the per-slice Cellpose normalization inside
    # segment_and_stitch) reads from this transformed volume. The REAL
    # `stabilized` volume is left untouched and is what gets handed to napari
    # further down, so on-screen you still inspect the raw image. This stays lazy
    # -- a dask view over the stabilized Zarr -- because sqrt is a cheap
    # elementwise op, so no second 4D volume is written to disk.
    stabilized_sqrt = da.sqrt(_as_dask(stabilized))

    # One global intensity window keeps dim fish dark for every slice. It is
    # estimated on the sqrt volume so the window matches the data Cellpose
    # actually sees; the bounds below are therefore in sqrt-intensity units, not
    # raw counts.
    norm_low, norm_high = estimate_intensity_bounds(
        stabilized_sqrt, low_percentile=NORM_LOW_PERCENTILE,
        high_percentile=NORM_HIGH_PERCENTILE, sample_frames=NORM_SAMPLE_FRAMES,
    )
    print(
        f"Global intensity window (sqrt-intensity units): "
        f"[{norm_low:.3f}, {norm_high:.3f}] "
        f"(p{NORM_LOW_PERCENTILE:g}-p{NORM_HIGH_PERCENTILE:g})"
    )

    # Segment each Z slice in 2D and stitch into 3D labels. This runs on the
    # sqrt volume (normalized with the sqrt-domain window above); napari is
    # still shown the raw `stabilized` volume.
    labels_3d_path = output_dir / "labels_3d.zarr"
    labels_3d = segment_and_stitch(
        stabilized_sqrt, output_path=labels_3d_path,
        pretrained_model=CELLPOSE_MODEL, diameter=CELLPOSE_DIAMETER, gpu=CELLPOSE_GPU,
        batch_size=CELLPOSE_BATCH_SIZE,
        flow_threshold=CELLPOSE_FLOW_THRESHOLD,
        cellprob_threshold=CELLPOSE_CELLPROB_THRESHOLD,
        norm_low=norm_low, norm_high=norm_high,
        min_cell_norm_intensity=MIN_CELL_NORM_INTENSITY,
        slice_filter=SLICE_FILTER_ENABLED,
        slice_min_area=MIN_AREA, slice_max_area=MAX_AREA,
        slice_max_eccentricity=SLICE_MAX_ECCENTRICITY,
        slice_min_solidity=SLICE_MIN_SOLIDITY,
        iou_threshold=STITCH_IOU_THRESHOLD, max_gap=STITCH_MAX_Z_GAP,
    )

    base_parameters = {
        "czi": str(czi_path),
        "output": str(output_dir),
        "scene": SCENE,
        "track_channel": TRACK_CHANNEL,
        "registration_channel": registration_channel,
        "no_stabilization": NO_STABILIZATION,
        "downstream_intensity_transform": "sqrt",  # window + segmentation run on sqrt(intensity)
        "time_subset": list(TIME_SUBSET) if isinstance(TIME_SUBSET, tuple) else TIME_SUBSET,
        "time_range_inclusive": [start, stop - 1],
        "n_frames": n_frames,
        "registration_downsample": REGISTRATION_DOWNSAMPLE,
        "registration_sigma": REGISTRATION_SIGMA,
        "registration_upsample": REGISTRATION_UPSAMPLE,
        "max_registration_shift": MAX_REGISTRATION_SHIFT,
        "cellpose_model": CELLPOSE_MODEL,
        "cellpose_diameter": CELLPOSE_DIAMETER,
        "cellpose_batch_size": CELLPOSE_BATCH_SIZE,
        "cellpose_gpu": CELLPOSE_GPU,
        "cellpose_flow_threshold": CELLPOSE_FLOW_THRESHOLD,
        "cellpose_cellprob_threshold": CELLPOSE_CELLPROB_THRESHOLD,
        "stitch_iou_threshold": STITCH_IOU_THRESHOLD,
        "stitch_max_z_gap": STITCH_MAX_Z_GAP,
        "norm_low_percentile": NORM_LOW_PERCENTILE,
        "norm_high_percentile": NORM_HIGH_PERCENTILE,
        "norm_window": (norm_low, norm_high),
        "min_cell_norm_intensity": MIN_CELL_NORM_INTENSITY,
        "slice_filter_enabled": SLICE_FILTER_ENABLED,
        "slice_filter_min_area": MIN_AREA,
        "slice_filter_max_area": MAX_AREA,
        "slice_max_eccentricity": SLICE_MAX_ECCENTRICITY,
        "slice_min_solidity": SLICE_MIN_SOLIDITY,
        "contour_sigma": CONTOUR_SIGMA,
        "physical_pixel_sizes_zyx": (pixel_z, pixel_y, pixel_x),
        "spatial_scale_zyx": spatial_scale,
    }

    # Too few frames to link: stop after segmentation and show the result.
    if segmentation_only:
        with open(output_dir / "run_parameters.json", "w", encoding="utf-8") as stream:
            json.dump({**base_parameters, "mode": "segmentation_test"}, stream, indent=2)
        print("Segmentation test complete. Wrote:")
        print(f"  {stabilized_path}")
        print(f"  {labels_3d_path}")
        return {
            "name": name,
            "title": f"{czi_path.name} — segmentation test (frame {start})",
            "stabilized": stabilized,
            "stabilized_context": stabilized_context,
            "labels_3d": labels_3d,
            "foreground": None,
            "contours": None,
            "tracked_segments": None,
            "tracks_df": None,
            "graph": None,
            "tracked_segments_unfiltered": None,
            "tracks_df_unfiltered": None,
            "graph_unfiltered": None,
            "napari_scale": napari_scale,
            "axis_labels": axis_labels,
            "tail_length": TAIL_LENGTH,
        }

    # Convert the 3D labels to Ultrack foreground/contours. Pass disk-backed store
    # paths when the installed Ultrack accepts them (keeps large volumes off-RAM).
    l2c_kwargs: dict[str, Any] = {"sigma": CONTOUR_SIGMA}
    l2c_params = inspect.signature(labels_to_contours).parameters
    if "foreground_store_or_path" in l2c_params:
        l2c_kwargs["foreground_store_or_path"] = str(output_dir / "foreground.zarr")
    if "contours_store_or_path" in l2c_params:
        l2c_kwargs["contours_store_or_path"] = str(output_dir / "contours.zarr")
    if "overwrite" in l2c_params:
        l2c_kwargs["overwrite"] = True
    foreground, contours = labels_to_contours(_as_dask(labels_3d), **l2c_kwargs)

    # Track in 3D.
    config = make_ultrack_config(working_dir=work_dir, n_time=int(stabilized.shape[0]))
    tracker = Tracker(config)
    tracker.track(
        foreground=foreground, contours=contours, scale=spatial_scale, overwrite="all",
    )

    tracks_df, graph = tracker.to_tracks_layer()
    # Ultrack returns the lineage graph as {child: parent(s)}, where each value
    # may be a single scalar parent id or a list depending on the installed
    # version. Everything below (the unfiltered snapshot, filter_tracks, the JSON
    # exports, and the napari overlays) iterates the parents, so normalize the
    # values to lists of ints once, here.
    graph = _normalize_graph(graph)
    tracked_segments = tracker.to_zarr(
        store_or_path=str(output_dir / "tracked_segments.zarr"), overwrite=True,
    )

    # Snapshot the UNFILTERED tracks and segmentation before filtering, so the
    # napari view (and these exports) can show what the filter removes.
    # filter_tracks rewrites tracked_segments in place, so the segmentation has to
    # be copied to a separate store first.
    tracks_df_unfiltered = tracks_df.copy()
    graph_unfiltered = {c: list(p) for c, p in graph.items()}
    tracked_segments_unfiltered = _duplicate_segmentation(
        tracked_segments, output_dir / "tracked_segments_unfiltered.zarr",
    )
    tracks_df_unfiltered.to_csv(output_dir / "tracks_unfiltered.csv", index=False)
    with open(
        output_dir / "tracks_graph_unfiltered.json", "w", encoding="utf-8"
    ) as stream:
        json.dump(graph_unfiltered, stream, indent=2)

    # Drop short / near-stationary tracks and erase those cells from the
    # segmentation, then export the FILTERED tracks and lineage graph.
    tracks_df, graph = filter_tracks(
        tracks_df, graph, tracked_segments,
        min_frames=MIN_TRACK_LENGTH, min_mean_speed=MIN_MEAN_SPEED,
    )
    tracks_df.to_csv(output_dir / "tracks.csv", index=False)
    with open(output_dir / "tracks_graph.json", "w", encoding="utf-8") as stream:
        json.dump(graph, stream, indent=2)

    parameters = {
        **base_parameters,
        "mode": "track",
        "data_workers": DATA_WORKERS,
        "min_area": MIN_AREA,
        "max_area": MAX_AREA,
        "min_area_factor": MIN_AREA_FACTOR,
        "min_frontier": MIN_FRONTIER,
        "seg_threshold": SEG_THRESHOLD,
        "max_noise": MAX_NOISE,
        "anisotropy_penalization": ANISOTROPY_PENALIZATION,
        "seg_random_seed": SEG_RANDOM_SEED,
        "seg_workers": SEG_WORKERS,
        "max_distance": MAX_DISTANCE,
        "max_neighbors": MAX_NEIGHBORS,
        "distance_weight": DISTANCE_WEIGHT,
        "link_z_score_threshold": LINK_Z_SCORE_THRESHOLD,
        "link_workers": LINK_WORKERS,
        "solver": SOLVER,
        "appear_weight": APPEAR_WEIGHT,
        "disappear_weight": DISAPPEAR_WEIGHT,
        "division_weight": DIVISION_WEIGHT,
        "tracking_threads": TRACKING_THREADS,
        "solution_gap": SOLUTION_GAP,
        "time_limit": TIME_LIMIT,
        "tracking_method": TRACKING_METHOD,
        "link_function": LINK_FUNCTION,
        "power": POWER,
        "bias": BIAS,
        "window_size": WINDOW_SIZE,
        "overlap_size": OVERLAP_SIZE,
        "min_track_length": MIN_TRACK_LENGTH,
        "min_mean_speed": MIN_MEAN_SPEED,
        "tail_length": TAIL_LENGTH,
        "tracks_unfiltered_csv": "tracks_unfiltered.csv",
        "tracks_graph_unfiltered_json": "tracks_graph_unfiltered.json",
        "tracked_segments_unfiltered_zarr": "tracked_segments_unfiltered.zarr",
    }
    with open(output_dir / "run_parameters.json", "w", encoding="utf-8") as stream:
        json.dump(parameters, stream, indent=2)

    n_tracks = int(tracks_df["track_id"].nunique()) if not tracks_df.empty else 0
    n_tracks_unfiltered = (
        int(tracks_df_unfiltered["track_id"].nunique())
        if not tracks_df_unfiltered.empty else 0
    )
    print(
        f"Exported {n_tracks} kept tracks ({len(tracks_df)} detections); "
        f"{n_tracks_unfiltered} tracks before filtering. Outputs in {output_dir}:"
    )
    print("  filtered:    tracks.csv, tracks_graph.json, tracked_segments.zarr")
    print(
        "  unfiltered:  tracks_unfiltered.csv, tracks_graph_unfiltered.json, "
        "tracked_segments_unfiltered.zarr"
    )

    return {
        "name": name,
        "title": f"{czi_path.name} — Ultrack neutrophils (3D){subset_label}",
        "stabilized": stabilized,
        "stabilized_context": stabilized_context,
        "labels_3d": labels_3d,
        "foreground": foreground,
        "contours": contours,
        "tracked_segments": tracked_segments,
        "tracks_df": tracks_df,
        "graph": graph,
        "tracked_segments_unfiltered": tracked_segments_unfiltered,
        "tracks_df_unfiltered": tracks_df_unfiltered,
        "graph_unfiltered": graph_unfiltered,
        "napari_scale": napari_scale,
        "axis_labels": axis_labels,
        "tail_length": TAIL_LENGTH,
    }


def main() -> None:
    """Process every movie in CZI_PATHS, then show them in one napari window.

    The movies are independent -- each is stabilized, segmented and tracked on its
    own and written to its own folder -- so a failure in one must not discard the
    others. A movie can fail for many reasons: a frame in which Cellpose segments
    no cells (leaving Ultrack with no foreground to track), a missing or unreadable
    file, a bad channel / TIME_SUBSET setting, a CUDA out-of-memory error, and so
    on.

    Each movie is therefore processed inside its own try/except. If it raises, the
    error and its traceback are printed, that movie is skipped, and processing
    continues -- so the run behaves exactly as if the failed movies had never been
    listed in CZI_PATHS. Only the movies that succeeded are opened in napari; a
    summary of what was skipped is printed at the end. If EVERY movie fails there
    is nothing to display, which is reported as an error (the same situation as an
    empty CZI_PATHS).

    KeyboardInterrupt (Ctrl-C) and SystemExit are deliberately NOT caught, so the
    whole batch can still be aborted.
    """
    paths = [Path(p).expanduser().resolve() for p in CZI_PATHS]
    if not paths:
        raise ValueError("CZI_PATHS is empty; add at least one .czi path.")

    base_output = (
        Path(OUTPUT_DIR).expanduser().resolve() if OUTPUT_DIR is not None else None
    )
    multi = len(paths) > 1

    bundles: list[dict[str, Any]] = []
    failures: list[tuple[Path, Exception]] = []
    for i, czi_path in enumerate(paths):
        label = f"Movie {i + 1}/{len(paths)}: {czi_path.name}"
        if multi:
            print(f"\n=== {label} ===")
        try:
            if not czi_path.is_file():
                raise FileNotFoundError(czi_path)
            if czi_path.suffix.lower() != ".czi":
                warnings.warn(f"Expected a .czi file, received: {czi_path.name}")

            output_dir = (
                base_output / czi_path.stem if base_output is not None
                else czi_path.with_name(f"{czi_path.stem}_ultrack")
            )
            name = f"{i + 1}. {czi_path.stem}" if multi else czi_path.stem
            bundles.append(process_movie(czi_path, output_dir=output_dir, name=name))
        except Exception as exc:  # noqa: BLE001 -- isolate each movie; see docstring
            failures.append((czi_path, exc))
            print(
                f"\n!!! Skipping {label}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc()

    if failures:
        print(
            f"\n{len(failures)} of {len(paths)} movie(s) failed and were skipped:",
            file=sys.stderr,
        )
        for czi_path, exc in failures:
            print(
                f"  - {czi_path.name}: {type(exc).__name__}: {exc}", file=sys.stderr
            )

    if not bundles:
        raise SystemExit(
            f"All {len(paths)} movie(s) failed; nothing to display "
            "(see the errors above)."
        )

    if failures:
        print(f"\nProceeding to napari with {len(bundles)} of {len(paths)} movie(s).")
    else:
        print(f"\nProceeding to napari with {len(bundles)} movie(s).")
    view_movies(bundles)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(
            "This script does not accept command-line parameters. "
            "Edit the USER SETTINGS block at the top of the file instead."
        )
    main()
