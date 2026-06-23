#!/usr/bin/env python3
"""Stabilize a CZI timelapse, track neutrophils with Ultrack, and inspect in napari.

Edit the USER SETTINGS block near the top of this file, then run:

    python fish_neutrophil_ultrack.py

For a 3-D CZI, tracking is volumetric by default. Set ``PROJECT_2D = True``
to track a maximum-intensity projection instead. Command-line parameters are
not accepted.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import dask.array as da
import napari
import numpy as np
import pandas as pd
from bioio import BioImage
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation
from tqdm.auto import tqdm
from ultrack import MainConfig, Tracker
from ultrack.imgproc import detect_foreground, robust_invert
from ultrack.utils.array import array_apply, create_zarr


# =============================================================================
# USER SETTINGS — edit these values before running the script
# =============================================================================

# Input and output
CZI_PATH = Path("fish.czi")
OUTPUT_DIR: Path | None = None  # None -> <CZI filename>_ultrack beside the CZI

# CZI selection and processing mode
SCENE = 0
TRACK_CHANNEL = 0
REGISTRATION_CHANNEL: int | None = None  # None -> use TRACK_CHANNEL
PROJECT_2D = False  # False tracks the full 3-D volume when Z has multiple planes
NO_STABILIZATION = False

# Registration settings (simple global XY translation)
REGISTRATION_DOWNSAMPLE = 2
REGISTRATION_SIGMA = 4.0
REGISTRATION_UPSAMPLE = 10
MAX_REGISTRATION_SHIFT = 50.0

# Classical foreground and contour generation for Ultrack
FOREGROUND_SIGMA = 12.0  # Choose larger than a neutrophil radius
BOUNDARY_SIGMA = 1.5
MIN_FOREGROUND = 0.0
KEEP_HISTOGRAM_MODE = False

# Ultrack settings. Areas are pixels in 2-D and voxels in 3-D.
MIN_AREA = 20
MAX_AREA = 20_000
MAX_DISTANCE = 25.0  # Maximum frame-to-frame movement in X-pixel units
MAX_NEIGHBORS = 10
WINDOW_SIZE = 50  # Set to 0 to solve the whole movie at once
OVERLAP_SIZE = 5
SOLVER = "auto"  # One of: "auto", "GUROBI", or "CBC"
WORKERS = max(1, min(8, (os.cpu_count() or 2) // 2))
TAIL_LENGTH = 30

# =============================================================================
# END USER SETTINGS
# =============================================================================


def validate_settings() -> None:
    """Fail early when an editable setting has an invalid value."""
    if SCENE < 0:
        raise ValueError("SCENE must be zero or greater.")
    if TRACK_CHANNEL < 0:
        raise ValueError("TRACK_CHANNEL must be zero or greater.")
    if REGISTRATION_CHANNEL is not None and REGISTRATION_CHANNEL < 0:
        raise ValueError("REGISTRATION_CHANNEL must be None or zero or greater.")
    if REGISTRATION_DOWNSAMPLE < 1:
        raise ValueError("REGISTRATION_DOWNSAMPLE must be at least 1.")
    if REGISTRATION_UPSAMPLE < 1:
        raise ValueError("REGISTRATION_UPSAMPLE must be at least 1.")
    if MAX_REGISTRATION_SHIFT <= 0:
        raise ValueError("MAX_REGISTRATION_SHIFT must be positive.")
    if MIN_AREA < 0 or MAX_AREA < MIN_AREA:
        raise ValueError("Require 0 <= MIN_AREA <= MAX_AREA.")
    if MAX_DISTANCE <= 0:
        raise ValueError("MAX_DISTANCE must be positive.")
    if MAX_NEIGHBORS < 1:
        raise ValueError("MAX_NEIGHBORS must be at least 1.")
    if WINDOW_SIZE < 0:
        raise ValueError("WINDOW_SIZE must be zero or greater.")
    if OVERLAP_SIZE < 0:
        raise ValueError("OVERLAP_SIZE must be zero or greater.")
    if SOLVER not in {"auto", "GUROBI", "CBC"}:
        raise ValueError('SOLVER must be "auto", "GUROBI", or "CBC".')
    if WORKERS < 1:
        raise ValueError("WORKERS must be at least 1.")
    if TAIL_LENGTH < 0:
        raise ValueError("TAIL_LENGTH must be zero or greater.")


def _compute(array: Any) -> np.ndarray:
    """Convert a NumPy/Dask/Zarr slice to a NumPy array."""
    if hasattr(array, "compute"):
        array = array.compute()
    return np.asarray(array)


def _positive_float(value: Any, fallback: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(value) or value <= 0:
        return fallback
    return value


def _prepare_registration_frame(
    frame_yx: np.ndarray,
    *,
    downsample: int,
    sigma: float,
) -> np.ndarray:
    """Normalize, blur, crop, and downsample one 2-D registration frame."""
    frame = np.asarray(frame_yx, dtype=np.float32)
    frame = np.nan_to_num(frame, copy=False)

    low, high = np.percentile(frame, (1.0, 99.8))
    if high <= low:
        high = low + 1.0
    frame = np.clip((frame - low) / (high - low), 0.0, 1.0)

    if sigma > 0:
        frame = ndi.gaussian_filter(frame, sigma=sigma)

    # Ignore a narrow border, which is often less stable and causes wrap ambiguity.
    border_y = int(round(frame.shape[0] * 0.04))
    border_x = int(round(frame.shape[1] * 0.04))
    if border_y > 0 and border_x > 0:
        cropped = frame[border_y:-border_y, border_x:-border_x]
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
    registration_ty_x: da.Array,
    *,
    downsample: int,
    sigma: float,
    upsample_factor: int,
    max_shift_px: float,
) -> np.ndarray:
    """Estimate the absolute XY shift required to stabilize every frame.

    Each raw frame is aligned to the previously stabilized frame. Only translation
    is modeled; rotation, deformation, and biological cell movement are untouched.
    """
    n_time = int(registration_ty_x.shape[0])
    shifts = np.zeros((n_time, 2), dtype=np.float64)

    previous_registered = _prepare_registration_frame(
        _compute(registration_ty_x[0]), downsample=downsample, sigma=sigma
    )

    for t in tqdm(range(1, n_time), desc="Estimating XY stabilization"):
        current = _prepare_registration_frame(
            _compute(registration_ty_x[t]), downsample=downsample, sigma=sigma
        )
        try:
            shift_small, error, _ = phase_cross_correlation(
                previous_registered,
                current,
                upsample_factor=max(1, int(upsample_factor)),
                normalization=None,
            )
            shift_full = np.asarray(shift_small[-2:], dtype=float) * max(
                1, int(downsample)
            )

            if not np.all(np.isfinite(shift_full)):
                raise ValueError("non-finite phase-correlation shift")
            if np.linalg.norm(shift_full) > max_shift_px:
                raise ValueError(
                    f"estimated shift {shift_full} exceeds MAX_REGISTRATION_SHIFT"
                )
            if not np.isfinite(error):
                warnings.warn(f"Frame {t}: phase-correlation error is non-finite")

        except Exception as exc:  # keep the pipeline usable on a low-contrast frame
            warnings.warn(
                f"Frame {t}: registration failed ({exc}); reusing the preceding shift"
            )
            shift_full = shifts[t - 1]

        shifts[t] = shift_full
        previous_registered = ndi.shift(
            current,
            shift=tuple(shift_full / max(1, int(downsample))),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )

    return shifts


def write_stabilized_movie(
    movie: da.Array,
    shifts_yx: np.ndarray,
    output_path: Path,
    *,
    description: str = "Writing stabilized movie",
) -> Any:
    """Apply XY shifts frame-by-frame and write a disk-backed Zarr array."""
    output = create_zarr(
        shape=tuple(int(v) for v in movie.shape),
        dtype=np.float32,
        store_or_path=str(output_path),
        overwrite=True,
    )

    for t in tqdm(range(int(movie.shape[0])), desc=description):
        frame = _compute(movie[t]).astype(np.float32, copy=False)
        dy, dx = shifts_yx[t]
        spatial_shift = (dy, dx) if frame.ndim == 2 else (0.0, dy, dx)
        finite = frame[np.isfinite(frame)]
        cval = float(np.percentile(finite, 1.0)) if finite.size else 0.0
        output[t] = ndi.shift(
            frame,
            shift=spatial_shift,
            order=1,
            mode="constant",
            cval=cval,
            prefilter=False,
        )

    return output


def generate_ultrack_inputs(
    stabilized: Any,
    *,
    output_dir: Path,
    spatial_scale: tuple[float, ...],
    foreground_sigma: float,
    boundary_sigma: float,
    min_foreground: float,
    remove_hist_mode: bool,
) -> tuple[Any, Any]:
    """Create Ultrack foreground and contour maps on disk."""
    foreground = create_zarr(
        shape=stabilized.shape,
        dtype=bool,
        store_or_path=str(output_dir / "foreground.zarr"),
        overwrite=True,
    )
    array_apply(
        stabilized,
        out_array=foreground,
        func=detect_foreground,
        voxel_size=spatial_scale,
        sigma=foreground_sigma,
        remove_hist_mode=remove_hist_mode,
        min_foreground=min_foreground,
    )

    contours = create_zarr(
        shape=stabilized.shape,
        dtype=np.float16,
        store_or_path=str(output_dir / "contours.zarr"),
        overwrite=True,
    )
    array_apply(
        stabilized,
        out_array=contours,
        func=robust_invert,
        voxel_size=spatial_scale,
        sigma=boundary_sigma,
    )
    return foreground, contours


def make_ultrack_config(
    *,
    working_dir: Path,
    n_time: int,
    workers: int,
    min_area: int,
    max_area: int,
    max_distance: float,
    max_neighbors: int,
    solver: str,
    window_size: int,
    overlap_size: int,
) -> MainConfig:
    config = MainConfig()
    config.data_config.working_dir = working_dir
    config.data_config.n_workers = workers

    config.segmentation_config.n_workers = workers
    config.segmentation_config.min_area = min_area
    config.segmentation_config.max_area = max_area
    config.segmentation_config.min_frontier = 0.0

    config.linking_config.n_workers = workers
    config.linking_config.max_distance = max_distance
    config.linking_config.max_neighbors = max_neighbors

    config.tracking_config.solver_name = "" if solver == "auto" else solver
    config.tracking_config.n_threads = workers
    # Neutrophils normally do not divide during a short movie.
    config.tracking_config.division_weight = -0.1

    if window_size > 0 and n_time > window_size:
        config.tracking_config.window_size = window_size
        config.tracking_config.overlap_size = min(
            overlap_size, max(1, window_size // 2)
        )

    return config


def open_results_in_napari(
    *,
    stabilized: Any,
    stabilized_context: Any | None,
    foreground: Any,
    contours: Any,
    tracked_segments: Any,
    tracks_df: pd.DataFrame,
    graph: dict[int, list[int]],
    napari_scale: tuple[float, ...],
    is_3d: bool,
    tail_length: int,
    title: str,
) -> None:
    viewer = napari.Viewer(title=title)
    axis_labels = ("t", "z", "y", "x") if is_3d else ("t", "y", "x")

    if stabilized_context is not None:
        viewer.add_image(
            stabilized_context,
            name="stabilized fish context",
            colormap="gray",
            scale=napari_scale,
            axis_labels=axis_labels,
        )
        viewer.add_image(
            stabilized,
            name="stabilized neutrophil channel",
            colormap="magenta",
            blending="additive",
            scale=napari_scale,
            axis_labels=axis_labels,
        )
    else:
        viewer.add_image(
            stabilized,
            name="stabilized fish / neutrophils",
            colormap="gray",
            scale=napari_scale,
            axis_labels=axis_labels,
        )
    viewer.add_labels(
        tracked_segments,
        name="Ultrack tracked segments",
        opacity=0.28,
        scale=napari_scale,
        axis_labels=axis_labels,
    )
    viewer.add_labels(
        foreground,
        name="foreground used by Ultrack",
        opacity=0.25,
        visible=False,
        scale=napari_scale,
        axis_labels=axis_labels,
    )
    viewer.add_image(
        contours,
        name="contours used by Ultrack",
        opacity=0.55,
        visible=False,
        blending="additive",
        scale=napari_scale,
        axis_labels=axis_labels,
    )

    if not tracks_df.empty:
        spatial_columns = ["z", "y", "x"] if is_3d else ["y", "x"]
        tracks_data = tracks_df[["track_id", "t", *spatial_columns]].to_numpy()
        viewer.add_tracks(
            tracks_data,
            graph=graph,
            name="Ultrack trajectories",
            tail_length=tail_length,
            tail_width=3,
            scale=napari_scale,
            axis_labels=axis_labels,
        )
    else:
        warnings.warn("Ultrack returned no tracks; inspect the foreground layer.")

    if is_3d:
        viewer.dims.ndisplay = 3
    napari.run()


def main() -> None:
    validate_settings()

    czi_path = Path(CZI_PATH).expanduser().resolve()
    if not czi_path.is_file():
        raise FileNotFoundError(czi_path)
    if czi_path.suffix.lower() != ".czi":
        warnings.warn(f"Expected a .czi file, received: {czi_path.name}")

    output_dir = (
        Path(OUTPUT_DIR).expanduser().resolve()
        if OUTPUT_DIR is not None
        else czi_path.with_name(f"{czi_path.stem}_ultrack")
    )
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
    for name, channel in (
        ("track", TRACK_CHANNEL),
        ("registration", registration_channel),
    ):
        if not 0 <= channel < n_channels:
            raise IndexError(
                f"{name} channel {channel} is outside 0..{n_channels - 1}"
            )

    track_tzyx = image.get_image_dask_data("TZYX", C=TRACK_CHANNEL)
    registration_tzyx = image.get_image_dask_data(
        "TZYX", C=registration_channel
    )
    if int(track_tzyx.shape[0]) < 2:
        raise ValueError("Tracking requires at least two time points.")

    z_count = int(track_tzyx.shape[1])
    is_3d = (not PROJECT_2D) and z_count > 1
    if is_3d:
        movie = track_tzyx
    elif z_count == 1:
        movie = track_tzyx[:, 0]
    else:
        movie = track_tzyx.max(axis=1)

    # Always estimate the simple XY camera shift from a Z maximum projection.
    if int(registration_tzyx.shape[1]) == 1:
        registration_projection = registration_tzyx[:, 0]
    else:
        registration_projection = registration_tzyx.max(axis=1)

    if NO_STABILIZATION:
        shifts_yx = np.zeros((int(movie.shape[0]), 2), dtype=float)
    else:
        shifts_yx = estimate_xy_shifts(
            registration_projection,
            downsample=REGISTRATION_DOWNSAMPLE,
            sigma=REGISTRATION_SIGMA,
            upsample_factor=REGISTRATION_UPSAMPLE,
            max_shift_px=MAX_REGISTRATION_SHIFT,
        )

    pd.DataFrame(
        {
            "t": np.arange(len(shifts_yx)),
            "shift_y_pixels": shifts_yx[:, 0],
            "shift_x_pixels": shifts_yx[:, 1],
        }
    ).to_csv(output_dir / "stabilization_shifts.csv", index=False)

    stabilized = write_stabilized_movie(
        movie,
        shifts_yx,
        output_dir / "stabilized.zarr",
        description="Writing stabilized neutrophil channel",
    )

    # When a separate structural/brightfield channel was used for registration,
    # stabilize it with the identical shifts so the fish is visible under the tracks.
    stabilized_context = None
    if registration_channel != TRACK_CHANNEL:
        if is_3d:
            context_movie = registration_tzyx
        elif int(registration_tzyx.shape[1]) == 1:
            context_movie = registration_tzyx[:, 0]
        else:
            context_movie = registration_tzyx.max(axis=1)
        stabilized_context = write_stabilized_movie(
            context_movie,
            shifts_yx,
            output_dir / "stabilized_context.zarr",
            description="Writing stabilized fish context",
        )

    pixel_sizes = image.physical_pixel_sizes
    pixel_x = _positive_float(pixel_sizes.X, 1.0)
    pixel_y = _positive_float(pixel_sizes.Y, pixel_x)
    pixel_z = _positive_float(pixel_sizes.Z, pixel_x)

    # Ultrack's link distance is measured after this scaling. Normalizing by X
    # keeps MAX_DISTANCE interpretable as approximately X pixels while still
    # accounting for anisotropic Z/Y sampling.
    if is_3d:
        spatial_scale = (pixel_z / pixel_x, pixel_y / pixel_x, 1.0)
        napari_scale = (1.0, pixel_z, pixel_y, pixel_x)
    else:
        spatial_scale = (pixel_y / pixel_x, 1.0)
        napari_scale = (1.0, pixel_y, pixel_x)

    foreground, contours = generate_ultrack_inputs(
        stabilized,
        output_dir=output_dir,
        spatial_scale=spatial_scale,
        foreground_sigma=FOREGROUND_SIGMA,
        boundary_sigma=BOUNDARY_SIGMA,
        min_foreground=MIN_FOREGROUND,
        remove_hist_mode=not KEEP_HISTOGRAM_MODE,
    )

    config = make_ultrack_config(
        working_dir=work_dir,
        n_time=int(stabilized.shape[0]),
        workers=WORKERS,
        min_area=MIN_AREA,
        max_area=MAX_AREA,
        max_distance=MAX_DISTANCE,
        max_neighbors=MAX_NEIGHBORS,
        solver=SOLVER,
        window_size=WINDOW_SIZE,
        overlap_size=OVERLAP_SIZE,
    )
    tracker = Tracker(config)
    tracker.track(
        foreground=foreground,
        contours=contours,
        scale=spatial_scale,
        overwrite="all",
    )

    tracks_df, graph = tracker.to_tracks_layer()
    tracker.export_by_extension(str(output_dir / "tracks.csv"), overwrite=True)
    tracker.export_by_extension(str(output_dir / "tracks.xml"), overwrite=True)
    tracker.export_by_extension(
        str(output_dir / "tracks_graph.json"), overwrite=True
    )
    tracked_segments = tracker.to_zarr(
        store_or_path=str(output_dir / "tracked_segments.zarr"),
        overwrite=True,
    )

    parameters = {
        "czi": str(czi_path),
        "output": str(output_dir),
        "scene": SCENE,
        "track_channel": TRACK_CHANNEL,
        "registration_channel": registration_channel,
        "project_2d": PROJECT_2D,
        "no_stabilization": NO_STABILIZATION,
        "registration_downsample": REGISTRATION_DOWNSAMPLE,
        "registration_sigma": REGISTRATION_SIGMA,
        "registration_upsample": REGISTRATION_UPSAMPLE,
        "max_registration_shift": MAX_REGISTRATION_SHIFT,
        "foreground_sigma": FOREGROUND_SIGMA,
        "boundary_sigma": BOUNDARY_SIGMA,
        "min_foreground": MIN_FOREGROUND,
        "keep_histogram_mode": KEEP_HISTOGRAM_MODE,
        "min_area": MIN_AREA,
        "max_area": MAX_AREA,
        "max_distance": MAX_DISTANCE,
        "max_neighbors": MAX_NEIGHBORS,
        "window_size": WINDOW_SIZE,
        "overlap_size": OVERLAP_SIZE,
        "solver": SOLVER,
        "workers": WORKERS,
        "tail_length": TAIL_LENGTH,
        "is_3d": is_3d,
        "spatial_scale": spatial_scale,
        "physical_pixel_sizes_zyx": (pixel_z, pixel_y, pixel_x),
    }
    with open(output_dir / "run_parameters.json", "w", encoding="utf-8") as stream:
        json.dump(parameters, stream, indent=2)

    n_tracks = int(tracks_df["track_id"].nunique()) if not tracks_df.empty else 0
    print(f"Exported {n_tracks} tracks and {len(tracks_df)} detections to:")
    print(output_dir)

    open_results_in_napari(
        stabilized=stabilized,
        stabilized_context=stabilized_context,
        foreground=foreground,
        contours=contours,
        tracked_segments=tracked_segments,
        tracks_df=tracks_df,
        graph=graph,
        napari_scale=napari_scale,
        is_3d=is_3d,
        tail_length=TAIL_LENGTH,
        title=f"{czi_path.name} — Ultrack neutrophils",
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(
            "This script does not accept command-line parameters. "
            "Edit the USER SETTINGS block at the top of the file instead."
        )
    main()
