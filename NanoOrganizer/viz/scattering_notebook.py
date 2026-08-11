"""Notebook-friendly scattering product loaders and plots.

This module is intentionally independent of Streamlit.  It provides a small,
stable reference implementation for inspecting CMS/SMI reduction products in
JupyterLab or ordinary Python scripts:

* q-image NPZ files are plotted with rows = qz and columns = qx;
* reduction masks and non-positive remesh pixels are hidden;
* large images are stride-reduced before plotting;
* logarithmic intensity uses robust percentile limits rather than the maximum;
* q-phi, circular-average, QC, and frame-index helpers share the same product
  naming conventions as the web explorer.

The plotting functions return Matplotlib objects, so callers can continue to
customize axes, colorbars, and layouts in a notebook.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]


@dataclass(frozen=True)
class QImageData:
    """A q-image and its coordinate axes.

    ``mask`` is normalized to Matplotlib/NumPy's convention: ``True`` means
    that a pixel should be hidden.  The SMI reduction files used by the
    notebooks store ``qimg_mask`` as a *valid-data* mask, so ``load_qimage``
    inverts that array once while loading it.
    """

    intensity: np.ndarray
    qx: np.ndarray
    qz: np.ndarray
    mask: Optional[np.ndarray] = None
    path: Optional[Path] = None


@dataclass(frozen=True)
class QPhiData:
    """A q-phi map and its coordinate axes."""

    intensity: np.ndarray
    q: np.ndarray
    phi: np.ndarray
    mask: Optional[np.ndarray] = None
    path: Optional[Path] = None


PRODUCT_PATTERNS: Mapping[str, Tuple[str, ...]] = {
    "stitched": ("*.tif", "*.tiff"),
    "qc": ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"),
    "q_image": ("*.npz",),
    "qphi": ("*.npz",),
    "cir_avg": ("*.csv",),
}


def _path(value: PathLike) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def load_qimage(path: PathLike) -> QImageData:
    """Load a q-image NPZ and validate its geometry.

    The q-image reduction convention is ``qimg_mask == True`` for pixels
    carrying remeshed data.  Internally this module uses the opposite,
    invalid-pixel convention so that ``valid_intensity`` and Matplotlib
    masked arrays behave as expected.
    """
    fpath = _path(path)
    with np.load(fpath) as data:
        required = {"qimg", "qx", "qz"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"{fpath} is missing q-image keys: {missing}")
        intensity = np.asarray(data["qimg"], dtype=float)
        qx = np.asarray(data["qx"], dtype=float)
        qz = np.asarray(data["qz"], dtype=float)
        raw_mask = np.asarray(data["qimg_mask"], dtype=bool) if "qimg_mask" in data else None

    if intensity.ndim != 2:
        raise ValueError(f"qimg must be 2-D, got shape {intensity.shape}")
    if intensity.shape != (len(qz), len(qx)):
        raise ValueError(
            "qimg geometry does not match axes: "
            f"qimg={intensity.shape}, qz={len(qz)}, qx={len(qx)}"
        )
    if raw_mask is not None and raw_mask.shape != intensity.shape:
        # A q-phi mask in some reduction versions belongs to the detector grid;
        # silently ignoring it here would hide a q-image bug, so make the
        # mismatch visible to the notebook user.
        raise ValueError(
            f"qimg_mask shape {raw_mask.shape} does not match qimg {intensity.shape}"
        )
    mask = None if raw_mask is None else ~raw_mask
    return QImageData(intensity, qx, qz, mask, fpath)


def load_qphi(path: PathLike) -> QPhiData:
    """Load a q-phi NPZ, ignoring detector-grid masks with another shape.

    When a q-phi mask has the q-phi map's shape, it is normalized from the
    reduction's valid-data convention to the internal invalid-pixel
    convention.  Current CMS/SMI files often carry a detector-grid mask with
    a different shape; that mask is deliberately ignored.
    """
    fpath = _path(path)
    with np.load(fpath) as data:
        required = {"q", "phi", "qphi"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"{fpath} is missing q-phi keys: {missing}")
        intensity = np.asarray(data["qphi"], dtype=float)
        q = np.asarray(data["q"], dtype=float)
        phi = np.asarray(data["phi"], dtype=float)
        raw_mask = np.asarray(data["qphi_mask"], dtype=bool) if "qphi_mask" in data else None
    if intensity.shape != (len(phi), len(q)):
        raise ValueError(
            "qphi geometry does not match axes: "
            f"qphi={intensity.shape}, phi={len(phi)}, q={len(q)}"
        )
    if raw_mask is not None and raw_mask.shape != intensity.shape:
        # Current CMS files commonly carry a raw-detector mask here. It is not
        # valid for qphi and is therefore not applied.
        raw_mask = None
    mask = None if raw_mask is None else ~raw_mask
    return QPhiData(intensity, q, phi, mask, fpath)


def valid_intensity(values: np.ndarray, mask: Optional[np.ndarray] = None):
    """Return a masked array hiding invalid, masked, and non-positive pixels.

    ``mask`` follows the normalized NumPy convention: ``True`` means
    invalid/hidden.  Loaders convert the reduction files' valid-data masks to
    this convention before exposing them on their data objects.
    """
    existing_mask = np.ma.getmaskarray(values) if np.ma.isMaskedArray(values) else None
    array = np.asarray(np.ma.getdata(values), dtype=float).copy()
    invalid = ~np.isfinite(array) | (array <= 0)
    if existing_mask is not None:
        invalid |= existing_mask
    if mask is not None and mask.shape == array.shape:
        invalid |= np.asarray(mask, dtype=bool)
    return np.ma.array(array, mask=invalid)


def qimage_stats(data: QImageData) -> Dict[str, float]:
    """Summarize q-image geometry and valid-pixel coverage."""
    masked = valid_intensity(data.intensity, data.mask)
    values = masked.compressed()
    if values.size:
        p01, p50, p999 = np.percentile(values, [1, 50, 99.9])
    else:
        p01 = p50 = p999 = float("nan")
    return {
        "rows": float(data.intensity.shape[0]),
        "columns": float(data.intensity.shape[1]),
        "qx_min": float(np.nanmin(data.qx)),
        "qx_max": float(np.nanmax(data.qx)),
        "qz_min": float(np.nanmin(data.qz)),
        "qz_max": float(np.nanmax(data.qz)),
        "valid_pixels": float(values.size),
        "valid_fraction": float(values.size / data.intensity.size),
        "p01": float(p01),
        "median": float(p50),
        "p999": float(p999),
    }


def crop_qimage(data: QImageData, qxlim=None, qzlim=None) -> QImageData:
    """Crop a q-image in coordinate space, preserving the qx/qz axes."""
    xmask = np.ones(len(data.qx), dtype=bool)
    zmask = np.ones(len(data.qz), dtype=bool)
    if qxlim is not None:
        xmask = (data.qx >= qxlim[0]) & (data.qx <= qxlim[1])
    if qzlim is not None:
        zmask = (data.qz >= qzlim[0]) & (data.qz <= qzlim[1])
    if not xmask.any() or not zmask.any():
        raise ValueError("The requested qx/qz crop contains no pixels")
    mask = data.mask[np.ix_(zmask, xmask)] if data.mask is not None else None
    return replace(
        data,
        intensity=data.intensity[np.ix_(zmask, xmask)],
        qx=data.qx[xmask],
        qz=data.qz[zmask],
        mask=mask,
    )


def downsample_qimage(data: QImageData, max_pixels: int = 600_000) -> QImageData:
    """Stride-reduce a q-image while retaining coordinate arrays."""
    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive")
    nrows, ncols = data.intensity.shape
    step = max(1, int(np.ceil(np.sqrt(nrows * ncols / max_pixels))))
    if step == 1:
        return data
    mask = data.mask[::step, ::step] if data.mask is not None else None
    return replace(
        data,
        intensity=data.intensity[::step, ::step],
        qx=data.qx[::step],
        qz=data.qz[::step],
        mask=mask,
    )


def robust_limits(values: np.ndarray, quantiles=(1.0, 99.5), vmin=None, vmax=None):
    """Get positive intensity limits, suitable for ``LogNorm``."""
    positive = valid_intensity(values).compressed()
    if positive.size == 0:
        raise ValueError("No positive finite intensity values are available")
    low, high = np.percentile(positive, quantiles)
    low = float(low if vmin is None else vmin)
    high = float(high if vmax is None else vmax)
    if low <= 0 or high <= 0 or high <= low:
        raise ValueError(f"Invalid display limits: vmin={low}, vmax={high}")
    return low, high


def plot_qimage(data: QImageData, *, ax=None, qxlim=None, qzlim=None,
                log_intensity=True, quantiles=(1.0, 99.5), vmin=None,
                vmax=None, cmap="magma", max_pixels=600_000,
                aspect="auto", colorbar=True, title=None):
    """Plot q-image with physical qx/qz axes and return ``(fig, ax, image)``."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    selected = crop_qimage(data, qxlim, qzlim) if (qxlim is not None or qzlim is not None) else data
    selected = downsample_qimage(selected, max_pixels=max_pixels)
    masked = valid_intensity(selected.intensity, selected.mask)
    if log_intensity:
        lo, hi = robust_limits(masked, quantiles, vmin, vmax)
        norm = LogNorm(vmin=lo, vmax=hi, clip=True)
        cbar_label = "Intensity (log scale)"
    else:
        lo, hi = robust_limits(masked, quantiles, vmin, vmax)
        norm = Normalize(vmin=lo, vmax=hi, clip=True)
        cbar_label = "Intensity"

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    else:
        fig = ax.figure
    image = ax.imshow(
        masked,
        origin="lower",
        extent=(selected.qx[0], selected.qx[-1], selected.qz[0], selected.qz[-1]),
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    ax.set_xlabel(r"$q_x$ ($\AA^{-1}$)")
    ax.set_ylabel(r"$q_z$ ($\AA^{-1}$)")
    ax.set_title(title or "q-image")
    if colorbar:
        cbar = fig.colorbar(image, ax=ax, pad=0.02)
        cbar.set_label(cbar_label)
    return fig, ax, image


def qimage_linecuts(data: QImageData, *, qx=None, qz=None):
    """Extract nearest-row/nearest-column profiles from a q-image.

    Returns a dictionary with ``qx_profile`` (at the requested qz),
    ``qz_profile`` (at the requested qx), and the actual selected coordinates.
    Masked pixels are returned as NaN, which makes the result directly
    plottable with Matplotlib or Pandas.
    """
    masked = valid_intensity(data.intensity, data.mask)
    if qz is None:
        z_index = len(data.qz) // 2
    else:
        z_index = int(np.argmin(np.abs(data.qz - float(qz))))
    if qx is None:
        x_index = len(data.qx) // 2
    else:
        x_index = int(np.argmin(np.abs(data.qx - float(qx))))
    return {
        "qx": data.qx.copy(),
        "qx_profile": masked[z_index, :].filled(np.nan),
        "qz": data.qz.copy(),
        "qz_profile": masked[:, x_index].filled(np.nan),
        "selected_qz": float(data.qz[z_index]),
        "selected_qx": float(data.qx[x_index]),
    }


def plot_qphi(data: QPhiData, *, ax=None, qlim=None, philim=None,
              log_intensity=True, quantiles=(1.0, 99.5), vmin=None,
              vmax=None, cmap="magma", colorbar=True, title="q–φ map"):
    """Plot a q-phi map with q horizontal and phi vertical."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    qmask = np.ones(len(data.q), dtype=bool) if qlim is None else (
        (data.q >= qlim[0]) & (data.q <= qlim[1]))
    phmask = np.ones(len(data.phi), dtype=bool) if philim is None else (
        (data.phi >= philim[0]) & (data.phi <= philim[1]))
    if not qmask.any() or not phmask.any():
        raise ValueError("The requested q/phi crop contains no pixels")
    z = data.intensity[np.ix_(phmask, qmask)]
    mask = data.mask[np.ix_(phmask, qmask)] if data.mask is not None else None
    masked = valid_intensity(z, mask)
    lo, hi = robust_limits(masked, quantiles, vmin, vmax)
    norm = LogNorm(vmin=lo, vmax=hi, clip=True) if log_intensity else Normalize(vmin=lo, vmax=hi, clip=True)
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    else:
        fig = ax.figure
    image = ax.imshow(
        masked, origin="lower",
        extent=(data.q[qmask][0], data.q[qmask][-1],
                data.phi[phmask][0], data.phi[phmask][-1]),
        aspect="auto", cmap=cmap, norm=norm, interpolation="nearest",
    )
    ax.set_xlabel(r"$q$ ($\AA^{-1}$)")
    ax.set_ylabel(r"$\phi$ (deg)")
    ax.set_title(title)
    if colorbar:
        cbar = fig.colorbar(image, ax=ax, pad=0.02)
        cbar.set_label("Intensity (log scale)" if log_intensity else "Intensity")
    return fig, ax, image


def load_cir_avg(path: PathLike):
    """Load a circular-average CSV using the reduction column conventions."""
    frame = pd.read_csv(_path(path))
    columns = {str(c).lower(): c for c in frame.columns}
    qcol = columns.get("q_ca") or columns.get("q") or frame.columns[-2]
    icol = columns.get("iq_ca") or columns.get("intensity") or columns.get("i") or frame.columns[-1]
    return frame[qcol].to_numpy(float), frame[icol].to_numpy(float)


def plot_cir_avg(path: PathLike, *, ax=None, qlim=None, logx=True,
                 logy=True, title="Circular average I(q)"):
    """Plot a circular-average CSV and return ``(fig, ax, line)``."""
    import matplotlib.pyplot as plt

    q, intensity = load_cir_avg(path)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    else:
        fig = ax.figure
    line, = ax.plot(q, intensity, lw=1.2)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if qlim:
        ax.set_xlim(*qlim)
    ax.set_xlabel(r"$q$ ($\AA^{-1}$)")
    ax.set_ylabel("I(q)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.2)
    return fig, ax, line


def load_qc_image(path: PathLike):
    """Load a QC image using Pillow (an optional notebook dependency)."""
    from PIL import Image
    with Image.open(_path(path)) as image:
        return np.asarray(image)


def plot_qc_image(path: PathLike, *, ax=None, title="QC image"):
    """Plot a QC image and return ``(fig, ax, image)``."""
    import matplotlib.pyplot as plt

    image = load_qc_image(path)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    else:
        fig = ax.figure
    shown = ax.imshow(image, origin="upper", cmap="gray" if image.ndim == 2 else None)
    ax.set_title(title)
    ax.set_axis_off()
    return fig, ax, shown


def _frame_stem(filename: str) -> str:
    stem = Path(filename).name
    for prefix in ("Cir_Avg_", "qphi_", "qimg_", "qc_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    while True:
        stripped = re.sub(r"\.(npz|csv|png|tiff|tif|jpg|jpeg)$", "", stem,
                          flags=re.IGNORECASE)
        if stripped == stem:
            return stem
        stem = stripped


def product_index(root: PathLike) -> pd.DataFrame:
    """Index product files by their shared reduction-frame stem."""
    base = _path(root)
    maps: Dict[str, Dict[str, str]] = {}
    for product, patterns in PRODUCT_PATTERNS.items():
        folder = base / product
        files = {}
        if folder.is_dir():
            for pattern in patterns:
                for file in folder.glob(pattern):
                    if file.is_file():
                        files[_frame_stem(file.name)] = str(file)
        maps[product] = files
    stems = sorted(set().union(*(set(values) for values in maps.values())))
    rows = []
    for stem in stems:
        row = {"stem": stem}
        for product, values in maps.items():
            row[product] = values.get(stem)
            row[f"has_{product}"] = stem in values
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "QImageData", "QPhiData", "PRODUCT_PATTERNS", "load_qimage",
    "load_qphi", "valid_intensity", "qimage_stats", "crop_qimage",
    "downsample_qimage", "robust_limits", "plot_qimage", "plot_qphi",
    "qimage_linecuts",
    "load_cir_avg", "plot_cir_avg", "load_qc_image", "plot_qc_image",
    "product_index",
]
