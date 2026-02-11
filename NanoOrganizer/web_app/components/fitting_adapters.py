#!/usr/bin/env python3
"""
Fitting backend adapters for NanoOrganizer CSV Plotter.

This module keeps backend-specific fitting logic isolated from Streamlit UI code.
"""

from pathlib import Path
import sys
import os
import io
import json
import re
import zipfile
import signal
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# pyFitting import (required by all adapters)
# ---------------------------------------------------------------------------
PYFITTING_AVAILABLE = False
PYFITTING_IMPORT_ERROR = ""
ArrayData = None
Fitter = None
MultiPeakModel = None
try:
    from pyFitting import ArrayData, Fitter  # type: ignore
    from pyFitting.models import MultiPeakModel  # type: ignore
    PYFITTING_AVAILABLE = True
except Exception as e:
    _repo_parent = Path(__file__).resolve().parents[4]
    _pyfitting_repo = _repo_parent / "pyFitting"
    if _pyfitting_repo.exists():
        sys.path.insert(0, str(_pyfitting_repo))
        try:
            from pyFitting import ArrayData, Fitter  # type: ignore
            from pyFitting.models import MultiPeakModel  # type: ignore
            PYFITTING_AVAILABLE = True
        except Exception as e2:
            PYFITTING_IMPORT_ERROR = str(e2)
    else:
        PYFITTING_IMPORT_ERROR = str(e)

# ---------------------------------------------------------------------------
# pySAXSFitting import (SAXS backend)
# ---------------------------------------------------------------------------
PYSAXS_AVAILABLE = False
PYSAXS_IMPORT_ERROR = ""
PYSAXS_INTERFACE_AVAILABLE = False
PYSAXS_INTERFACE_IMPORT_ERROR = ""
SphereModel = None
CubeModel = None
TetrahedronModel = None
OctahedronModel = None
RhombicDodecahedronModel = None
SquarePyramidModel = None
PolydisperseSphereModel = None
PolydisperseCubeModel = None
PolydisperseTetrahedronModel = None
PolydisperseOctahedronModel = None
PolydisperseRhombicDodecahedronModel = None
PolydisperseSquarePyramidModel = None
_pysaxs_get_model_library_rows = None
_pysaxs_suggest_saxs_seed = None
_pysaxs_predict_ml_seed = None
_pysaxs_fit_multipeak_curve = None
_pysaxs_fit_formfactor_curve = None
try:
    from pySAXSFitting.formfactors import (  # type: ignore
        SphereModel,
        CubeModel,
        TetrahedronModel,
        OctahedronModel,
        RhombicDodecahedronModel,
        SquarePyramidModel,
        PolydisperseSphereModel,
        PolydisperseCubeModel,
        PolydisperseTetrahedronModel,
        PolydisperseOctahedronModel,
        PolydisperseRhombicDodecahedronModel,
        PolydisperseSquarePyramidModel,
    )
    import pySAXSFitting.fitting as _pysaxs_fitting_api  # type: ignore
    _pysaxs_get_model_library_rows = getattr(_pysaxs_fitting_api, "get_model_library_rows", None)
    _pysaxs_suggest_saxs_seed = getattr(_pysaxs_fitting_api, "suggest_saxs_seed", None)
    _pysaxs_predict_ml_seed = getattr(_pysaxs_fitting_api, "predict_ml_seed", None)
    _pysaxs_fit_multipeak_curve = getattr(_pysaxs_fitting_api, "fit_multipeak_curve", None)
    _pysaxs_fit_formfactor_curve = getattr(_pysaxs_fitting_api, "fit_formfactor_curve", None)
    PYSAXS_AVAILABLE = True
    PYSAXS_INTERFACE_AVAILABLE = True
except Exception as e:
    _repo_parent = Path(__file__).resolve().parents[4]
    _pysaxs_repo = _repo_parent / "pySAXSFitting"
    if _pysaxs_repo.exists():
        sys.path.insert(0, str(_pysaxs_repo))
        try:
            from pySAXSFitting.formfactors import (  # type: ignore
                SphereModel,
                CubeModel,
                TetrahedronModel,
                OctahedronModel,
                RhombicDodecahedronModel,
                SquarePyramidModel,
                PolydisperseSphereModel,
                PolydisperseCubeModel,
                PolydisperseTetrahedronModel,
                PolydisperseOctahedronModel,
                PolydisperseRhombicDodecahedronModel,
                PolydisperseSquarePyramidModel,
            )
            import pySAXSFitting.fitting as _pysaxs_fitting_api  # type: ignore
            _pysaxs_get_model_library_rows = getattr(_pysaxs_fitting_api, "get_model_library_rows", None)
            _pysaxs_suggest_saxs_seed = getattr(_pysaxs_fitting_api, "suggest_saxs_seed", None)
            _pysaxs_predict_ml_seed = getattr(_pysaxs_fitting_api, "predict_ml_seed", None)
            _pysaxs_fit_multipeak_curve = getattr(_pysaxs_fitting_api, "fit_multipeak_curve", None)
            _pysaxs_fit_formfactor_curve = getattr(_pysaxs_fitting_api, "fit_formfactor_curve", None)
            PYSAXS_AVAILABLE = True
            PYSAXS_INTERFACE_AVAILABLE = True
        except Exception as e2:
            PYSAXS_IMPORT_ERROR = str(e2)
            PYSAXS_INTERFACE_IMPORT_ERROR = str(e2)
    else:
        PYSAXS_IMPORT_ERROR = str(e)
        PYSAXS_INTERFACE_IMPORT_ERROR = str(e)

SAXS_MODEL_LABEL_TO_KEY = {
    "Sphere": "sphere",
    "Cube": "cube",
    "Octahedron": "octahedron",
}
SAXS_MODEL_KEY_TO_LABEL = {v: k for k, v in SAXS_MODEL_LABEL_TO_KEY.items()}

SAXS_MODEL_LIBRARY = {
    "sphere": {
        "description": "Isotropic spherical nanoparticles (smooth form-factor oscillations).",
        "size_parameter": "radius",
        "size_units": "A",
        "recommended_range": "5 to 500",
        "notes": "Best first choice for nearly isotropic SAXS form factor.",
    },
    "cube": {
        "description": "Cubic nanoparticles with sharper oscillatory features.",
        "size_parameter": "radius (half-edge)",
        "size_units": "A",
        "recommended_range": "5 to 500",
        "notes": "Can require more iterations due to orientational averaging.",
    },
    "octahedron": {
        "description": "Octahedral nanoparticles with characteristic minima locations.",
        "size_parameter": "radius (circumradius)",
        "size_units": "A",
        "recommended_range": "5 to 500",
        "notes": "Useful when dip positions disagree with sphere/cube.",
    },
}

ML_MODEL_CATALOG = [
    {
        "key": "heuristic_peaks_v1",
        "label": "Heuristic Peaks v1",
        "target_backend": "general_peaks",
        "status": "ready",
        "description": "Detects candidate peaks from preprocessed 1D curves and seeds multi-peak fitting.",
    },
    {
        "key": "heuristic_saxs_seed_v1",
        "label": "Heuristic SAXS Seed v1",
        "target_backend": "saxs_physics",
        "status": "ready",
        "description": "Classifies SAXS-like curves and estimates initial SAXS model/size seeds.",
    },
    {
        "key": "dip_shape_seed_v1",
        "label": "Dip Shape Seed v1",
        "target_backend": "saxs_physics",
        "status": "ready" if PYSAXS_INTERFACE_AVAILABLE else "fallback",
        "description": "Uses pySAXSFitting dip analysis to seed SAXS model shape and initial parameters.",
    },
]

FIT_SHAPE_LABEL_TO_KEY = {
    "Gaussian": "gaussian",
    "Lorentzian": "lorentzian",
    "Pseudo-Voigt": "pseudo_voigt",
}
FIT_SHAPE_KEY_TO_LABEL = {v: k for k, v in FIT_SHAPE_LABEL_TO_KEY.items()}


def get_saxs_shape_options():
    """Return (label->key, key->label) for SAXS shape selectors."""
    if PYSAXS_INTERFACE_AVAILABLE and _pysaxs_get_model_library_rows is not None:
        try:
            label_to_key = {}
            for item in _pysaxs_get_model_library_rows():
                shape_key = str(item.get("shape", "")).strip().lower()
                label = str(item.get("label", "")).strip()
                if shape_key and label:
                    label_to_key[label] = shape_key
            if label_to_key:
                key_to_label = {v: k for k, v in label_to_key.items()}
                return label_to_key, key_to_label
        except Exception:
            pass

    return dict(SAXS_MODEL_LABEL_TO_KEY), dict(SAXS_MODEL_KEY_TO_LABEL)


def get_saxs_model_library_rows():
    """Return model-library table rows for SAXS models."""
    if PYSAXS_INTERFACE_AVAILABLE and _pysaxs_get_model_library_rows is not None:
        try:
            rows = []
            for item in _pysaxs_get_model_library_rows():
                shape_key = str(item.get("shape", "")).strip().lower()
                label = str(item.get("label", shape_key.title()))
                rows.append({
                    "model": label,
                    "shape_key": shape_key,
                    "description": item.get("description", ""),
                    "size_parameter": item.get("size_parameter", "radius"),
                    "size_units": item.get("size_units", "A"),
                    "recommended_range": item.get("recommended_range", ""),
                    "notes": item.get("notes", ""),
                })
            if rows:
                return rows
        except Exception:
            pass

    rows = []
    for shape_key, info in SAXS_MODEL_LIBRARY.items():
        rows.append({
            "model": SAXS_MODEL_KEY_TO_LABEL.get(shape_key, shape_key.title()),
            "shape_key": shape_key,
            "description": info["description"],
            "size_parameter": info["size_parameter"],
            "size_units": info["size_units"],
            "recommended_range": info["recommended_range"],
            "notes": info["notes"],
        })
    return rows


def get_saxs_model_detail(shape_key):
    """Return detail dict for one SAXS model shape key."""
    shape_key = str(shape_key).lower().strip()
    local = SAXS_MODEL_LIBRARY.get(shape_key)
    if local:
        return local
    if PYSAXS_INTERFACE_AVAILABLE:
        for row in get_saxs_model_library_rows():
            if str(row.get("shape_key", "")).lower() == shape_key:
                return {
                    "description": row.get("description", ""),
                    "size_parameter": row.get("size_parameter", "radius"),
                    "size_units": row.get("size_units", "A"),
                    "recommended_range": row.get("recommended_range", ""),
                    "notes": row.get("notes", ""),
                }
    return {}


def list_ml_models():
    """Return ML model catalog rows."""
    return list(ML_MODEL_CATALOG)


def _moving_average(y, window):
    """Simple moving-average smoother."""
    window = int(max(1, window))
    if window <= 1:
        return np.asarray(y, dtype=float)
    if window % 2 == 0:
        window += 1
    if len(y) < window:
        return np.asarray(y, dtype=float)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(np.asarray(y, dtype=float), kernel, mode="same")


def dedupe_sorted(values, tol):
    """Sort numeric values and remove near-duplicates within tolerance."""
    cleaned = sorted(float(v) for v in values)
    unique = []
    tol = float(max(0.0, tol))
    for value in cleaned:
        if not unique or abs(value - unique[-1]) > tol:
            unique.append(value)
    return unique


def _shape_profile(x, center, width, shape, eta=0.5):
    """Return normalized peak profile for requested shape."""
    width = max(abs(float(width)), 1e-12)
    x = np.asarray(x, dtype=float)
    if shape == "gaussian":
        return np.exp(-0.5 * ((x - center) / width) ** 2)
    if shape == "lorentzian":
        return 1.0 / (1.0 + ((x - center) / width) ** 2)
    eta = float(np.clip(eta, 0.0, 1.0))
    gaussian = np.exp(-0.5 * ((x - center) / width) ** 2)
    lorentzian = 1.0 / (1.0 + ((x - center) / width) ** 2)
    return eta * lorentzian + (1.0 - eta) * gaussian


def run_general_peak_fit(
    x_data,
    y_data,
    *,
    shape,
    shape_label=None,
    n_peaks,
    maxiter,
    peak_guesses=None,
    x_col_name="x",
    y_col_name="y",
):
    """
    Run one multi-peak fit and return a UI-ready fit-state dict.

    Expected shape keys: gaussian, lorentzian, pseudo_voigt.
    """
    shape = str(shape).strip().lower()
    if shape not in FIT_SHAPE_KEY_TO_LABEL:
        raise ValueError(f"Unsupported peak shape: {shape}")

    if PYSAXS_INTERFACE_AVAILABLE and _pysaxs_fit_multipeak_curve is not None:
        try:
            state = _pysaxs_fit_multipeak_curve(
                np.asarray(x_data, dtype=float),
                np.asarray(y_data, dtype=float),
                shape=shape,
                n_peaks=int(n_peaks),
                maxiter=int(maxiter),
                peak_guesses=peak_guesses,
                shape_label=shape_label,
                x_col_name=x_col_name,
                y_col_name=y_col_name,
            )
            return dict(state)
        except Exception:
            pass

    if not PYFITTING_AVAILABLE or MultiPeakModel is None:
        raise RuntimeError(
            "pyFitting unavailable for general peak fitting: "
            f"{PYFITTING_IMPORT_ERROR or 'unknown import error'}"
        )

    n_peaks = int(n_peaks)
    if n_peaks < 1:
        raise ValueError("n_peaks must be >= 1")

    x_fit = np.asarray(x_data, dtype=float)
    y_fit_data = np.asarray(y_data, dtype=float)
    if len(x_fit) < 5 or len(y_fit_data) < 5:
        raise ValueError("Not enough points to fit.")
    if len(x_fit) != len(y_fit_data):
        raise ValueError("x/y length mismatch.")

    model = MultiPeakModel(n_peaks=n_peaks, shape=shape)
    initial_guess = model.get_initial_guess(x_fit, y_fit_data)

    x_span = max(float(np.ptp(x_fit)), 1e-12)
    y_span = max(float(np.ptp(y_fit_data)), 1e-9)
    baseline_guess = float(np.percentile(y_fit_data, 10))
    guess_peaks = dedupe_sorted(peak_guesses or [], max(x_span / 5000.0, 1e-9))

    for idx in range(1, n_peaks + 1):
        if idx <= len(guess_peaks):
            peak_x = float(guess_peaks[idx - 1])
            data_idx = int(np.argmin(np.abs(x_fit - peak_x)))
            initial_guess[f"mu{idx}"] = float(x_fit[data_idx])
            initial_guess[f"A{idx}"] = max(float(y_fit_data[data_idx] - baseline_guess), 0.05 * y_span)
        initial_guess[f"w{idx}"] = max(
            float(initial_guess.get(f"w{idx}", x_span / (12 * n_peaks))),
            x_span / 2000.0,
        )
        if shape == "pseudo_voigt":
            initial_guess[f"eta{idx}"] = float(np.clip(initial_guess.get(f"eta{idx}", 0.5), 0.0, 1.0))
    initial_guess["c"] = float(initial_guess.get("c", baseline_guess))

    lower_c = float(np.min(y_fit_data) - y_span)
    upper_c = float(np.max(y_fit_data) + y_span)
    bounds = {"c": (lower_c, upper_c)}
    for idx in range(1, n_peaks + 1):
        bounds[f"A{idx}"] = (0.0, max(3.0 * y_span, 1.0))
        bounds[f"mu{idx}"] = (float(np.min(x_fit)), float(np.max(x_fit)))
        bounds[f"w{idx}"] = (x_span / 2000.0, x_span)
        if shape == "pseudo_voigt":
            bounds[f"eta{idx}"] = (0.0, 1.0)

    fit_result = Fitter(ArrayData(x_fit, y_fit_data), model).fit(
        initial_guess=initial_guess,
        bounds=bounds,
        maxiter=int(maxiter),
    )

    fit_params = dict(fit_result.parameters.values)
    component_curves = []
    component_rows = []
    for idx in range(1, n_peaks + 1):
        amp = float(fit_params.get(f"A{idx}", 0.0))
        mu = float(fit_params.get(f"mu{idx}", 0.0))
        width = float(fit_params.get(f"w{idx}", 1.0))
        eta = float(fit_params.get(f"eta{idx}", 0.5))
        component = amp * _shape_profile(x_fit, mu, width, shape, eta=eta)
        component_curves.append(np.asarray(component, dtype=float))
        row = {
            "peak": idx,
            "A": amp,
            "mu": mu,
            "w": width,
        }
        if shape == "pseudo_voigt":
            row["eta"] = eta
        component_rows.append(row)

    display_shape = shape_label or FIT_SHAPE_KEY_TO_LABEL.get(shape, shape.title())
    return {
        "backend": "general_peaks",
        "shape": shape,
        "shape_label": display_shape,
        "x_col": x_col_name,
        "y_col": y_col_name,
        "x": np.asarray(x_fit, dtype=float),
        "y": np.asarray(y_fit_data, dtype=float),
        "y_fit": np.asarray(fit_result.y_fit, dtype=float),
        "params": fit_params,
        "metrics": dict(fit_result.metrics),
        "components": component_curves,
        "component_table": component_rows,
        "success": bool(fit_result.success),
        "message": str(fit_result.message),
        "n_peaks": n_peaks,
        "peak_guesses": guess_peaks,
    }


def preprocess_curve_for_ml(
    x_data,
    y_data,
    *,
    x_min=None,
    x_max=None,
    log_y=False,
    normalize_y=True,
    smooth_window=1,
    resample_points=256,
):
    """
    Preprocess a curve for ML inference.

    Returns:
        dict with x_raw, y_raw, x_proc, y_proc, x_resampled, y_resampled, metadata
    """
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if x_min is not None:
        mask &= x >= float(x_min)
    if x_max is not None:
        mask &= x <= float(x_max)
    x = x[mask]
    y = y[mask]
    if len(x) < 5:
        raise ValueError("Not enough valid points after filtering.")
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    y_proc = np.asarray(y, dtype=float)
    if log_y:
        y_shift = np.min(y_proc)
        if y_shift <= 0:
            y_proc = y_proc + abs(y_shift) + 1e-9
        y_proc = np.log10(np.maximum(y_proc, 1e-12))

    y_proc = _moving_average(y_proc, smooth_window)

    if normalize_y:
        y_min = float(np.min(y_proc))
        y_max = float(np.max(y_proc))
        y_span = y_max - y_min
        if y_span > 0:
            y_proc = (y_proc - y_min) / y_span

    n_resample = int(max(32, min(2048, resample_points)))
    x_resampled = np.linspace(float(np.min(x)), float(np.max(x)), n_resample)
    y_resampled = np.interp(x_resampled, x, y_proc)

    return {
        "x_raw": x,
        "y_raw": y,
        "x_proc": x,
        "y_proc": y_proc,
        "x_resampled": x_resampled,
        "y_resampled": y_resampled,
        "metadata": {
            "x_min": float(np.min(x)),
            "x_max": float(np.max(x)),
            "n_points": int(len(x)),
            "log_y": bool(log_y),
            "normalize_y": bool(normalize_y),
            "smooth_window": int(max(1, smooth_window)),
            "resample_points": int(n_resample),
        },
    }


def _find_local_maxima_indices(y):
    """Find local maxima indices in a 1D signal."""
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return np.array([], dtype=int)
    candidates = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    return candidates.astype(int)


def _estimate_radius_from_first_minimum(x, y):
    """Estimate SAXS size radius from first local minimum (sphere approximation)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(y) < 5:
        return None
    minima = np.where((y[1:-1] < y[:-2]) & (y[1:-1] <= y[2:]))[0] + 1
    if len(minima) == 0:
        return None
    q_min = float(x[minima[0]])
    if q_min <= 0:
        return None
    # Sphere first minimum: qR ~= 4.493
    return float(4.493 / q_min)


def predict_ml_seed(preprocessed, model_key):
    """
    Predict fitting seed from preprocessed curve.

    Returns standardized prediction dict.
    """
    model_key = str(model_key)
    x = np.asarray(preprocessed["x_resampled"], dtype=float)
    y = np.asarray(preprocessed["y_resampled"], dtype=float)
    y_raw = np.asarray(preprocessed["y_raw"], dtype=float)

    y_span = float(np.ptp(y)) if len(y) else 0.0
    y_med = float(np.median(y)) if len(y) else 0.0

    if model_key == "heuristic_peaks_v1":
        idx = _find_local_maxima_indices(y)
        if len(idx):
            threshold = y_med + 0.12 * max(y_span, 1e-9)
            idx = np.array([i for i in idx if y[i] >= threshold], dtype=int)
        if len(idx):
            idx = idx[np.argsort(y[idx])[::-1]]
        idx = idx[:6]
        centers = sorted(float(x[i]) for i in idx)
        n_peaks = max(1, len(centers))
        confidence = min(0.95, 0.45 + 0.1 * len(centers))
        if not centers:
            centers = [float(x[int(np.argmax(y))])]
        return {
            "model_key": model_key,
            "recommended_backend": "general_peaks",
            "confidence": float(confidence),
            "peak_count": int(n_peaks),
            "peak_centers": centers,
            "peak_shape_label": "Pseudo-Voigt",
            "notes": "Heuristic peak detector based on local maxima after preprocessing.",
            "saxs_shape": None,
            "saxs_polydisperse": None,
            "saxs_use_porod": None,
            "initial_overrides": {},
        }

    if model_key in {"heuristic_saxs_seed_v1", "dip_shape_seed_v1"}:
        if PYSAXS_INTERFACE_AVAILABLE and _pysaxs_predict_ml_seed is not None:
            try:
                metadata = dict(preprocessed.get("metadata", {}))
                q_range = None
                if "x_min" in metadata and "x_max" in metadata:
                    q_range = (float(metadata["x_min"]), float(metadata["x_max"]))
                pred = _pysaxs_predict_ml_seed(
                    np.asarray(preprocessed["x_raw"], dtype=float),
                    np.asarray(preprocessed["y_raw"], dtype=float),
                    q_range=q_range,
                )
                pred = dict(pred)
                pred["model_key"] = model_key
                return pred
            except Exception:
                pass

        # SAXS-likeness: mostly decreasing raw intensity vs x
        if len(y_raw) > 2:
            decay_ratio = float(np.mean(np.diff(y_raw) <= 0))
        else:
            decay_ratio = 0.5

        local_max = _find_local_maxima_indices(y)
        oscillation_score = int(len(local_max))

        if oscillation_score >= 6:
            shape = "cube"
        elif oscillation_score >= 4:
            shape = "octahedron"
        else:
            shape = "sphere"

        radius_seed = _estimate_radius_from_first_minimum(x, y)
        if radius_seed is None or not np.isfinite(radius_seed):
            radius_seed = 50.0
        scale_seed = float(max(np.max(y_raw), 1.0))
        background_seed = float(max(np.min(y_raw), 0.0))
        polydisperse = bool(decay_ratio < 0.85)
        confidence = min(0.9, 0.4 + 0.3 * decay_ratio)

        return {
            "model_key": model_key,
            "recommended_backend": "saxs_physics",
            "confidence": float(confidence),
            "peak_count": None,
            "peak_centers": [],
            "peak_shape_label": None,
            "notes": "Heuristic SAXS seed from monotonicity and oscillation count.",
            "saxs_shape": shape,
            "saxs_polydisperse": polydisperse,
            "saxs_use_porod": False,
            "initial_overrides": {
                "radius": float(radius_seed),
                "scale": float(scale_seed),
                "background": float(background_seed),
                "sigma_rel": 0.10,
                "porod_scale": 0.01,
                "porod_exp": 4.0,
            },
        }

    raise ValueError(f"Unsupported ML model key: {model_key}")


def run_ml_prediction(
    x_data,
    y_data,
    *,
    model_key,
    x_min=None,
    x_max=None,
    log_y=False,
    normalize_y=True,
    smooth_window=1,
    resample_points=256,
):
    """Run preprocessing + prediction in one call."""
    pre = preprocess_curve_for_ml(
        x_data,
        y_data,
        x_min=x_min,
        x_max=x_max,
        log_y=log_y,
        normalize_y=normalize_y,
        smooth_window=smooth_window,
        resample_points=resample_points,
    )
    pred = predict_ml_seed(pre, model_key=model_key)
    return {
        "preprocessed": pre,
        "prediction": pred,
    }


def _pick_saxs_model_class(shape_key, polydisperse=False):
    """Map shape key + mode to a pySAXSFitting model class."""
    shape_key = str(shape_key).lower().strip()
    if polydisperse:
        mapping = {
            "sphere": PolydisperseSphereModel,
            "cube": PolydisperseCubeModel,
            "tetrahedron": PolydisperseTetrahedronModel,
            "octahedron": PolydisperseOctahedronModel,
            "rhombic_dodecahedron": PolydisperseRhombicDodecahedronModel,
            "square_pyramid": PolydisperseSquarePyramidModel,
        }
    else:
        mapping = {
            "sphere": SphereModel,
            "cube": CubeModel,
            "tetrahedron": TetrahedronModel,
            "octahedron": OctahedronModel,
            "rhombic_dodecahedron": RhombicDodecahedronModel,
            "square_pyramid": SquarePyramidModel,
        }
    model_cls = mapping.get(shape_key)
    if model_cls is None:
        raise ValueError(f"Unsupported SAXS model shape: {shape_key}")
    return model_cls


def run_saxs_fit(
    x_data,
    y_data,
    *,
    shape,
    shape_label=None,
    polydisperse=False,
    use_porod=False,
    maxiter=2000,
    x_col_name="x",
    y_col_name="y",
    initial_overrides=None,
):
    """Run a SAXS fit through pySAXSFitting + pyFitting and return fit-state dict."""
    if PYSAXS_INTERFACE_AVAILABLE and _pysaxs_fit_formfactor_curve is not None:
        try:
            state = _pysaxs_fit_formfactor_curve(
                np.asarray(x_data, dtype=float),
                np.asarray(y_data, dtype=float),
                shape=str(shape),
                polydisperse=bool(polydisperse),
                use_porod=bool(use_porod),
                maxiter=int(maxiter),
                initial_overrides=initial_overrides,
                auto_shape_from_dips=False,
            )
            state = dict(state)
            state["x_col"] = x_col_name
            state["y_col"] = y_col_name
            if shape_label:
                state["shape_label"] = f"SAXS {shape_label}"
                state["saxs_shape_label"] = str(shape_label)
                if state.get("component_table"):
                    state["component_table"][0]["model"] = str(shape_label)
            return state
        except Exception:
            pass

    if not PYFITTING_AVAILABLE:
        raise RuntimeError(
            "pyFitting unavailable for SAXS backend: "
            f"{PYFITTING_IMPORT_ERROR or 'unknown import error'}"
        )
    if not PYSAXS_AVAILABLE:
        raise RuntimeError(
            "pySAXSFitting backend unavailable: "
            f"{PYSAXS_IMPORT_ERROR or 'unknown import error'}"
        )

    x_fit = np.asarray(x_data, dtype=float)
    y_fit_data = np.asarray(y_data, dtype=float)
    if len(x_fit) < 5 or len(y_fit_data) < 5:
        raise ValueError("Not enough points to fit.")
    if len(x_fit) != len(y_fit_data):
        raise ValueError("x/y length mismatch.")

    model_cls = _pick_saxs_model_class(shape, polydisperse=polydisperse)
    model_kwargs = {
        "use_porod": bool(use_porod),
        "normalize": True,
    }
    if polydisperse:
        model_kwargs["num_points"] = 60
    model = model_cls(**model_kwargs)

    if hasattr(model, "set_default_bounds"):
        model.set_default_bounds()

    initial_guess = model.get_initial_guess(x_fit, y_fit_data)
    if initial_overrides:
        for key, value in initial_overrides.items():
            if value is None:
                continue
            try:
                initial_guess[key] = float(value)
            except Exception:
                continue

    bounds = {}
    try:
        bounds = dict(model.get_parameters().bounds)
    except Exception:
        bounds = {}

    fit_kwargs = {
        "initial_guess": initial_guess,
        "maxiter": int(maxiter),
    }
    if bounds:
        fit_kwargs["bounds"] = bounds

    fit_result = Fitter(ArrayData(x_fit, y_fit_data), model).fit(**fit_kwargs)
    fit_params = dict(fit_result.parameters.values)
    shape_key = str(shape).lower().strip()
    display_shape = shape_label or SAXS_MODEL_KEY_TO_LABEL.get(shape_key, shape_key.title())

    component_row = {
        "model": display_shape,
        "polydisperse": bool(polydisperse),
        "use_porod": bool(use_porod),
    }
    for key in ["radius", "sigma_rel", "scale", "background", "porod_scale", "porod_exp"]:
        if key in fit_params:
            component_row[key] = float(fit_params[key])

    return {
        "backend": "saxs_physics",
        "shape": shape_key,
        "shape_label": f"SAXS {display_shape}",
        "x_col": x_col_name,
        "y_col": y_col_name,
        "x": np.asarray(x_fit, dtype=float),
        "y": np.asarray(y_fit_data, dtype=float),
        "y_fit": np.asarray(fit_result.y_fit, dtype=float),
        "params": fit_params,
        "metrics": dict(fit_result.metrics),
        "components": [],
        "component_table": [component_row],
        "success": bool(fit_result.success),
        "message": str(fit_result.message),
        "n_peaks": 0,
        "peak_guesses": [],
        "saxs_shape": shape_key,
        "saxs_shape_label": display_shape,
        "saxs_polydisperse": bool(polydisperse),
        "saxs_use_porod": bool(use_porod),
    }


def simulate_saxs_curve(
    *,
    shape,
    q_min=0.01,
    q_max=0.30,
    n_points=1200,
    radius=50.0,
    scale=1.0,
    polydisperse=False,
    sigma_rel=0.08,
    use_porod=False,
    porod_scale=0.01,
    porod_exp=4.0,
    background_mode="constant",
    background_const=0.01,
    background_decay_amp=0.05,
    background_decay_q0=0.05,
    background_decay_exp=3.0,
    noise_level=0.02,
    seed=1234,
    q_col_name="q",
    y_col_name="intensity",
    include_components=True,
):
    """
    Simulate one SAXS curve for testing fitting workflows.

    Returns:
        (dataframe, metadata dict)
    """
    if not PYSAXS_AVAILABLE:
        raise RuntimeError(
            "pySAXSFitting backend unavailable for simulation: "
            f"{PYSAXS_IMPORT_ERROR or 'unknown import error'}"
        )

    q_min = float(q_min)
    q_max = float(q_max)
    if q_max <= q_min:
        raise ValueError("q_max must be larger than q_min.")
    n_points = int(max(16, n_points))

    q = np.linspace(q_min, q_max, n_points)
    shape_key = str(shape).strip().lower()

    model_cls = _pick_saxs_model_class(shape_key, polydisperse=bool(polydisperse))
    model_kwargs = {
        "use_porod": bool(use_porod),
        "normalize": True,
    }
    if polydisperse:
        model_kwargs["num_points"] = 60
    model = model_cls(**model_kwargs)

    # Build parameter dict from model's own initial-guess keys for compatibility.
    params = {}
    try:
        guess = dict(model.get_initial_guess(q, np.ones_like(q)))
    except Exception:
        guess = {}
    for key, value in guess.items():
        try:
            params[str(key)] = float(value)
        except Exception:
            pass

    if "radius" in params:
        params["radius"] = float(radius)
    if "scale" in params:
        params["scale"] = float(scale)
    if "height" in params:
        params["height"] = float(max(1e-9, 1.3 * float(radius)))
    if "background" in params:
        params["background"] = 0.0
    if "sigma_rel" in params:
        params["sigma_rel"] = float(max(0.0, sigma_rel))
    if "sigma_rel_1" in params:
        params["sigma_rel_1"] = float(max(0.0, sigma_rel))
    if "sigma_rel_2" in params:
        params["sigma_rel_2"] = float(max(0.0, sigma_rel))
    if "porod_scale" in params:
        params["porod_scale"] = float(max(0.0, porod_scale))
    if "porod_exp" in params:
        params["porod_exp"] = float(max(0.1, porod_exp))

    y_shape = np.asarray(model.evaluate(q, **params), dtype=float)
    finite_mask = np.isfinite(y_shape)
    if not np.all(finite_mask):
        fill_value = float(np.median(y_shape[finite_mask])) if np.any(finite_mask) else 0.0
        y_shape = np.where(finite_mask, y_shape, fill_value)

    mode = str(background_mode).strip().lower()
    if mode not in {"constant", "decay"}:
        raise ValueError(f"Unsupported background_mode: {background_mode}")
    if mode == "constant":
        y_background = np.full_like(q, float(background_const), dtype=float)
    else:
        q0 = float(max(1e-12, background_decay_q0))
        exponent = float(max(0.1, background_decay_exp))
        y_background = float(background_const) + float(background_decay_amp) / (
            1.0 + (q / q0) ** exponent
        )

    y_clean = np.asarray(y_shape + y_background, dtype=float)
    noise_sigma = float(max(0.0, noise_level)) * max(float(np.ptp(y_clean)), 1e-12)
    y_noisy = y_clean.copy()
    if noise_sigma > 0:
        rng = np.random.default_rng(int(seed))
        y_noisy = y_noisy + rng.normal(0.0, noise_sigma, size=len(y_noisy))

    y_noisy = np.maximum(y_noisy, 1e-12)

    output = {
        str(q_col_name): q,
        str(y_col_name): y_noisy,
    }
    if include_components:
        output[f"{y_col_name}_clean"] = y_clean
        output[f"{y_col_name}_shape"] = y_shape
        output[f"{y_col_name}_background"] = y_background

    key_to_label = get_saxs_shape_options()[1]
    shape_label = key_to_label.get(shape_key, shape_key.replace("_", " ").title())
    metadata = {
        "shape": shape_key,
        "shape_label": shape_label,
        "q_min": q_min,
        "q_max": q_max,
        "n_points": int(n_points),
        "radius": float(radius),
        "scale": float(scale),
        "polydisperse": bool(polydisperse),
        "sigma_rel": float(max(0.0, sigma_rel)),
        "use_porod": bool(use_porod),
        "porod_scale": float(max(0.0, porod_scale)),
        "porod_exp": float(max(0.1, porod_exp)),
        "background_mode": mode,
        "background_const": float(background_const),
        "background_decay_amp": float(background_decay_amp),
        "background_decay_q0": float(max(1e-12, background_decay_q0)),
        "background_decay_exp": float(max(0.1, background_decay_exp)),
        "noise_level": float(max(0.0, noise_level)),
        "noise_sigma": float(noise_sigma),
        "seed": int(seed),
        "model_params": dict(params),
    }
    return pd.DataFrame(output), metadata


def format_metric(value, fmt=".5g"):
    """Format numeric metric safely for display."""
    try:
        value = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(value):
        return "n/a"
    return format(value, fmt)


def sanitize_filename(name):
    """Convert arbitrary label to a safe filename fragment."""
    if not name:
        return "fit_result"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._-") or "fit_result"


def fit_state_to_tables(fit_state):
    """Convert fit state into export-friendly DataFrames and metadata dict."""
    x_name = fit_state.get("x_col", "x")
    y_name = fit_state.get("y_col", "y")

    x = np.asarray(fit_state.get("x", []), dtype=float)
    y_raw = np.asarray(fit_state.get("y", []), dtype=float)
    y_fit = np.asarray(fit_state.get("y_fit", []), dtype=float)
    params = dict(fit_state.get("params", {}))
    baseline = float(params.get("c", params.get("background", 0.0)))

    curve_df = pd.DataFrame({
        x_name: x,
        f"{y_name}_raw": y_raw,
        f"{y_name}_fit_sum": y_fit,
        "fit_baseline": np.full(len(x), baseline, dtype=float),
        "fit_residual": y_raw - y_fit if len(x) else np.array([], dtype=float),
    })
    for idx, comp in enumerate(fit_state.get("components", []), start=1):
        curve_df[f"fit_peak_{idx}"] = np.asarray(comp, dtype=float)

    params_df = pd.DataFrame(
        [{"parameter": k, "value": v} for k, v in sorted(params.items())]
    )
    metrics_df = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in sorted(fit_state.get("metrics", {}).items())]
    )
    peaks_df = pd.DataFrame(fit_state.get("component_table", []))
    metadata = {
        "backend": fit_state.get("backend", "general_peaks"),
        "shape": fit_state.get("shape"),
        "shape_label": fit_state.get("shape_label"),
        "source_file": fit_state.get("source_file"),
        "source_curve_key": fit_state.get("source_curve_key"),
        "curve_label": fit_state.get("curve_label"),
        "success": fit_state.get("success"),
        "message": fit_state.get("message"),
        "n_peaks": fit_state.get("n_peaks"),
        "peak_guesses": fit_state.get("peak_guesses", []),
        "saxs_shape": fit_state.get("saxs_shape"),
        "saxs_shape_label": fit_state.get("saxs_shape_label"),
        "saxs_polydisperse": fit_state.get("saxs_polydisperse"),
        "saxs_use_porod": fit_state.get("saxs_use_porod"),
        "ml_seed_model": fit_state.get("ml_seed_model"),
        "ml_seed_confidence": fit_state.get("ml_seed_confidence"),
    }
    return curve_df, params_df, metrics_df, peaks_df, metadata


def create_fit_plot_figure(fit_state):
    """Create a standardized plotly fit figure for display/export."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fit_state["x"],
            y=fit_state["y"],
            mode="markers",
            name="Data",
            marker=dict(size=5, color="#111111"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fit_state["x"],
            y=fit_state["y_fit"],
            mode="lines",
            name="Fit",
            line=dict(color="#d62728", width=2.2),
        )
    )
    for idx, component in enumerate(fit_state.get("components", []), start=1):
        fig.add_trace(
            go.Scatter(
                x=fit_state["x"],
                y=component,
                mode="lines",
                name=f"Peak {idx}",
                line=dict(color="#7f7f7f", width=1.2, dash="dot"),
                opacity=0.65,
            )
        )
    fig.update_layout(
        title=f"Fit Overlay ({fit_state.get('shape_label', fit_state.get('shape', 'fit'))})",
        height=460,
        showlegend=True,
    )
    fig.update_xaxes(title=fit_state.get("x_col", "x"))
    fig.update_yaxes(title=fit_state.get("y_col", "y"))
    return fig


def try_plot_png_bytes(fig):
    """
    Try exporting plotly figure to PNG. Returns None if unavailable or timed out.

    Set `NANOORGANIZER_PNG_EXPORT_TIMEOUT_SEC` to override timeout.
    Set `NANOORGANIZER_DISABLE_PNG_EXPORT=1` to disable PNG export attempts.
    """
    if os.environ.get("NANOORGANIZER_DISABLE_PNG_EXPORT", "0").strip() == "1":
        return None

    try:
        timeout_sec = float(os.environ.get("NANOORGANIZER_PNG_EXPORT_TIMEOUT_SEC", "4.0"))
    except Exception:
        timeout_sec = 4.0
    timeout_sec = max(0.0, timeout_sec)

    if timeout_sec <= 0:
        try:
            return fig.to_image(format="png", width=1400, height=900)
        except Exception:
            return None

    class _PngExportTimeout(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise _PngExportTimeout()

    if hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer"):
        prev_handler = signal.getsignal(signal.SIGALRM)
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout_sec)
            return fig.to_image(format="png", width=1400, height=900)
        except Exception:
            return None
        finally:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
            except Exception:
                pass
            try:
                signal.signal(signal.SIGALRM, prev_handler)
            except Exception:
                pass

    try:
        return fig.to_image(format="png", width=1400, height=900)
    except Exception:
        return None


def _fit_arrays_payload(fit_state):
    """Build a compact arrays payload for NPZ export."""
    payload = {
        "x": np.asarray(fit_state.get("x", []), dtype=float),
        "y_raw": np.asarray(fit_state.get("y", []), dtype=float),
        "y_fit_sum": np.asarray(fit_state.get("y_fit", []), dtype=float),
    }
    for i, comp in enumerate(fit_state.get("components", []), start=1):
        payload[f"peak_{i}"] = np.asarray(comp, dtype=float)
    return payload


def build_fit_zip_bytes(fit_state):
    """Build one ZIP payload with fitted data, parameters, metrics, and plot."""
    curve_df, params_df, metrics_df, peaks_df, metadata = fit_state_to_tables(fit_state)
    fig = create_fit_plot_figure(fit_state)

    npz_buffer = io.BytesIO()
    np.savez_compressed(npz_buffer, **_fit_arrays_payload(fit_state))

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("fitted_curve.csv", curve_df.to_csv(index=False))
        zf.writestr("fit_parameters.csv", params_df.to_csv(index=False))
        zf.writestr("fit_metrics.csv", metrics_df.to_csv(index=False))
        zf.writestr("fit_peaks.csv", peaks_df.to_csv(index=False))
        zf.writestr("fit_arrays.npz", npz_buffer.getvalue())
        zf.writestr("fit_summary.json", json.dumps(metadata, indent=2))
        zf.writestr("fit_plot.html", fig.to_html(full_html=True, include_plotlyjs="cdn"))
        png_bytes = try_plot_png_bytes(fig)
        if png_bytes is not None:
            zf.writestr("fit_plot.png", png_bytes)
        else:
            zf.writestr(
                "fit_plot_note.txt",
                "PNG export unavailable (install kaleido). HTML plot is included.",
            )

    buffer.seek(0)
    return buffer.getvalue()


def build_batch_fit_zip_bytes(fit_states_by_curve):
    """Build ZIP payload for multiple fit states."""
    buffer = io.BytesIO()
    summary_rows = []
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, (curve_key, fit_state) in enumerate(fit_states_by_curve.items(), start=1):
            curve_df, params_df, metrics_df, peaks_df, metadata = fit_state_to_tables(fit_state)
            fig = create_fit_plot_figure(fit_state)
            curve_name = fit_state.get("curve_label") or curve_key
            prefix = f"{idx:03d}_{sanitize_filename(curve_name)}"

            zf.writestr(f"{prefix}/fitted_curve.csv", curve_df.to_csv(index=False))
            zf.writestr(f"{prefix}/fit_parameters.csv", params_df.to_csv(index=False))
            zf.writestr(f"{prefix}/fit_metrics.csv", metrics_df.to_csv(index=False))
            zf.writestr(f"{prefix}/fit_peaks.csv", peaks_df.to_csv(index=False))

            npz_buffer = io.BytesIO()
            np.savez_compressed(npz_buffer, **_fit_arrays_payload(fit_state))
            zf.writestr(f"{prefix}/fit_arrays.npz", npz_buffer.getvalue())
            zf.writestr(f"{prefix}/fit_summary.json", json.dumps(metadata, indent=2))
            zf.writestr(f"{prefix}/fit_plot.html", fig.to_html(full_html=True, include_plotlyjs="cdn"))

            png_bytes = try_plot_png_bytes(fig)
            if png_bytes is not None:
                zf.writestr(f"{prefix}/fit_plot.png", png_bytes)

            summary_rows.append({
                "curve_key": curve_key,
                "curve_label": curve_name,
                "success": fit_state.get("success"),
                "message": fit_state.get("message"),
                "shape": fit_state.get("shape"),
                "n_peaks": fit_state.get("n_peaks"),
                "r2": fit_state.get("metrics", {}).get("r2"),
                "rmse": fit_state.get("metrics", {}).get("rmse"),
            })

        summary_df = pd.DataFrame(summary_rows)
        zf.writestr("batch_summary.csv", summary_df.to_csv(index=False))

    buffer.seek(0)
    return buffer.getvalue()


def get_export_root_dir(base_dir=None):
    """Get or create the server-side folder for persisted fit exports."""
    if base_dir is None:
        root = Path(__file__).resolve().parents[3] / "results" / "fitting_exports"
    else:
        root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_single_fit_to_server(fit_state, *, export_root=None):
    """Persist one fit result to server files and return saved paths."""
    root = get_export_root_dir(base_dir=export_root)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = sanitize_filename(fit_state.get("curve_label") or fit_state.get("source_curve_key"))
    run_dir = root / f"{ts}_{label}"
    run_dir.mkdir(parents=True, exist_ok=True)

    curve_df, params_df, metrics_df, peaks_df, metadata = fit_state_to_tables(fit_state)
    fig = create_fit_plot_figure(fit_state)

    curve_df.to_csv(run_dir / "fitted_curve.csv", index=False)
    params_df.to_csv(run_dir / "fit_parameters.csv", index=False)
    metrics_df.to_csv(run_dir / "fit_metrics.csv", index=False)
    peaks_df.to_csv(run_dir / "fit_peaks.csv", index=False)
    (run_dir / "fit_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (run_dir / "fit_plot.html").write_text(fig.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")

    png_path = run_dir / "fit_plot.png"
    png_bytes = try_plot_png_bytes(fig)
    if png_bytes is not None:
        png_path.write_bytes(png_bytes)
    else:
        png_path = None

    npz_path = run_dir / "fit_arrays.npz"
    np.savez_compressed(npz_path, **_fit_arrays_payload(fit_state))

    zip_path = run_dir / "fit_results.zip"
    zip_path.write_bytes(build_fit_zip_bytes(fit_state))

    return {
        "run_dir": run_dir,
        "zip_path": zip_path,
        "npz_path": npz_path,
        "png_path": png_path,
    }


def save_batch_fit_to_server(fit_states_by_curve, summary_rows, *, export_root=None):
    """Persist batch fit results to server files and return saved paths."""
    root = get_export_root_dir(base_dir=export_root)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{ts}_batch_{len(fit_states_by_curve)}curves"
    run_dir.mkdir(parents=True, exist_ok=True)

    per_curve_dirs = []
    for idx, (curve_key, fit_state) in enumerate(fit_states_by_curve.items(), start=1):
        curve_df, params_df, metrics_df, peaks_df, metadata = fit_state_to_tables(fit_state)
        fig = create_fit_plot_figure(fit_state)
        curve_name = fit_state.get("curve_label") or curve_key
        curve_dir = run_dir / f"{idx:03d}_{sanitize_filename(curve_name)}"
        curve_dir.mkdir(parents=True, exist_ok=True)

        curve_df.to_csv(curve_dir / "fitted_curve.csv", index=False)
        params_df.to_csv(curve_dir / "fit_parameters.csv", index=False)
        metrics_df.to_csv(curve_dir / "fit_metrics.csv", index=False)
        peaks_df.to_csv(curve_dir / "fit_peaks.csv", index=False)
        (curve_dir / "fit_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (curve_dir / "fit_plot.html").write_text(fig.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")

        png_bytes = try_plot_png_bytes(fig)
        if png_bytes is not None:
            (curve_dir / "fit_plot.png").write_bytes(png_bytes)
        else:
            (curve_dir / "fit_plot_note.txt").write_text(
                "PNG export unavailable (install kaleido). HTML plot is included.",
                encoding="utf-8",
            )

        np.savez_compressed(curve_dir / "fit_arrays.npz", **_fit_arrays_payload(fit_state))
        per_curve_dirs.append(curve_dir)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = run_dir / "batch_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    zip_path = run_dir / "batch_fit_results.zip"
    zip_path.write_bytes(build_batch_fit_zip_bytes(fit_states_by_curve))

    return {
        "run_dir": run_dir,
        "zip_path": zip_path,
        "summary_path": summary_path,
        "curve_dirs": per_curve_dirs,
    }
