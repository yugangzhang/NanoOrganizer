#!/usr/bin/env python3
"""
Fitting backend adapters for NanoOrganizer CSV Plotter.

This module keeps backend-specific fitting logic isolated from Streamlit UI code.
"""

from pathlib import Path
import sys
import numpy as np

# ---------------------------------------------------------------------------
# pyFitting import (required by all adapters)
# ---------------------------------------------------------------------------
PYFITTING_AVAILABLE = False
PYFITTING_IMPORT_ERROR = ""
ArrayData = None
Fitter = None
try:
    from pyFitting import ArrayData, Fitter  # type: ignore
    PYFITTING_AVAILABLE = True
except Exception as e:
    _repo_parent = Path(__file__).resolve().parents[4]
    _pyfitting_repo = _repo_parent / "pyFitting"
    if _pyfitting_repo.exists():
        sys.path.insert(0, str(_pyfitting_repo))
        try:
            from pyFitting import ArrayData, Fitter  # type: ignore
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
    from pySAXSFitting.fitting import (  # type: ignore
        get_model_library_rows as _pysaxs_get_model_library_rows,
        suggest_saxs_seed as _pysaxs_suggest_saxs_seed,
        predict_ml_seed as _pysaxs_predict_ml_seed,
        fit_formfactor_curve as _pysaxs_fit_formfactor_curve,
    )
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
            from pySAXSFitting.fitting import (  # type: ignore
                get_model_library_rows as _pysaxs_get_model_library_rows,
                suggest_saxs_seed as _pysaxs_suggest_saxs_seed,
                predict_ml_seed as _pysaxs_predict_ml_seed,
                fit_formfactor_curve as _pysaxs_fit_formfactor_curve,
            )
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
                pred = _pysaxs_predict_ml_seed(
                    np.asarray(preprocessed["x_raw"], dtype=float),
                    np.asarray(preprocessed["y_raw"], dtype=float),
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
