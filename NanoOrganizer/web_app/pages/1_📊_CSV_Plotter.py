#!/usr/bin/env python3
"""
Enhanced CSV/NPZ Plotter - Quick visualization with per-curve styling.

Features:
- CSV, TXT, DAT, NPZ file support
- Individual color/marker selection per curve
- Smart filename display
- Multi-curve overlay
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402

import streamlit as st  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pathlib import Path  # noqa: E402
import io  # noqa: E402
import sys  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import zipfile  # noqa: E402
from datetime import datetime  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from components.folder_browser import folder_browser  # noqa: E402
from components.floating_button import floating_sidebar_toggle  # noqa: E402
from components.fitting_adapters import (  # noqa: E402
    PYSAXS_AVAILABLE,
    PYSAXS_IMPORT_ERROR,
    SAXS_MODEL_LABEL_TO_KEY,
    SAXS_MODEL_KEY_TO_LABEL,
    get_saxs_model_detail,
    get_saxs_model_library_rows,
    get_saxs_shape_options,
    list_ml_models,
    run_ml_prediction,
    run_saxs_fit,
)

# Optional pyFitting dependency (tries installed package first, then sibling repo)
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

# User-mode restriction (set by nanoorganizer_user)
_user_mode = st.session_state.get("user_mode", False)
_start_dir = st.session_state.get("user_start_dir", None)

# ---------------------------------------------------------------------------
# Styling options
# ---------------------------------------------------------------------------
COLORS_NAMED = {
    'Blue': '#1f77b4', 'Orange': '#ff7f0e', 'Green': '#2ca02c',
    'Red': '#d62728', 'Purple': '#9467bd', 'Brown': '#8c564b',
    'Pink': '#e377c2', 'Gray': '#7f7f7f', 'Olive': '#bcbd22',
    'Cyan': '#17becf', 'Navy': '#000080', 'Magenta': '#FF00FF',
    'Yellow': '#FFD700', 'Teal': '#008080', 'Lime': '#00FF00'
}

MARKERS_DICT = {
    'Circle': 'o', 'Square': 's', 'Triangle Up': '^', 'Triangle Down': 'v',
    'Diamond': 'D', 'Pentagon': 'p', 'Star': '*', 'Hexagon': 'h',
    'Plus': '+', 'X': 'x', 'Point': '.', 'None': 'None'
}

LINESTYLES_DICT = {
    'Solid': '-', 'Dashed': '--', 'Dash-dot': '-.', 'Dotted': ':',
    'None': 'None'
}

FIT_SHAPES = {
    "Gaussian": "gaussian",
    "Lorentzian": "lorentzian",
    "Pseudo-Voigt": "pseudo_voigt",
}

FIT_BACKENDS = {
    "General Peaks": "general_peaks",
    "SAXS Physics": "saxs_physics",
    "ML-Assisted (Preview)": "ml_assisted",
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def shorten_path(path_str, max_length=40):
    """Shorten long file paths for display."""
    if len(path_str) <= max_length:
        return path_str

    path = Path(path_str)
    filename = path.name

    if len(filename) > max_length - 3:
        return "..." + filename[-(max_length-3):]

    # Show ... parent .../filename
    parent = str(path.parent)
    if len(parent) + len(filename) + 4 > max_length:
        return ".../" + filename
    else:
        remaining = max_length - len(filename) - 4
        return parent[:remaining] + ".../" + filename


def load_data_file(file_path):
    """Load CSV, TXT, DAT, or NPZ file."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    try:
        if suffix == '.npz':
            data = np.load(file_path)
            # Convert to dataframe (assumes 1D or 2D arrays)
            df_dict = {}
            for key in data.files:
                arr = data[key]
                if arr.ndim == 1:
                    df_dict[key] = arr
                elif arr.ndim == 2:
                    # Flatten or take first column
                    df_dict[key] = arr.flatten()
            return pd.DataFrame(df_dict)
        else:
            # Try CSV/TXT with different delimiters
            df = pd.read_csv(file_path, sep=',')
            if len(df.columns) == 1:
                df = pd.read_csv(file_path, sep='\t')
            if len(df.columns) == 1:
                df = pd.read_csv(file_path, sep=r'\s+')
            return df
    except Exception as e:
        st.error(f"Error loading {path.name}: {e}")
        return None


def _save_fig_to_bytes(fig, format='png', dpi=300):
    """Save matplotlib figure to bytes buffer for download."""
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf


def browse_directory(base_dir, pattern="*.*"):
    """Browse directory and find files matching pattern."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    files = list(base_path.rglob(pattern))
    return [str(f) for f in sorted(files)]


def _safe_widget_key(prefix, name):
    """Create compact deterministic keys for Streamlit widgets."""
    return f"{prefix}_{abs(hash(str(name)))}"


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


def _dedupe_sorted(values, tol):
    """Sort values and remove near-duplicates."""
    cleaned = sorted(float(v) for v in values)
    unique = []
    for value in cleaned:
        if not unique or abs(value - unique[-1]) > tol:
            unique.append(value)
    return unique


def _prepare_xy(x_values, y_values, x_min=None, x_max=None):
    """Convert to numeric arrays, drop NaNs, optionally apply x-range."""
    x = pd.to_numeric(pd.Series(x_values), errors='coerce').to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(y_values), errors='coerce').to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if x_min is not None:
        mask &= x >= float(x_min)
    if x_max is not None:
        mask &= x <= float(x_max)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return x, y
    order = np.argsort(x)
    return x[order], y[order]


def _simulate_peak_curve(n_peaks, shape, x_min, x_max, n_points, baseline, noise_level, seed):
    """Generate a synthetic 1D curve with one or more peaks."""
    rng = np.random.default_rng(int(seed))
    x = np.linspace(float(x_min), float(x_max), int(n_points))
    span = max(float(x_max) - float(x_min), 1e-12)

    centers = np.sort(rng.uniform(x_min + 0.1 * span, x_max - 0.1 * span, size=int(n_peaks)))
    widths = rng.uniform(0.02 * span, 0.08 * span, size=int(n_peaks))
    amplitudes = rng.uniform(0.8, 2.0, size=int(n_peaks))
    etas = rng.uniform(0.2, 0.8, size=int(n_peaks)) if shape == "pseudo_voigt" else np.zeros(int(n_peaks))

    y = np.full_like(x, float(baseline), dtype=float)
    component_rows = []
    for i in range(int(n_peaks)):
        eta = float(etas[i]) if shape == "pseudo_voigt" else 0.5
        profile = _shape_profile(x, centers[i], widths[i], shape, eta=eta)
        y += amplitudes[i] * profile
        component_rows.append({
            "peak": i + 1,
            "A": float(amplitudes[i]),
            "mu": float(centers[i]),
            "w": float(widths[i]),
            "eta": float(eta) if shape == "pseudo_voigt" else np.nan,
        })

    noise_sigma = float(noise_level) * max(float(np.ptp(y)), 1e-9)
    if noise_sigma > 0:
        y = y + rng.normal(0.0, noise_sigma, size=len(x))

    df = pd.DataFrame({"x": x, "intensity": y})
    return df, component_rows


def _format_metric(value, fmt=".5g"):
    """Format numeric metric safely for display."""
    try:
        value = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(value):
        return "n/a"
    return format(value, fmt)


def _sanitize_filename(name):
    """Convert arbitrary label to safe filename fragment."""
    if not name:
        return "fit_result"
    return re.sub(r'[^A-Za-z0-9._-]+', "_", str(name)).strip("._-") or "fit_result"


def _run_multipeak_fit(
    x_data,
    y_data,
    *,
    shape,
    shape_label,
    n_peaks,
    maxiter,
    peak_guesses=None,
    x_col_name="x",
    y_col_name="y",
):
    """Run one multi-peak fit and return a UI-ready state dict."""
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
    guess_peaks = _dedupe_sorted(peak_guesses or [], max(x_span / 5000.0, 1e-9))

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

    return {
        "backend": "general_peaks",
        "shape": shape,
        "shape_label": shape_label,
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


def _fit_state_to_tables(fit_state):
    """Convert fit state into export-friendly DataFrames."""
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


def _create_fit_plot_figure(fit_state):
    """Create a standardized fit plot figure for display/export."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fit_state["x"],
            y=fit_state["y"],
            mode='lines+markers',
            name='Data',
            line=dict(color='#1f77b4', width=1.3),
            marker=dict(size=4, color='#1f77b4'),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fit_state["x"],
            y=fit_state["y_fit"],
            mode='lines',
            name='Fit',
            line=dict(color='#2ca02c', width=2.2),
        )
    )
    for idx, component in enumerate(fit_state.get("components", []), start=1):
        fig.add_trace(
            go.Scatter(
                x=fit_state["x"],
                y=component,
                mode='lines',
                name=f'Peak {idx}',
                line=dict(width=1.2, dash='dot'),
                opacity=0.8,
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


def _try_plot_png_bytes(fig):
    """Try exporting plotly figure to PNG; return None if unavailable."""
    try:
        return fig.to_image(format="png", width=1400, height=900)
    except Exception:
        return None


def _get_export_root_dir():
    """Server-side folder for stored fit exports."""
    root = Path(__file__).resolve().parents[3] / "results" / "fitting_exports"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build_fit_zip_bytes(fit_state):
    """Build one ZIP payload with fitted data, parameters, and metrics."""
    curve_df, params_df, metrics_df, peaks_df, metadata = _fit_state_to_tables(fit_state)
    fig = _create_fit_plot_figure(fit_state)
    npz_data = {
        "x": np.asarray(fit_state.get("x", []), dtype=float),
        "y_raw": np.asarray(fit_state.get("y", []), dtype=float),
        "y_fit_sum": np.asarray(fit_state.get("y_fit", []), dtype=float),
    }
    for i, comp in enumerate(fit_state.get("components", []), start=1):
        npz_data[f"peak_{i}"] = np.asarray(comp, dtype=float)
    npz_buffer = io.BytesIO()
    np.savez_compressed(npz_buffer, **npz_data)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("fitted_curve.csv", curve_df.to_csv(index=False))
        zf.writestr("fit_parameters.csv", params_df.to_csv(index=False))
        zf.writestr("fit_metrics.csv", metrics_df.to_csv(index=False))
        zf.writestr("fit_peaks.csv", peaks_df.to_csv(index=False))
        zf.writestr("fit_arrays.npz", npz_buffer.getvalue())
        zf.writestr("fit_summary.json", json.dumps(metadata, indent=2))
        zf.writestr("fit_plot.html", fig.to_html(full_html=True, include_plotlyjs="cdn"))
        png_bytes = _try_plot_png_bytes(fig)
        if png_bytes is not None:
            zf.writestr("fit_plot.png", png_bytes)
        else:
            zf.writestr(
                "fit_plot_note.txt",
                "PNG export unavailable (install kaleido). HTML plot is included.",
            )
    buffer.seek(0)
    return buffer.getvalue()


def _build_batch_fit_zip_bytes(fit_states_by_curve):
    """Build ZIP payload for multiple fit states."""
    buffer = io.BytesIO()
    summary_rows = []
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, (curve_key, fit_state) in enumerate(fit_states_by_curve.items(), start=1):
            curve_df, params_df, metrics_df, peaks_df, metadata = _fit_state_to_tables(fit_state)
            fig = _create_fit_plot_figure(fit_state)
            curve_name = fit_state.get("curve_label") or curve_key
            prefix = f"{idx:03d}_{_sanitize_filename(curve_name)}"

            zf.writestr(f"{prefix}/fitted_curve.csv", curve_df.to_csv(index=False))
            zf.writestr(f"{prefix}/fit_parameters.csv", params_df.to_csv(index=False))
            zf.writestr(f"{prefix}/fit_metrics.csv", metrics_df.to_csv(index=False))
            zf.writestr(f"{prefix}/fit_peaks.csv", peaks_df.to_csv(index=False))
            npz_data = {
                "x": np.asarray(fit_state.get("x", []), dtype=float),
                "y_raw": np.asarray(fit_state.get("y", []), dtype=float),
                "y_fit_sum": np.asarray(fit_state.get("y_fit", []), dtype=float),
            }
            for i, comp in enumerate(fit_state.get("components", []), start=1):
                npz_data[f"peak_{i}"] = np.asarray(comp, dtype=float)
            npz_buffer = io.BytesIO()
            np.savez_compressed(npz_buffer, **npz_data)
            zf.writestr(f"{prefix}/fit_arrays.npz", npz_buffer.getvalue())
            zf.writestr(f"{prefix}/fit_summary.json", json.dumps(metadata, indent=2))
            zf.writestr(f"{prefix}/fit_plot.html", fig.to_html(full_html=True, include_plotlyjs="cdn"))
            png_bytes = _try_plot_png_bytes(fig)
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


def _save_single_fit_to_server(fit_state):
    """Persist one fit result to server files and return saved paths."""
    export_root = _get_export_root_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = _sanitize_filename(fit_state.get("curve_label") or fit_state.get("source_curve_key"))
    run_dir = export_root / f"{ts}_{label}"
    run_dir.mkdir(parents=True, exist_ok=True)

    curve_df, params_df, metrics_df, peaks_df, metadata = _fit_state_to_tables(fit_state)
    fig = _create_fit_plot_figure(fit_state)

    curve_df.to_csv(run_dir / "fitted_curve.csv", index=False)
    params_df.to_csv(run_dir / "fit_parameters.csv", index=False)
    metrics_df.to_csv(run_dir / "fit_metrics.csv", index=False)
    peaks_df.to_csv(run_dir / "fit_peaks.csv", index=False)
    (run_dir / "fit_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (run_dir / "fit_plot.html").write_text(fig.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")

    png_path = run_dir / "fit_plot.png"
    png_bytes = _try_plot_png_bytes(fig)
    if png_bytes is not None:
        png_path.write_bytes(png_bytes)
    else:
        png_path = None

    npz_data = {
        "x": np.asarray(fit_state.get("x", []), dtype=float),
        "y_raw": np.asarray(fit_state.get("y", []), dtype=float),
        "y_fit_sum": np.asarray(fit_state.get("y_fit", []), dtype=float),
    }
    for i, comp in enumerate(fit_state.get("components", []), start=1):
        npz_data[f"peak_{i}"] = np.asarray(comp, dtype=float)
    npz_path = run_dir / "fit_arrays.npz"
    np.savez_compressed(npz_path, **npz_data)

    zip_path = run_dir / "fit_results.zip"
    zip_path.write_bytes(_build_fit_zip_bytes(fit_state))

    return {
        "run_dir": run_dir,
        "zip_path": zip_path,
        "npz_path": npz_path,
        "png_path": png_path,
    }


def _save_batch_fit_to_server(fit_states_by_curve, summary_rows):
    """Persist batch fit results to server files and return saved paths."""
    export_root = _get_export_root_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = export_root / f"{ts}_batch_{len(fit_states_by_curve)}curves"
    run_dir.mkdir(parents=True, exist_ok=True)

    per_curve_dirs = []
    for idx, (curve_key, fit_state) in enumerate(fit_states_by_curve.items(), start=1):
        curve_df, params_df, metrics_df, peaks_df, metadata = _fit_state_to_tables(fit_state)
        fig = _create_fit_plot_figure(fit_state)
        curve_name = fit_state.get("curve_label") or curve_key
        curve_dir = run_dir / f"{idx:03d}_{_sanitize_filename(curve_name)}"
        curve_dir.mkdir(parents=True, exist_ok=True)

        curve_df.to_csv(curve_dir / "fitted_curve.csv", index=False)
        params_df.to_csv(curve_dir / "fit_parameters.csv", index=False)
        metrics_df.to_csv(curve_dir / "fit_metrics.csv", index=False)
        peaks_df.to_csv(curve_dir / "fit_peaks.csv", index=False)
        (curve_dir / "fit_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (curve_dir / "fit_plot.html").write_text(fig.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")

        png_bytes = _try_plot_png_bytes(fig)
        if png_bytes is not None:
            (curve_dir / "fit_plot.png").write_bytes(png_bytes)
        else:
            (curve_dir / "fit_plot_note.txt").write_text(
                "PNG export unavailable (install kaleido). HTML plot is included.",
                encoding="utf-8",
            )

        npz_data = {
            "x": np.asarray(fit_state.get("x", []), dtype=float),
            "y_raw": np.asarray(fit_state.get("y", []), dtype=float),
            "y_fit_sum": np.asarray(fit_state.get("y_fit", []), dtype=float),
        }
        for i, comp in enumerate(fit_state.get("components", []), start=1):
            npz_data[f"peak_{i}"] = np.asarray(comp, dtype=float)
        np.savez_compressed(curve_dir / "fit_arrays.npz", **npz_data)
        per_curve_dirs.append(curve_dir)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = run_dir / "batch_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    zip_path = run_dir / "batch_fit_results.zip"
    zip_path.write_bytes(_build_batch_fit_zip_bytes(fit_states_by_curve))

    return {
        "run_dir": run_dir,
        "zip_path": zip_path,
        "summary_path": summary_path,
        "curve_dirs": per_curve_dirs,
    }


# ---------------------------------------------------------------------------
# Initialize session state
# ---------------------------------------------------------------------------

if 'curve_styles' not in st.session_state:
    st.session_state['curve_styles'] = {}

if 'dataframes_csv' not in st.session_state:
    st.session_state['dataframes_csv'] = {}

if 'file_paths_csv' not in st.session_state:
    st.session_state['file_paths_csv'] = {}

if 'sim_curve_counter' not in st.session_state:
    st.session_state['sim_curve_counter'] = 0

if 'simulated_curve_meta' not in st.session_state:
    st.session_state['simulated_curve_meta'] = {}

if 'fit_peak_guesses' not in st.session_state:
    st.session_state['fit_peak_guesses'] = {}

if 'fit_results_csv' not in st.session_state:
    st.session_state['fit_results_csv'] = {}

if 'fit_batch_last_summary' not in st.session_state:
    st.session_state['fit_batch_last_summary'] = []

if 'fit_last_single_setup' not in st.session_state:
    st.session_state['fit_last_single_setup'] = {}

if 'fit_last_single_setup_saxs' not in st.session_state:
    st.session_state['fit_last_single_setup_saxs'] = {}

if 'fit_active_backend' not in st.session_state:
    st.session_state['fit_active_backend'] = FIT_BACKENDS["General Peaks"]

if 'fit_ml_predictions' not in st.session_state:
    st.session_state['fit_ml_predictions'] = {}

# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

st.title("📊 Enhanced CSV/NPZ Plotter")
st.markdown("Quick visualization with per-curve styling - supports CSV, TXT, DAT, NPZ")

# Floating sidebar toggle button (bottom-left)
floating_sidebar_toggle()

# ---------------------------------------------------------------------------
# Sidebar: Data Source
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📁 Data Source")

    data_source = st.radio(
        "Data location",
        ["Upload files", "Browse server"],
        help="Upload files from your computer or browse server filesystem"
    )

    if data_source == "Upload files":
        uploaded_files = st.file_uploader(
            "Upload data files",
            type=['csv', 'txt', 'dat', 'npz'],
            accept_multiple_files=True,
            help="Upload CSV, TXT, DAT, or NPZ files"
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    # Save temporarily to load
                    temp_path = Path(f"/tmp/{uploaded_file.name}")
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())

                    df = load_data_file(str(temp_path))
                    if df is not None:
                        st.session_state['dataframes_csv'][uploaded_file.name] = df
                        st.session_state['file_paths_csv'][uploaded_file.name] = uploaded_file.name
                except Exception as e:
                    st.error(f"Error loading {uploaded_file.name}: {e}")

    else:  # Browse server
        st.markdown("**🗂️ Interactive Folder Browser**")
        st.markdown("Click folders to navigate, select files with checkboxes:")

        # File pattern selector
        st.markdown("**📋 File Type Filter:**")
        pattern = st.selectbox(
            "Extension pattern",
            ["*.*", "*.csv", "*.npz", "*.txt", "*.dat"],
            help="Filter files by extension",
            label_visibility="collapsed"
        )

        st.info("💡 Tip: Use '🔍 Advanced Filters' below for name-based filtering (contains, not contains, etc.)")

        # Use folder browser component
        selected_files = folder_browser(
            key="csv_plotter_browser",
            show_files=True,
            file_pattern=pattern,
            multi_select=True,
            initial_path=_start_dir if _user_mode else None,
            restrict_to_start_dir=_user_mode,
        )

        # Load button
        if selected_files and st.button("📥 Load Selected Files", key="csv_load_btn"):
            for full_path in selected_files:
                df = load_data_file(full_path)
                if df is not None:
                    file_name = Path(full_path).name
                    st.session_state['dataframes_csv'][file_name] = df
                    st.session_state['file_paths_csv'][file_name] = full_path
                    st.success(f"✅ Loaded {file_name}")

    # Synthetic curve generator for fitting workflows
    with st.expander("🧪 Simulate 1D Peak Curve", expanded=False):
        st.caption("Generate one/two/multi-peak synthetic data for fitting tests.")
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            sim_shape_label = st.selectbox(
                "Peak shape",
                list(FIT_SHAPES.keys()),
                index=0,
                key="sim_peak_shape",
            )
            sim_n_peaks = st.number_input(
                "Number of peaks",
                min_value=1,
                max_value=8,
                value=2,
                step=1,
                key="sim_peak_count",
            )
            sim_points = st.number_input(
                "Data points",
                min_value=100,
                max_value=5000,
                value=1200,
                step=100,
                key="sim_points",
            )
        with sim_col2:
            sim_x_min = st.number_input("X min", value=0.0, format="%.4f", key="sim_x_min")
            sim_x_max = st.number_input("X max", value=100.0, format="%.4f", key="sim_x_max")
            sim_baseline = st.number_input("Baseline c", value=0.1, format="%.4f", key="sim_baseline")
            sim_noise = st.slider("Noise level", 0.0, 0.20, 0.02, 0.01, key="sim_noise")
        sim_seed = st.number_input("Random seed", min_value=0, value=1234, step=1, key="sim_seed")

        if st.button("➕ Generate Simulated Curve", key="generate_sim_curve"):
            if sim_x_max <= sim_x_min:
                st.error("X max must be larger than X min.")
            else:
                shape_key = FIT_SHAPES[sim_shape_label]
                sim_df, sim_params = _simulate_peak_curve(
                    n_peaks=int(sim_n_peaks),
                    shape=shape_key,
                    x_min=float(sim_x_min),
                    x_max=float(sim_x_max),
                    n_points=int(sim_points),
                    baseline=float(sim_baseline),
                    noise_level=float(sim_noise),
                    seed=int(sim_seed),
                )
                st.session_state['sim_curve_counter'] += 1
                sim_name = (
                    f"sim_{shape_key}_{int(sim_n_peaks)}peaks_"
                    f"{st.session_state['sim_curve_counter']:03d}.csv"
                )
                st.session_state['dataframes_csv'][sim_name] = sim_df
                st.session_state['file_paths_csv'][sim_name] = f"[simulated] {sim_name}"
                st.session_state['simulated_curve_meta'][sim_name] = {
                    "shape": shape_key,
                    "params": sim_params,
                    "baseline": float(sim_baseline),
                    "noise_level": float(sim_noise),
                    "seed": int(sim_seed),
                }
                st.success(f"Generated {sim_name} with {int(sim_n_peaks)} {sim_shape_label} peak(s).")
                st.rerun()

    # Get dataframes from session state
    dataframes = st.session_state['dataframes_csv']
    file_paths = st.session_state['file_paths_csv']

    # Clear button
    if dataframes:
        if st.button("🗑️ Clear All Data", key="clear_csv_data"):
            st.session_state['dataframes_csv'] = {}
            st.session_state['file_paths_csv'] = {}
            st.session_state['curve_styles'] = {}
            st.session_state['simulated_curve_meta'] = {}
            st.session_state['fit_peak_guesses'] = {}
            st.session_state['fit_results_csv'] = {}
            st.session_state['fit_batch_last_summary'] = []
            st.session_state['fit_last_single_setup'] = {}
            st.session_state['fit_last_single_setup_saxs'] = {}
            st.session_state['fit_active_backend'] = FIT_BACKENDS["General Peaks"]
            st.session_state['fit_ml_predictions'] = {}
            st.rerun()

    if not dataframes:
        st.info("👆 Upload or select files to get started")
        st.stop()

    st.success(f"✅ Loaded {len(dataframes)} file(s)")

    # ---------------------------------------------------------------------------
    # Column Selection
    # ---------------------------------------------------------------------------

    st.header("📐 Column Selection")

    # Get all unique column names
    all_columns = set()
    for df in dataframes.values():
        all_columns.update(df.columns)
    all_columns = sorted(list(all_columns))

    if not all_columns:
        st.error("No columns found")
        st.stop()

    # Auto-detect likely X column
    x_default = None
    for col in all_columns:
        col_lower = col.lower()
        if x_default is None and any(kw in col_lower for kw in
                                     ['wavelength', 'q', 'theta', 'energy', 'time', 'x']):
            x_default = col

    if x_default is None and len(all_columns) > 0:
        x_default = all_columns[0]

    x_col = st.selectbox(
        "X-axis column (common for all files)",
        all_columns,
        index=all_columns.index(x_default) if x_default in all_columns else 0,
        help="Select X-axis column - will be used for all files"
    )

    # Per-file Y column selection
    st.markdown("**Y-axis columns (per file):**")
    st.info("💡 Select multiple Y columns from each file to plot them as separate curves")

    # Store Y column selections in session state
    if 'y_column_selections' not in st.session_state:
        st.session_state['y_column_selections'] = {}

    file_y_columns = {}  # {file_name: [y_col1, y_col2, ...]}

    for file_name, df in dataframes.items():
        available_y_cols = [col for col in df.columns if col != x_col]

        if not available_y_cols:
            st.warning(f"⚠️ {file_name}: No Y columns available (only X column found)")
            continue

        # Auto-detect Y columns for first time
        if file_name not in st.session_state['y_column_selections']:
            # Default: select columns with keywords
            default_y = []
            for col in available_y_cols:
                col_lower = col.lower()
                if any(kw in col_lower for kw in ['intensity', 'absorbance', 'absorption', 'y', 'signal']):
                    default_y.append(col)

            # If no keywords match, select first available column
            if not default_y and available_y_cols:
                default_y = [available_y_cols[0]]

            st.session_state['y_column_selections'][file_name] = default_y

        selected_y_cols = st.multiselect(
            f"📄 {shorten_path(file_name, 40)}",
            available_y_cols,
            default=st.session_state['y_column_selections'][file_name],
            key=f"y_cols_{file_name}",
            help=f"Select one or more Y columns from {file_name}"
        )

        st.session_state['y_column_selections'][file_name] = selected_y_cols
        file_y_columns[file_name] = selected_y_cols

    # Check if any Y columns selected
    total_curves = sum(len(cols) for cols in file_y_columns.values())
    if total_curves == 0:
        st.error("No Y columns selected. Please select at least one Y column from any file.")
        st.stop()

    st.success(f"✅ Total curves to plot: {total_curves}")

    # ---------------------------------------------------------------------------
    # Global Plot Controls
    # ---------------------------------------------------------------------------

    st.header("⚙️ Global Controls")

    # Plot mode
    plot_mode = st.radio(
        "Plot mode",
        ["Interactive (Plotly)", "Static (Matplotlib)"],
        horizontal=True,
        help="Plotly shows values on hover! Matplotlib is static but publication-ready."
    )
    use_plotly = plot_mode.startswith("Interactive")

    # Scale
    col1, col2 = st.columns(2)
    with col1:
        x_scale = st.radio("X Scale", ["linear", "log"], horizontal=True)
    with col2:
        y_scale = st.radio("Y Scale", ["linear", "log"], horizontal=True)

    # Axis limits
    with st.expander("📏 Axis Limits", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**X-axis limits:**")
            use_xlim = st.checkbox("Set X limits", value=False)
            if use_xlim:
                xlim_min = st.number_input("X min", value=0.0, format="%.4f")
                xlim_max = st.number_input("X max", value=100.0, format="%.4f")
            else:
                xlim_min, xlim_max = None, None

        with col2:
            st.markdown("**Y-axis limits:**")
            use_ylim = st.checkbox("Set Y limits", value=False)
            if use_ylim:
                ylim_min = st.number_input("Y min", value=0.0, format="%.4f")
                ylim_max = st.number_input("Y max", value=100.0, format="%.4f")
            else:
                ylim_min, ylim_max = None, None

    # Global style
    with st.expander("🎨 Global Style", expanded=False):
        show_grid = st.checkbox("Show grid", value=True)
        show_legend = st.checkbox("Show legend", value=True)

    # Figure size
    with st.expander("📐 Figure Size", expanded=False):
        use_custom_size = st.checkbox("Custom figure size", value=False,
                                       help="Default: auto width, 600px height")
        if use_custom_size:
            sc1, sc2 = st.columns(2)
            with sc1:
                fig_width = st.number_input("Width (px)", min_value=300, max_value=3000,
                                            value=1000, step=50)
            with sc2:
                fig_height = st.number_input("Height (px)", min_value=200, max_value=2000,
                                             value=600, step=50)
        else:
            fig_width = None
            fig_height = 600

    # Labels
    with st.expander("📝 Labels", expanded=False):
        plot_title = st.text_input("Plot title", value="Data Comparison")
        x_label = st.text_input("X-axis label", value=x_col)
        y_label = st.text_input("Y-axis label", value="Intensity")

# ---------------------------------------------------------------------------
# Main Area: Per-Curve Styling
# ---------------------------------------------------------------------------

st.header("🎨 Per-Curve Styling")

# Create expanders for each curve (file_name, y_col combination)
curve_settings = {}
curve_idx = 0

for file_name, y_cols in file_y_columns.items():
    for y_col in y_cols:
        curve_key = f"{file_name}::{y_col}"  # Unique key for each curve
        curve_label = f"{shorten_path(file_name, 30)} : {y_col}"

        with st.expander(f"🔧 {curve_label}", expanded=False):
            # Initialize defaults if not in session state
            if curve_key not in st.session_state['curve_styles']:
                st.session_state['curve_styles'][curve_key] = {
                    'color': list(COLORS_NAMED.keys())[curve_idx % len(COLORS_NAMED)],
                    'marker': list(MARKERS_DICT.keys())[curve_idx % len(MARKERS_DICT)],
                    'linestyle': 'Solid',
                    'linewidth': 2.0,
                    'markersize': 8.0,
                    'alpha': 0.8,
                    'enabled': True
                }

            # Initialize curve_settings entry
            if curve_key not in curve_settings:
                curve_settings[curve_key] = {}

            # Enable/Disable checkbox
            enabled = st.checkbox(
                f"✅ Show this curve",
                value=st.session_state['curve_styles'][curve_key].get('enabled', True),
                key=f"enabled_{curve_key}",
                help="Toggle to show/hide this curve in the plot"
            )
            st.session_state['curve_styles'][curve_key]['enabled'] = enabled
            curve_settings[curve_key]['enabled'] = enabled

            if not enabled:
                st.info("⚠️ This curve is hidden. Enable to show in plot.")
                curve_idx += 1
                continue

            st.divider()
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                color_name = st.selectbox(
                    "Color",
                    list(COLORS_NAMED.keys()),
                    index=list(COLORS_NAMED.keys()).index(
                        st.session_state['curve_styles'][curve_key]['color']
                    ),
                    key=f"color_{curve_key}"
                )
                curve_settings[curve_key]['color'] = COLORS_NAMED[color_name]

            with col2:
                marker_name = st.selectbox(
                    "Marker",
                    list(MARKERS_DICT.keys()),
                    index=list(MARKERS_DICT.keys()).index(
                        st.session_state['curve_styles'][curve_key]['marker']
                    ),
                    key=f"marker_{curve_key}"
                )
                curve_settings[curve_key]['marker'] = MARKERS_DICT[marker_name]

            with col3:
                linestyle_name = st.selectbox(
                    "Line Style",
                    list(LINESTYLES_DICT.keys()),
                    index=list(LINESTYLES_DICT.keys()).index(
                        st.session_state['curve_styles'][curve_key]['linestyle']
                    ),
                    key=f"linestyle_{curve_key}"
                )
                curve_settings[curve_key]['linestyle'] = LINESTYLES_DICT[linestyle_name]

            with col4:
                linewidth = st.slider(
                    "Line Width",
                    0.5, 5.0,
                    st.session_state['curve_styles'][curve_key]['linewidth'],
                    0.5,
                    key=f"linewidth_{curve_key}"
                )
                curve_settings[curve_key]['linewidth'] = linewidth

            with col5:
                markersize = st.slider(
                    "Marker Size",
                    1.0, 20.0,
                    st.session_state['curve_styles'][curve_key].get('markersize', 8.0),
                    1.0,
                    key=f"markersize_{curve_key}"
                )
                curve_settings[curve_key]['markersize'] = markersize

            with col6:
                alpha = st.slider(
                    "Opacity",
                    0.1, 1.0,
                    st.session_state['curve_styles'][curve_key]['alpha'],
                    0.1,
                    key=f"alpha_{curve_key}"
                )
                curve_settings[curve_key]['alpha'] = alpha

            # Update session state
            st.session_state['curve_styles'][curve_key].update({
                'color': color_name,
                'marker': marker_name,
                'linestyle': linestyle_name,
                'linewidth': linewidth,
                'markersize': markersize,
                'alpha': alpha
            })

        curve_idx += 1

# Curves eligible for 1D fitting (one at a time)
fit_curve_options = []
for fit_file_name, fit_y_cols in file_y_columns.items():
    df = dataframes.get(fit_file_name)
    if df is None:
        continue
    for fit_y_name in fit_y_cols:
        if x_col not in df.columns or fit_y_name not in df.columns:
            continue
        curve_key = f"{fit_file_name}::{fit_y_name}"
        fit_curve_options.append({
            "label": f"{shorten_path(fit_file_name, 30)} : {fit_y_name}",
            "curve_key": curve_key,
            "file_name": fit_file_name,
            "y_col": fit_y_name,
        })

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

st.divider()
st.header(f"📈 Plot: Y columns vs {x_col}")

# Debug: Show what will be plotted
with st.expander("🔍 Debug: Curves to Plot", expanded=False):
    for fname, ycols in file_y_columns.items():
        st.write(f"**{fname}**: {len(ycols)} columns → {', '.join(ycols)}")

if use_plotly:
    # -------------------------------------------------------------------------
    # Plotly Interactive Plot (with hover!)
    # -------------------------------------------------------------------------
    fig = go.Figure()

    plotted_count = 0
    skipped_count = 0
    for file_name, y_cols in file_y_columns.items():
        df = dataframes[file_name]

        for y_col in y_cols:
            curve_key = f"{file_name}::{y_col}"

            # Check if enabled
            if not curve_settings.get(curve_key, {}).get('enabled', True):
                skipped_count += 1
                continue

            # Check if columns exist
            if x_col not in df.columns or y_col not in df.columns:
                st.warning(f"⚠️ {file_name}, {y_col}: missing columns")
                continue

            # Get data
            x_data = df[x_col].values
            y_data = df[y_col].values

            # Remove NaN
            mask = ~(np.isnan(x_data) | np.isnan(y_data))
            x_data = x_data[mask]
            y_data = y_data[mask]

            if len(x_data) == 0:
                st.warning(f"⚠️ {file_name}, {y_col}: no valid data")
                continue

                # Get styling
            style = curve_settings.get(curve_key, {})

            # Map matplotlib markers to plotly symbols
            marker_map = {
                'o': 'circle', 's': 'square', '^': 'triangle-up', 'v': 'triangle-down',
                'D': 'diamond', 'p': 'pentagon', '*': 'star', 'h': 'hexagon',
                '+': 'cross', 'x': 'x', '.': 'circle', 'None': None
            }

            # Map matplotlib linestyles to plotly dash
            linestyle_map = {
                '-': 'solid', '--': 'dash', '-.': 'dashdot', ':': 'dot', 'None': None
            }

            marker_style = marker_map.get(style.get('marker', 'o'), 'circle')
            line_dash = linestyle_map.get(style.get('linestyle', '-'), 'solid')

            # Determine mode
            if marker_style and line_dash:
                mode = 'lines+markers'
            elif marker_style:
                mode = 'markers'
            elif line_dash:
                mode = 'lines'
            else:
                mode = 'lines'

            # Add trace
            trace_name = f"{shorten_path(file_name, 20)}: {y_col}"
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode=mode,
                name=trace_name,
                line=dict(
                    color=style.get('color', '#1f77b4'),
                    width=style.get('linewidth', 2.0),
                    dash=line_dash if line_dash else 'solid'
                ),
                marker=dict(
                    symbol=marker_style if marker_style else 'circle',
                    size=style.get('markersize', 8.0),
                    color=style.get('color', '#1f77b4')
                ),
                opacity=style.get('alpha', 0.8),
                hovertemplate=f'<b>{x_col}</b>: %{{x:.4f}}<br><b>{y_col}</b>: %{{y:.4f}}<extra></extra>'
            ))

            plotted_count += 1

    if plotted_count == 0:
        st.error("No curves enabled. Enable curves in Per-Curve Styling section.")
        st.stop()

    # Apply settings
    fig.update_xaxes(
        title=x_label,
        type='log' if x_scale == 'log' else 'linear',
        range=[np.log10(xlim_min) if x_scale == 'log' and use_xlim and xlim_min > 0 else xlim_min,
               np.log10(xlim_max) if x_scale == 'log' and use_xlim else xlim_max] if use_xlim else None,
        showgrid=show_grid,
        gridcolor='lightgray',
        gridwidth=0.5
    )

    fig.update_yaxes(
        title=y_label,
        type='log' if y_scale == 'log' else 'linear',
        range=[np.log10(ylim_min) if y_scale == 'log' and use_ylim and ylim_min > 0 else ylim_min,
               np.log10(ylim_max) if y_scale == 'log' and use_ylim else ylim_max] if use_ylim else None,
        showgrid=show_grid,
        gridcolor='lightgray',
        gridwidth=0.5
    )

    layout_kwargs = dict(
        title=plot_title,
        height=fig_height,
        hovermode='closest',
        showlegend=show_legend,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    if fig_width is not None:
        layout_kwargs['width'] = fig_width
    fig.update_layout(**layout_kwargs)

    # Show plot
    st.plotly_chart(fig, use_container_width=(fig_width is None))
    if skipped_count > 0:
        st.success(f"✅ {plotted_count} curve(s) plotted. **Hover over curves to see (x,y) values!** ({skipped_count} disabled)")
    else:
        st.success(f"✅ {plotted_count} curve(s) plotted. **Hover over curves to see (x,y) values!**")

else:
    # -------------------------------------------------------------------------
    # Matplotlib Static Plot
    # -------------------------------------------------------------------------
    if fig_width is not None:
        fig, ax = plt.subplots(figsize=(fig_width / 100, fig_height / 100))
    else:
        fig, ax = plt.subplots(figsize=(12, fig_height / 100))

    plotted_count = 0
    skipped_count = 0
    for file_name, y_cols in file_y_columns.items():
        df = dataframes[file_name]

        for y_col in y_cols:
            curve_key = f"{file_name}::{y_col}"

            # Check if enabled
            if not curve_settings.get(curve_key, {}).get('enabled', True):
                skipped_count += 1
                continue

            # Check if columns exist
            if x_col not in df.columns or y_col not in df.columns:
                st.warning(f"⚠️ {file_name}, {y_col}: missing columns")
                continue

            # Get data
            x_data = df[x_col].values
            y_data = df[y_col].values

            # Remove NaN
            mask = ~(np.isnan(x_data) | np.isnan(y_data))
            x_data = x_data[mask]
            y_data = y_data[mask]

            if len(x_data) == 0:
                st.warning(f"⚠️ {file_name}, {y_col}: no valid data")
                continue

            # Get styling
            style = curve_settings.get(curve_key, {})

            # Plot
            marker = style.get('marker', 'None')
            marker = None if marker == 'None' else marker

            linestyle = style.get('linestyle', '-')
            linestyle = '' if linestyle == 'None' else linestyle

            trace_label = f"{shorten_path(file_name, 20)}: {y_col}"

            ax.plot(
                x_data, y_data,
                color=style.get('color', '#1f77b4'),
                marker=marker,
                linestyle=linestyle,
                linewidth=style.get('linewidth', 2.0),
                alpha=style.get('alpha', 0.8),
                label=trace_label,
                markersize=style.get('markersize', 8.0),
                markevery=max(1, len(x_data)//20) if marker else None
            )

            plotted_count += 1

    if plotted_count == 0:
        st.error("No curves enabled. Enable curves in Per-Curve Styling section.")
        st.stop()

    # Apply settings
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(plot_title, fontsize=14, fontweight='bold')

    # Set axis limits
    if use_xlim:
        ax.set_xlim(xlim_min, xlim_max)
    if use_ylim:
        ax.set_ylim(ylim_min, ylim_max)

    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--')

    if show_legend and len(dataframes) > 1:
        ax.legend(loc='best', framealpha=0.9, fontsize=9)

    # Show plot
    st.pyplot(fig)
    if skipped_count > 0:
        st.info(f"✅ {plotted_count} curve(s) plotted. Switch to 'Interactive (Plotly)' mode to see hover values! ({skipped_count} disabled)")
    else:
        st.info(f"✅ {plotted_count} curve(s) plotted. Switch to 'Interactive (Plotly)' mode to see hover values!")

# ---------------------------------------------------------------------------
# 1D Peak Fitting (pyFitting)
# ---------------------------------------------------------------------------

st.divider()
st.header("🔬 1D Peak Fitting (pyFitting)")
st.caption("Choose a fitting backend for the selected curve: General peaks, SAXS physics, or ML preview.")

if not PYFITTING_AVAILABLE:
    st.warning(
        "pyFitting is not available in this environment. "
        f"Import error: {PYFITTING_IMPORT_ERROR or 'unknown'}"
    )
elif not fit_curve_options:
    st.info("No eligible curves available for fitting. Select at least one Y column first.")
else:
    fit_labels = [item["label"] for item in fit_curve_options]
    fit_option_by_label = {item["label"]: item for item in fit_curve_options}
    fit_backend_label = st.radio(
        "Fitting backend",
        list(FIT_BACKENDS.keys()),
        horizontal=True,
        key="fit_backend_selector",
    )
    fit_backend = FIT_BACKENDS[fit_backend_label]
    if st.session_state.get("fit_active_backend") != fit_backend:
        st.session_state["fit_active_backend"] = fit_backend
        st.session_state["fit_batch_last_summary"] = []
    if fit_backend == FIT_BACKENDS["ML-Assisted (Preview)"]:
        st.info(
            "ML-assisted fitting scaffold is enabled below: "
            "select an ML model, preprocess, predict seeds, then optionally refine."
        )

    saxs_label_to_key, saxs_key_to_label = get_saxs_shape_options()
    if not saxs_label_to_key:
        saxs_label_to_key = dict(SAXS_MODEL_LABEL_TO_KEY)
        saxs_key_to_label = dict(SAXS_MODEL_KEY_TO_LABEL)

    fit_selected_label = st.selectbox(
        "Curve to fit",
        fit_labels,
        key="fit_curve_selector",
        help="v1 fits one selected curve at a time.",
    )
    fit_selected = fit_option_by_label[fit_selected_label]
    fit_curve_key = fit_selected["curve_key"]
    fit_file_name = fit_selected["file_name"]
    fit_y_col = fit_selected["y_col"]
    fit_df = dataframes[fit_file_name]

    x_full, y_full = _prepare_xy(fit_df[x_col].values, fit_df[fit_y_col].values)
    if len(x_full) < 5:
        st.error("Selected curve has too few valid points for fitting.")
    elif fit_backend == FIT_BACKENDS["General Peaks"]:
        fit_use_range = False
        x_range_min, x_range_max = None, None
        current_peaks = []
        x_fit, y_fit_data = x_full, y_full
        single_fit_state = st.session_state['fit_results_csv'].get(fit_curve_key)
        if single_fit_state and single_fit_state.get("backend") != FIT_BACKENDS["General Peaks"]:
            single_fit_state = None

        with st.expander("🎯 Single-Curve Fitting", expanded=True):
            control_col1, control_col2, control_col3 = st.columns([1.2, 1.0, 1.0])
            with control_col1:
                fit_shape_label = st.selectbox(
                    "Fit shape",
                    list(FIT_SHAPES.keys()),
                    index=0,
                    key=_safe_widget_key("fit_shape", fit_curve_key),
                )
                fit_shape = FIT_SHAPES[fit_shape_label]
            with control_col2:
                fit_maxiter = st.number_input(
                    "Max iterations",
                    min_value=100,
                    max_value=10000,
                    value=2000,
                    step=100,
                    key=_safe_widget_key("fit_maxiter", fit_curve_key),
                )
            with control_col3:
                fit_use_range = st.checkbox(
                    "Use x-range",
                    value=False,
                    key=_safe_widget_key("fit_use_range", fit_curve_key),
                )

            if fit_use_range:
                x_min_full = float(np.min(x_full))
                x_max_full = float(np.max(x_full))
                if x_max_full > x_min_full:
                    x_range_min, x_range_max = st.slider(
                        "Fit x-range",
                        min_value=x_min_full,
                        max_value=x_max_full,
                        value=(x_min_full, x_max_full),
                        key=_safe_widget_key("fit_range_slider", fit_curve_key),
                    )
                else:
                    st.warning("X range is degenerate; fitting full range instead.")

            x_fit, y_fit_data = _prepare_xy(
                fit_df[x_col].values,
                fit_df[fit_y_col].values,
                x_min=x_range_min,
                x_max=x_range_max,
            )
            if len(x_fit) < 5:
                st.error("Not enough points remain after range filtering.")
            else:
                peak_tol = max(float(np.ptp(x_fit)) / 5000.0, 1e-9)
                picked_peaks = st.session_state['fit_peak_guesses'].setdefault(fit_curve_key, [])
                picked_peaks[:] = [p for p in picked_peaks if float(np.min(x_fit)) <= p <= float(np.max(x_fit))]
                picked_peaks[:] = _dedupe_sorted(picked_peaks, peak_tol)

                existing_fit_state = st.session_state['fit_results_csv'].get(fit_curve_key)

                picker_fig = go.Figure()
                picker_fig.add_trace(
                    go.Scatter(
                        x=x_fit,
                        y=y_fit_data,
                        mode='lines+markers',
                        name='Data',
                        line=dict(color='#1f77b4', width=1.5),
                        marker=dict(size=5, color='#1f77b4'),
                        hovertemplate=f'<b>{x_col}</b>: %{{x:.5g}}<br><b>{fit_y_col}</b>: %{{y:.5g}}<extra></extra>',
                    )
                )

                for peak_x in picked_peaks:
                    picker_fig.add_vline(
                        x=float(peak_x),
                        line_width=1.2,
                        line_dash='dot',
                        line_color='#d62728',
                    )

                if existing_fit_state and existing_fit_state.get("success", False):
                    picker_fig.add_trace(
                        go.Scatter(
                            x=existing_fit_state["x"],
                            y=existing_fit_state["y_fit"],
                            mode='lines',
                            name='Latest fit',
                            line=dict(color='#2ca02c', width=2.0),
                        )
                    )

                picker_fig.update_layout(
                    title=f"Peak Picker: {fit_selected_label}",
                    height=420,
                    showlegend=True,
                    dragmode='select',
                    clickmode='event+select',
                )
                picker_fig.update_xaxes(title=x_col)
                picker_fig.update_yaxes(title=fit_y_col)

                selection_state = st.plotly_chart(
                    picker_fig,
                    use_container_width=True,
                    key=_safe_widget_key("fit_picker", fit_curve_key),
                    on_select="rerun",
                    selection_mode="points",
                )

                selected_points = []
                try:
                    selected_points = list(selection_state.selection.points)
                except Exception:
                    selected_points = []
                selected_peak_x = _dedupe_sorted(
                    [
                        float(p["x"])
                        for p in selected_points
                        if "x" in p and int(p.get("curve_number", 0)) == 0
                    ],
                    peak_tol,
                )

                current_peaks = _dedupe_sorted(st.session_state['fit_peak_guesses'].get(fit_curve_key, []), peak_tol)
                auto_add_click = st.checkbox(
                    "Quick add from selected point(s)",
                    value=False,
                    help="Enable this to auto-add selected points as peak guesses.",
                    key=_safe_widget_key("fit_auto_add_click", fit_curve_key),
                )
                if auto_add_click and selected_peak_x:
                    merged = _dedupe_sorted(current_peaks + selected_peak_x, peak_tol)
                    if len(merged) > len(current_peaks):
                        st.session_state['fit_peak_guesses'][fit_curve_key] = merged
                        st.rerun()

                if selected_peak_x:
                    st.info(
                        "Selected peak x-values: "
                        + ", ".join(f"{value:.6g}" for value in selected_peak_x)
                    )
                else:
                    st.caption("Tip: select one or more points near peak centers, then add them.")

                manual_peak_x = st.number_input(
                    "Manual peak x",
                    value=float(np.median(x_fit)),
                    format="%.6f",
                    key=_safe_widget_key("manual_peak_x", fit_curve_key),
                )

                action_col1, action_col2, action_col3, action_col4 = st.columns(4)
                with action_col1:
                    if st.button("➕ Add Selected", key=_safe_widget_key("fit_add_selected", fit_curve_key)):
                        if selected_peak_x:
                            picked_peaks.extend(selected_peak_x)
                            st.session_state['fit_peak_guesses'][fit_curve_key] = _dedupe_sorted(
                                picked_peaks, peak_tol
                            )
                            st.rerun()
                        else:
                            st.warning("No selected points to add.")
                with action_col2:
                    if st.button("➕ Add Manual", key=_safe_widget_key("fit_add_manual", fit_curve_key)):
                        picked_peaks.append(float(manual_peak_x))
                        st.session_state['fit_peak_guesses'][fit_curve_key] = _dedupe_sorted(
                            picked_peaks, peak_tol
                        )
                        st.rerun()
                with action_col3:
                    if st.button("↩️ Undo Last", key=_safe_widget_key("fit_undo_peak", fit_curve_key)):
                        if picked_peaks:
                            picked_peaks.pop()
                            st.session_state['fit_peak_guesses'][fit_curve_key] = picked_peaks
                            st.rerun()
                with action_col4:
                    if st.button("🧹 Clear Peaks", key=_safe_widget_key("fit_clear_peaks", fit_curve_key)):
                        st.session_state['fit_peak_guesses'][fit_curve_key] = []
                        st.rerun()

                current_peaks = _dedupe_sorted(st.session_state['fit_peak_guesses'].get(fit_curve_key, []), peak_tol)
                if current_peaks:
                    st.dataframe(
                        pd.DataFrame({
                            "peak_index": np.arange(1, len(current_peaks) + 1),
                            "x_guess": current_peaks,
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )

                    remove_options = [
                        f"#{idx + 1} @ {value:.6g}" for idx, value in enumerate(current_peaks)
                    ]
                    peaks_to_remove = st.multiselect(
                        "Remove selected peak(s)",
                        remove_options,
                        key=_safe_widget_key("fit_remove_choices", fit_curve_key),
                    )
                    if st.button("🗑️ Remove Selected Peak(s)", key=_safe_widget_key("fit_remove_selected", fit_curve_key)):
                        remove_indices = {
                            remove_options.index(item) for item in peaks_to_remove if item in remove_options
                        }
                        if remove_indices:
                            remaining = [
                                peak for idx, peak in enumerate(current_peaks)
                                if idx not in remove_indices
                            ]
                            st.session_state['fit_peak_guesses'][fit_curve_key] = remaining
                            st.rerun()
                else:
                    st.warning("No peak guesses yet. Add at least one peak before fitting.")

                run_fit = st.button(
                    f"🚀 Fit {len(current_peaks)} Peak(s)",
                    key=_safe_widget_key("fit_run_button", fit_curve_key),
                    disabled=len(current_peaks) == 0,
                )

                if run_fit:
                    st.session_state['fit_last_single_setup'] = {
                        "shape": fit_shape,
                        "shape_label": fit_shape_label,
                        "n_peaks": len(current_peaks),
                        "maxiter": int(fit_maxiter),
                        "use_range": bool(fit_use_range),
                        "x_range": (
                            float(x_range_min),
                            float(x_range_max),
                        ) if fit_use_range and x_range_min is not None and x_range_max is not None else None,
                        "curve_key": fit_curve_key,
                        "curve_label": fit_selected_label,
                    }
                    try:
                        fit_state = _run_multipeak_fit(
                            x_fit,
                            y_fit_data,
                            shape=fit_shape,
                            shape_label=fit_shape_label,
                            n_peaks=len(current_peaks),
                            maxiter=int(fit_maxiter),
                            peak_guesses=current_peaks,
                            x_col_name=x_col,
                            y_col_name=fit_y_col,
                        )
                        fit_state.update({
                            "source_file": fit_file_name,
                            "source_curve_key": fit_curve_key,
                            "curve_label": fit_selected_label,
                        })
                        st.session_state['fit_results_csv'][fit_curve_key] = fit_state
                        single_fit_state = fit_state
                        if fit_state.get("success"):
                            st.success("Fit completed.")
                        else:
                            st.warning(f"Fit finished with warning: {fit_state.get('message')}")
                    except Exception as fit_error:
                        st.error(f"Fitting failed: {fit_error}")

        single_fit_state = st.session_state['fit_results_csv'].get(fit_curve_key)
        if single_fit_state and single_fit_state.get("backend") != FIT_BACKENDS["General Peaks"]:
            single_fit_state = None
        with st.expander("📋 Single-Fit Results", expanded=bool(single_fit_state)):
            if single_fit_state:
                if single_fit_state.get("success"):
                    st.success(f"Status: success ({single_fit_state.get('message', 'ok')})")
                else:
                    st.warning(f"Status: failed ({single_fit_state.get('message', 'unknown')})")

                fit_plot = _create_fit_plot_figure(single_fit_state)
                st.plotly_chart(fit_plot, use_container_width=True)

                metrics = single_fit_state.get("metrics", {})
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R²", _format_metric(metrics.get("r2")))
                m2.metric("RMSE", _format_metric(metrics.get("rmse")))
                m3.metric("MAE", _format_metric(metrics.get("mae")))
                m4.metric("Chi²(red)", _format_metric(metrics.get("chi2_reduced")))

                param_rows = [{"parameter": k, "value": v} for k, v in sorted(single_fit_state.get("params", {}).items())]
                if param_rows:
                    st.markdown("**Model Parameters**")
                    st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)

                component_table = single_fit_state.get("component_table", [])
                if component_table:
                    st.markdown("**Per-Peak Parameters**")
                    st.dataframe(pd.DataFrame(component_table), use_container_width=True, hide_index=True)

                metric_rows = [{"metric": k, "value": v} for k, v in sorted(metrics.items())]
                if metric_rows:
                    st.markdown("**Fit Metrics (all)**")
                    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

                single_zip = _build_fit_zip_bytes(single_fit_state)
                single_export_name = _sanitize_filename(single_fit_state.get("curve_label") or fit_curve_key)
                export_col1, export_col2 = st.columns(2)
                with export_col1:
                    st.download_button(
                        label="💾 Export This Fit (ZIP)",
                        data=single_zip,
                        file_name=f"{single_export_name}_fit_results.zip",
                        mime="application/zip",
                        key=_safe_widget_key("fit_export_zip", fit_curve_key),
                        help="Includes raw data, fit sum, individual peaks, parameters, metrics, and plot files.",
                    )
                with export_col2:
                    if st.button("💾 Save This Fit to Server", key=_safe_widget_key("fit_save_server", fit_curve_key)):
                        saved = _save_single_fit_to_server(single_fit_state)
                        st.success(f"Saved fit files to `{saved['run_dir']}`")
            else:
                st.info("Run a single fit to view/export results.")

        # -----------------------------------------------------------------
        # Batch fitting
        # -----------------------------------------------------------------
        with st.expander("📦 Batch Fitting", expanded=False):
            st.caption("Run fitting for multiple curves at once using shared settings.")

            batch_selected_labels = st.multiselect(
                "Curves to batch fit",
                fit_labels,
                default=fit_labels,
                key="fit_batch_curve_selector",
            )
            last_single_setup = st.session_state.get("fit_last_single_setup", {})
            if single_fit_state:
                default_setup = {
                    "shape": single_fit_state.get("shape", fit_shape),
                    "shape_label": single_fit_state.get("shape_label", fit_shape_label),
                    "n_peaks": int(single_fit_state.get("n_peaks", max(1, len(current_peaks)))),
                    "maxiter": int(last_single_setup.get("maxiter", 2000)),
                    "use_range": bool(last_single_setup.get("use_range", fit_use_range)),
                    "x_range": last_single_setup.get("x_range"),
                }
            elif last_single_setup:
                default_setup = {
                    "shape": last_single_setup.get("shape", fit_shape),
                    "shape_label": last_single_setup.get("shape_label", fit_shape_label),
                    "n_peaks": int(last_single_setup.get("n_peaks", max(1, len(current_peaks)))),
                    "maxiter": int(last_single_setup.get("maxiter", 2000)),
                    "use_range": bool(last_single_setup.get("use_range", fit_use_range)),
                    "x_range": last_single_setup.get("x_range"),
                }
            else:
                default_setup = {
                    "shape": fit_shape,
                    "shape_label": fit_shape_label,
                    "n_peaks": max(1, len(current_peaks)),
                    "maxiter": 2000,
                    "use_range": bool(fit_use_range),
                    "x_range": None,
                }

            single_setup_peaks = int(default_setup.get("n_peaks", max(1, len(current_peaks))))
            single_setup_shape_label = str(default_setup.get("shape_label", fit_shape_label))
            if single_setup_shape_label not in FIT_SHAPES:
                single_setup_shape_label = fit_shape_label
            single_setup_shape = str(default_setup.get("shape", FIT_SHAPES[single_setup_shape_label]))
            bcol1, bcol2, bcol3 = st.columns(3)
            with bcol1:
                batch_shape_label = st.selectbox(
                    "Batch shape",
                    list(FIT_SHAPES.keys()),
                    index=list(FIT_SHAPES.keys()).index(single_setup_shape_label),
                    key="fit_batch_shape",
                )
                batch_shape = FIT_SHAPES[batch_shape_label]
            with bcol2:
                batch_maxiter = st.number_input(
                    "Batch max iterations",
                    min_value=100,
                    max_value=10000,
                    value=int(default_setup.get("maxiter", 2000)),
                    step=100,
                    key="fit_batch_maxiter",
                )
            with bcol3:
                batch_default_peaks = st.number_input(
                    "Default peaks per curve",
                    min_value=1,
                    max_value=12,
                    value=single_setup_peaks,
                    step=1,
                    key="fit_batch_default_peaks",
                )

            batch_use_single_setup = st.checkbox(
                "Use current single-fit setup (shape + peak count)",
                value=True,
                key="fit_batch_use_single_setup",
            )
            batch_use_manual_peaks = st.checkbox(
                "Use manually picked peaks when available",
                value=True,
                key="fit_batch_use_manual",
            )
            batch_use_range = st.checkbox(
                "Use x-range for batch fits",
                value=bool(default_setup.get("use_range", fit_use_range)),
                key="fit_batch_use_range",
            )

            batch_x_min, batch_x_max = None, None
            if batch_use_range:
                bx_min = float(np.min(x_full))
                bx_max = float(np.max(x_full))
                if bx_max > bx_min:
                    range_default = default_setup.get("x_range")
                    if range_default is None and x_range_min is not None and x_range_max is not None:
                        range_default = (x_range_min, x_range_max)
                    if range_default is None:
                        range_default = (bx_min, bx_max)
                    rd_min = float(np.clip(range_default[0], bx_min, bx_max))
                    rd_max = float(np.clip(range_default[1], bx_min, bx_max))
                    if rd_max < rd_min:
                        rd_min, rd_max = rd_max, rd_min
                    batch_x_min, batch_x_max = st.slider(
                        "Batch fit x-range",
                        min_value=bx_min,
                        max_value=bx_max,
                        value=(rd_min, rd_max),
                        key="fit_batch_range_slider",
                    )
                else:
                    st.warning("Batch x-range unavailable for this curve.")

            run_batch = st.button(
                "🚀 Run Batch Fitting",
                key="fit_batch_run_button",
                disabled=len(batch_selected_labels) == 0,
            )

            if run_batch:
                summary_rows = []
                for label in batch_selected_labels:
                    item = fit_option_by_label[label]
                    curve_key = item["curve_key"]
                    file_name = item["file_name"]
                    y_name = item["y_col"]
                    df_curve = dataframes[file_name]

                    x_batch, y_batch = _prepare_xy(
                        df_curve[x_col].values,
                        df_curve[y_name].values,
                        x_min=batch_x_min,
                        x_max=batch_x_max,
                    )
                    if len(x_batch) < 5:
                        summary_rows.append({
                            "curve_key": curve_key,
                            "curve": label,
                            "status": "failed",
                            "message": "not enough data points",
                            "n_peaks": np.nan,
                            "r2": np.nan,
                            "rmse": np.nan,
                        })
                        continue

                    manual_peaks = _dedupe_sorted(
                        st.session_state['fit_peak_guesses'].get(curve_key, []),
                        max(float(np.ptp(x_batch)) / 5000.0, 1e-9),
                    )
                    if batch_x_min is not None and batch_x_max is not None:
                        manual_peaks = [
                            peak for peak in manual_peaks
                            if float(batch_x_min) <= peak <= float(batch_x_max)
                        ]
                    if batch_use_manual_peaks and manual_peaks:
                        n_peaks = len(manual_peaks)
                        peak_guesses = manual_peaks
                    else:
                        n_peaks = int(single_setup_peaks if batch_use_single_setup else batch_default_peaks)
                        peak_guesses = []

                    run_shape = batch_shape
                    run_shape_label = batch_shape_label
                    if batch_use_single_setup:
                        run_shape = single_setup_shape or batch_shape
                        run_shape_label = single_setup_shape_label or batch_shape_label

                    try:
                        fit_state = _run_multipeak_fit(
                            x_batch,
                            y_batch,
                            shape=run_shape,
                            shape_label=run_shape_label,
                            n_peaks=n_peaks,
                            maxiter=int(batch_maxiter),
                            peak_guesses=peak_guesses,
                            x_col_name=x_col,
                            y_col_name=y_name,
                        )
                        fit_state.update({
                            "source_file": file_name,
                            "source_curve_key": curve_key,
                            "curve_label": label,
                        })
                        st.session_state['fit_results_csv'][curve_key] = fit_state
                        summary_rows.append({
                            "curve_key": curve_key,
                            "curve": label,
                            "status": "success" if fit_state.get("success") else "failed",
                            "message": fit_state.get("message"),
                            "n_peaks": n_peaks,
                            "r2": fit_state.get("metrics", {}).get("r2"),
                            "rmse": fit_state.get("metrics", {}).get("rmse"),
                        })
                    except Exception as batch_error:
                        summary_rows.append({
                            "curve_key": curve_key,
                            "curve": label,
                            "status": "failed",
                            "message": str(batch_error),
                            "n_peaks": n_peaks,
                            "r2": np.nan,
                            "rmse": np.nan,
                        })

                st.session_state['fit_batch_last_summary'] = summary_rows
                st.rerun()

            batch_summary = st.session_state.get('fit_batch_last_summary', [])
            if batch_summary:
                batch_df = pd.DataFrame(batch_summary)
                st.markdown("**Batch summary**")
                st.dataframe(batch_df, use_container_width=True, hide_index=True)

                successful_states = {}
                for row in batch_summary:
                    if row.get("status") == "success":
                        ck = row.get("curve_key")
                        state = st.session_state['fit_results_csv'].get(ck) if ck else None
                        if ck and state and state.get("backend") == FIT_BACKENDS["General Peaks"]:
                            successful_states[ck] = state

                if successful_states:
                    batch_zip = _build_batch_fit_zip_bytes(successful_states)
                    b_export_col1, b_export_col2 = st.columns(2)
                    with b_export_col1:
                        st.download_button(
                            label="💾 Export Batch Results (ZIP)",
                            data=batch_zip,
                            file_name="batch_fit_results.zip",
                            mime="application/zip",
                            key="fit_batch_export_zip",
                            help="Exports each fitted curve with raw/sum/individual peaks, parameters, metrics, and plot files.",
                        )
                    with b_export_col2:
                        if st.button("💾 Save Batch to Server", key="fit_batch_save_server"):
                            saved = _save_batch_fit_to_server(successful_states, batch_summary)
                            st.success(f"Saved batch files to `{saved['run_dir']}`")
    elif fit_backend == FIT_BACKENDS["SAXS Physics"]:
        if not PYSAXS_AVAILABLE:
            st.warning(
                "SAXS backend is not available. "
                f"Import error: {PYSAXS_IMPORT_ERROR or 'unknown'}"
            )
        else:
            with st.expander("📚 SAXS Model Library", expanded=False):
                library_rows = get_saxs_model_library_rows()
                if library_rows:
                    st.dataframe(pd.DataFrame(library_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("SAXS model metadata not available.")

            fit_use_range = False
            x_range_min, x_range_max = None, None
            x_fit, y_fit_data = x_full, y_full

            single_fit_state = st.session_state['fit_results_csv'].get(fit_curve_key)
            if single_fit_state and single_fit_state.get("backend") != FIT_BACKENDS["SAXS Physics"]:
                single_fit_state = None

            with st.expander("🎯 Single-Curve Fitting (SAXS)", expanded=True):
                last_saxs_setup = st.session_state.get("fit_last_single_setup_saxs", {})
                default_shape_key = str(last_saxs_setup.get("shape", "sphere"))
                default_shape_label = SAXS_MODEL_KEY_TO_LABEL.get(default_shape_key, "Sphere")
                if default_shape_label not in SAXS_MODEL_LABEL_TO_KEY:
                    default_shape_label = "Sphere"

                scol1, scol2, scol3 = st.columns(3)
                with scol1:
                    saxs_shape_label = st.selectbox(
                        "SAXS model",
                        list(SAXS_MODEL_LABEL_TO_KEY.keys()),
                        index=list(SAXS_MODEL_LABEL_TO_KEY.keys()).index(default_shape_label),
                        key=_safe_widget_key("saxs_shape", fit_curve_key),
                    )
                    saxs_shape = SAXS_MODEL_LABEL_TO_KEY[saxs_shape_label]
                    shape_detail = get_saxs_model_detail(saxs_shape)
                    if shape_detail:
                        st.caption(
                            f"{shape_detail.get('description', '')} "
                            f"Recommended {shape_detail.get('size_parameter', 'size')}: "
                            f"{shape_detail.get('recommended_range', 'n/a')} {shape_detail.get('size_units', '')}"
                        )
                with scol2:
                    saxs_polydisperse = st.checkbox(
                        "Polydisperse model",
                        value=bool(last_saxs_setup.get("polydisperse", False)),
                        key=_safe_widget_key("saxs_poly", fit_curve_key),
                    )
                with scol3:
                    saxs_use_porod = st.checkbox(
                        "Include Porod background",
                        value=bool(last_saxs_setup.get("use_porod", False)),
                        key=_safe_widget_key("saxs_porod", fit_curve_key),
                    )

                scol4, scol5 = st.columns([1.1, 1.0])
                with scol4:
                    saxs_maxiter = st.number_input(
                        "Max iterations",
                        min_value=100,
                        max_value=10000,
                        value=int(last_saxs_setup.get("maxiter", 2000)),
                        step=100,
                        key=_safe_widget_key("saxs_maxiter", fit_curve_key),
                    )
                with scol5:
                    fit_use_range = st.checkbox(
                        "Use q-range",
                        value=bool(last_saxs_setup.get("use_range", False)),
                        key=_safe_widget_key("saxs_use_range", fit_curve_key),
                    )

                if fit_use_range:
                    x_min_full = float(np.min(x_full))
                    x_max_full = float(np.max(x_full))
                    if x_max_full > x_min_full:
                        setup_range = last_saxs_setup.get("x_range")
                        if setup_range is None:
                            setup_range = (x_min_full, x_max_full)
                        rd_min = float(np.clip(setup_range[0], x_min_full, x_max_full))
                        rd_max = float(np.clip(setup_range[1], x_min_full, x_max_full))
                        if rd_max < rd_min:
                            rd_min, rd_max = rd_max, rd_min
                        x_range_min, x_range_max = st.slider(
                            "Fit q-range",
                            min_value=x_min_full,
                            max_value=x_max_full,
                            value=(rd_min, rd_max),
                            key=_safe_widget_key("saxs_range_slider", fit_curve_key),
                        )
                    else:
                        st.warning("Q range is degenerate; fitting full range instead.")

                x_fit, y_fit_data = _prepare_xy(
                    fit_df[x_col].values,
                    fit_df[fit_y_col].values,
                    x_min=x_range_min,
                    x_max=x_range_max,
                )
                if len(x_fit) < 5:
                    st.error("Not enough points remain after range filtering.")
                else:
                    use_initial_override = st.checkbox(
                        "Override initial guess",
                        value=False,
                        key=_safe_widget_key("saxs_override", fit_curve_key),
                    )
                    initial_overrides = {}
                    if use_initial_override:
                        ocol1, ocol2, ocol3 = st.columns(3)
                        with ocol1:
                            initial_overrides["radius"] = st.number_input(
                                "Initial radius",
                                min_value=0.01,
                                value=float(last_saxs_setup.get("override_radius", 50.0)),
                                format="%.5g",
                                key=_safe_widget_key("saxs_init_radius", fit_curve_key),
                            )
                        with ocol2:
                            initial_overrides["scale"] = st.number_input(
                                "Initial scale",
                                min_value=0.0,
                                value=float(last_saxs_setup.get("override_scale", 1.0)),
                                format="%.5g",
                                key=_safe_widget_key("saxs_init_scale", fit_curve_key),
                            )
                        with ocol3:
                            initial_overrides["background"] = st.number_input(
                                "Initial background",
                                value=float(last_saxs_setup.get("override_background", 0.0)),
                                format="%.5g",
                                key=_safe_widget_key("saxs_init_background", fit_curve_key),
                            )
                        if saxs_polydisperse:
                            initial_overrides["sigma_rel"] = st.number_input(
                                "Initial sigma_rel",
                                min_value=0.0,
                                max_value=0.8,
                                value=float(last_saxs_setup.get("override_sigma_rel", 0.1)),
                                step=0.01,
                                format="%.4f",
                                key=_safe_widget_key("saxs_init_sigma", fit_curve_key),
                            )
                        if saxs_use_porod:
                            pcol1, pcol2 = st.columns(2)
                            with pcol1:
                                initial_overrides["porod_scale"] = st.number_input(
                                    "Initial porod_scale",
                                    min_value=0.0,
                                    value=float(last_saxs_setup.get("override_porod_scale", 0.01)),
                                    format="%.5g",
                                    key=_safe_widget_key("saxs_init_porod_scale", fit_curve_key),
                                )
                            with pcol2:
                                initial_overrides["porod_exp"] = st.number_input(
                                    "Initial porod_exp",
                                    min_value=2.0,
                                    max_value=6.0,
                                    value=float(last_saxs_setup.get("override_porod_exp", 4.0)),
                                    step=0.1,
                                    format="%.3f",
                                    key=_safe_widget_key("saxs_init_porod_exp", fit_curve_key),
                                )

                    run_saxs_single = st.button(
                        "🚀 Fit SAXS Model",
                        key=_safe_widget_key("saxs_fit_run_button", fit_curve_key),
                    )

                    if run_saxs_single:
                        setup_record = {
                            "shape": saxs_shape,
                            "shape_label": saxs_shape_label,
                            "polydisperse": bool(saxs_polydisperse),
                            "use_porod": bool(saxs_use_porod),
                            "maxiter": int(saxs_maxiter),
                            "use_range": bool(fit_use_range),
                            "x_range": (
                                float(x_range_min),
                                float(x_range_max),
                            ) if fit_use_range and x_range_min is not None and x_range_max is not None else None,
                            "curve_key": fit_curve_key,
                            "curve_label": fit_selected_label,
                        }
                        if use_initial_override:
                            for key, value in initial_overrides.items():
                                setup_record[f"override_{key}"] = float(value)
                        st.session_state["fit_last_single_setup_saxs"] = setup_record

                        try:
                            fit_state = run_saxs_fit(
                                x_fit,
                                y_fit_data,
                                shape=saxs_shape,
                                shape_label=saxs_shape_label,
                                polydisperse=bool(saxs_polydisperse),
                                use_porod=bool(saxs_use_porod),
                                maxiter=int(saxs_maxiter),
                                x_col_name=x_col,
                                y_col_name=fit_y_col,
                                initial_overrides=initial_overrides if use_initial_override else None,
                            )
                            fit_state.update({
                                "source_file": fit_file_name,
                                "source_curve_key": fit_curve_key,
                                "curve_label": fit_selected_label,
                            })
                            st.session_state['fit_results_csv'][fit_curve_key] = fit_state
                            single_fit_state = fit_state
                            if fit_state.get("success"):
                                st.success("SAXS fit completed.")
                            else:
                                st.warning(f"SAXS fit finished with warning: {fit_state.get('message')}")
                        except Exception as fit_error:
                            st.error(f"SAXS fitting failed: {fit_error}")

            single_fit_state = st.session_state['fit_results_csv'].get(fit_curve_key)
            if single_fit_state and single_fit_state.get("backend") != FIT_BACKENDS["SAXS Physics"]:
                single_fit_state = None

            with st.expander("📋 Single-Fit Results", expanded=bool(single_fit_state)):
                if single_fit_state:
                    if single_fit_state.get("success"):
                        st.success(f"Status: success ({single_fit_state.get('message', 'ok')})")
                    else:
                        st.warning(f"Status: failed ({single_fit_state.get('message', 'unknown')})")

                    fit_plot = _create_fit_plot_figure(single_fit_state)
                    st.plotly_chart(fit_plot, use_container_width=True)

                    smeta = {
                        "Model": single_fit_state.get("saxs_shape_label", single_fit_state.get("shape")),
                        "Polydisperse": single_fit_state.get("saxs_polydisperse", False),
                        "Use Porod": single_fit_state.get("saxs_use_porod", False),
                    }
                    st.caption(
                        f"Model: {smeta['Model']} | "
                        f"Polydisperse: {smeta['Polydisperse']} | "
                        f"Porod: {smeta['Use Porod']}"
                    )

                    metrics = single_fit_state.get("metrics", {})
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("R²", _format_metric(metrics.get("r2")))
                    m2.metric("RMSE", _format_metric(metrics.get("rmse")))
                    m3.metric("MAE", _format_metric(metrics.get("mae")))
                    m4.metric("Chi²(red)", _format_metric(metrics.get("chi2_reduced")))

                    param_rows = [{"parameter": k, "value": v} for k, v in sorted(single_fit_state.get("params", {}).items())]
                    if param_rows:
                        st.markdown("**Model Parameters**")
                        st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)

                    component_table = single_fit_state.get("component_table", [])
                    if component_table:
                        st.markdown("**Model Summary**")
                        st.dataframe(pd.DataFrame(component_table), use_container_width=True, hide_index=True)

                    metric_rows = [{"metric": k, "value": v} for k, v in sorted(metrics.items())]
                    if metric_rows:
                        st.markdown("**Fit Metrics (all)**")
                        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

                    single_zip = _build_fit_zip_bytes(single_fit_state)
                    single_export_name = _sanitize_filename(single_fit_state.get("curve_label") or fit_curve_key)
                    export_col1, export_col2 = st.columns(2)
                    with export_col1:
                        st.download_button(
                            label="💾 Export This Fit (ZIP)",
                            data=single_zip,
                            file_name=f"{single_export_name}_fit_results.zip",
                            mime="application/zip",
                            key=_safe_widget_key("saxs_fit_export_zip", fit_curve_key),
                            help="Includes raw data, fit sum, parameters, metrics, and plot files.",
                        )
                    with export_col2:
                        if st.button("💾 Save This Fit to Server", key=_safe_widget_key("saxs_fit_save_server", fit_curve_key)):
                            saved = _save_single_fit_to_server(single_fit_state)
                            st.success(f"Saved fit files to `{saved['run_dir']}`")
                else:
                    st.info("Run a SAXS fit to view/export results.")

            # -----------------------------------------------------------------
            # SAXS batch fitting
            # -----------------------------------------------------------------
            with st.expander("📦 Batch Fitting", expanded=False):
                st.caption("Run SAXS model fitting for multiple curves using shared settings.")

                batch_selected_labels = st.multiselect(
                    "Curves to batch fit",
                    fit_labels,
                    default=fit_labels,
                    key="saxs_batch_curve_selector",
                )

                last_saxs_setup = st.session_state.get("fit_last_single_setup_saxs", {})
                default_shape_key = str(last_saxs_setup.get("shape", "sphere"))
                default_shape_label = SAXS_MODEL_KEY_TO_LABEL.get(default_shape_key, "Sphere")
                if default_shape_label not in SAXS_MODEL_LABEL_TO_KEY:
                    default_shape_label = "Sphere"

                bcol1, bcol2, bcol3 = st.columns(3)
                with bcol1:
                    batch_shape_label = st.selectbox(
                        "Batch SAXS model",
                        list(SAXS_MODEL_LABEL_TO_KEY.keys()),
                        index=list(SAXS_MODEL_LABEL_TO_KEY.keys()).index(default_shape_label),
                        key="saxs_fit_batch_shape",
                    )
                    batch_shape = SAXS_MODEL_LABEL_TO_KEY[batch_shape_label]
                with bcol2:
                    batch_poly = st.checkbox(
                        "Batch polydisperse",
                        value=bool(last_saxs_setup.get("polydisperse", False)),
                        key="saxs_fit_batch_poly",
                    )
                with bcol3:
                    batch_porod = st.checkbox(
                        "Batch use Porod",
                        value=bool(last_saxs_setup.get("use_porod", False)),
                        key="saxs_fit_batch_porod",
                    )

                bcol4, bcol5 = st.columns(2)
                with bcol4:
                    batch_maxiter = st.number_input(
                        "Batch max iterations",
                        min_value=100,
                        max_value=10000,
                        value=int(last_saxs_setup.get("maxiter", 2000)),
                        step=100,
                        key="saxs_fit_batch_maxiter",
                    )
                with bcol5:
                    batch_use_single_setup = st.checkbox(
                        "Use current single-fit setup",
                        value=True,
                        key="saxs_fit_batch_use_single_setup",
                    )

                batch_use_range = st.checkbox(
                    "Use q-range for batch fits",
                    value=bool(last_saxs_setup.get("use_range", False)),
                    key="saxs_fit_batch_use_range",
                )

                batch_x_min, batch_x_max = None, None
                if batch_use_range:
                    bx_min = float(np.min(x_full))
                    bx_max = float(np.max(x_full))
                    if bx_max > bx_min:
                        range_default = last_saxs_setup.get("x_range")
                        if range_default is None:
                            range_default = (bx_min, bx_max)
                        rd_min = float(np.clip(range_default[0], bx_min, bx_max))
                        rd_max = float(np.clip(range_default[1], bx_min, bx_max))
                        if rd_max < rd_min:
                            rd_min, rd_max = rd_max, rd_min
                        batch_x_min, batch_x_max = st.slider(
                            "Batch fit q-range",
                            min_value=bx_min,
                            max_value=bx_max,
                            value=(rd_min, rd_max),
                            key="saxs_fit_batch_range_slider",
                        )
                    else:
                        st.warning("Batch q-range unavailable for this curve.")

                run_batch = st.button(
                    "🚀 Run Batch Fitting",
                    key="saxs_fit_batch_run_button",
                    disabled=len(batch_selected_labels) == 0,
                )

                if run_batch:
                    summary_rows = []
                    for label in batch_selected_labels:
                        item = fit_option_by_label[label]
                        curve_key = item["curve_key"]
                        file_name = item["file_name"]
                        y_name = item["y_col"]
                        df_curve = dataframes[file_name]

                        x_batch, y_batch = _prepare_xy(
                            df_curve[x_col].values,
                            df_curve[y_name].values,
                            x_min=batch_x_min,
                            x_max=batch_x_max,
                        )
                        if len(x_batch) < 5:
                            summary_rows.append({
                                "curve_key": curve_key,
                                "curve": label,
                                "status": "failed",
                                "message": "not enough data points",
                                "model": np.nan,
                                "r2": np.nan,
                                "rmse": np.nan,
                            })
                            continue

                        run_shape = batch_shape
                        run_shape_label = batch_shape_label
                        run_poly = batch_poly
                        run_porod = batch_porod
                        run_maxiter = int(batch_maxiter)
                        if batch_use_single_setup and last_saxs_setup:
                            run_shape = str(last_saxs_setup.get("shape", run_shape))
                            run_shape_label = str(last_saxs_setup.get("shape_label", run_shape_label))
                            run_poly = bool(last_saxs_setup.get("polydisperse", run_poly))
                            run_porod = bool(last_saxs_setup.get("use_porod", run_porod))
                            run_maxiter = int(last_saxs_setup.get("maxiter", run_maxiter))

                        try:
                            fit_state = run_saxs_fit(
                                x_batch,
                                y_batch,
                                shape=run_shape,
                                shape_label=run_shape_label,
                                polydisperse=run_poly,
                                use_porod=run_porod,
                                maxiter=run_maxiter,
                                x_col_name=x_col,
                                y_col_name=y_name,
                                initial_overrides=None,
                            )
                            fit_state.update({
                                "source_file": file_name,
                                "source_curve_key": curve_key,
                                "curve_label": label,
                            })
                            st.session_state['fit_results_csv'][curve_key] = fit_state
                            summary_rows.append({
                                "curve_key": curve_key,
                                "curve": label,
                                "status": "success" if fit_state.get("success") else "failed",
                                "message": fit_state.get("message"),
                                "model": fit_state.get("saxs_shape_label"),
                                "r2": fit_state.get("metrics", {}).get("r2"),
                                "rmse": fit_state.get("metrics", {}).get("rmse"),
                            })
                        except Exception as batch_error:
                            summary_rows.append({
                                "curve_key": curve_key,
                                "curve": label,
                                "status": "failed",
                                "message": str(batch_error),
                                "model": run_shape_label,
                                "r2": np.nan,
                                "rmse": np.nan,
                            })

                    st.session_state['fit_batch_last_summary'] = summary_rows
                    st.rerun()

                batch_summary = st.session_state.get('fit_batch_last_summary', [])
                if batch_summary:
                    batch_df = pd.DataFrame(batch_summary)
                    st.markdown("**Batch summary**")
                    st.dataframe(batch_df, use_container_width=True, hide_index=True)

                    successful_states = {}
                    for row in batch_summary:
                        if row.get("status") == "success":
                            ck = row.get("curve_key")
                            state = st.session_state['fit_results_csv'].get(ck) if ck else None
                            if ck and state and state.get("backend") == FIT_BACKENDS["SAXS Physics"]:
                                successful_states[ck] = state

                    if successful_states:
                        batch_zip = _build_batch_fit_zip_bytes(successful_states)
                        b_export_col1, b_export_col2 = st.columns(2)
                        with b_export_col1:
                            st.download_button(
                                label="💾 Export Batch Results (ZIP)",
                                data=batch_zip,
                                file_name="batch_fit_results.zip",
                                mime="application/zip",
                                key="saxs_fit_batch_export_zip",
                                help="Exports each fitted curve with raw/sum, parameters, metrics, and plot files.",
                            )
                        with b_export_col2:
                            if st.button("💾 Save Batch to Server", key="saxs_fit_batch_save_server"):
                                saved = _save_batch_fit_to_server(successful_states, batch_summary)
                                st.success(f"Saved batch files to `{saved['run_dir']}`")
    elif fit_backend == FIT_BACKENDS["ML-Assisted (Preview)"]:
        ml_models = list_ml_models()
        if not ml_models:
            st.warning("No ML models configured.")
        else:
            ml_model_by_label = {row["label"]: row for row in ml_models}

            def _run_ml_refinement(ml_bundle, refine_maxiter):
                pred = dict(ml_bundle.get("prediction", {}))
                pre = dict(ml_bundle.get("preprocessed", {}))
                x_ref = np.asarray(pre.get("x_raw", x_full), dtype=float)
                y_ref = np.asarray(pre.get("y_raw", y_full), dtype=float)
                target_backend = str(pred.get("recommended_backend", FIT_BACKENDS["General Peaks"]))

                if target_backend == FIT_BACKENDS["SAXS Physics"]:
                    if not PYSAXS_AVAILABLE:
                        raise RuntimeError("SAXS backend unavailable for ML refinement.")
                    saxs_shape = str(pred.get("saxs_shape", "sphere"))
                    saxs_shape_label = SAXS_MODEL_KEY_TO_LABEL.get(saxs_shape, saxs_shape.title())
                    fit_state = run_saxs_fit(
                        x_ref,
                        y_ref,
                        shape=saxs_shape,
                        shape_label=saxs_shape_label,
                        polydisperse=bool(pred.get("saxs_polydisperse", False)),
                        use_porod=bool(pred.get("saxs_use_porod", False)),
                        maxiter=int(refine_maxiter),
                        x_col_name=x_col,
                        y_col_name=fit_y_col,
                        initial_overrides=pred.get("initial_overrides", {}),
                    )
                else:
                    peak_centers = [float(v) for v in pred.get("peak_centers", [])]
                    peak_count = int(pred.get("peak_count", len(peak_centers) or 1))
                    peak_count = max(1, peak_count)
                    if len(peak_centers) > peak_count:
                        peak_centers = peak_centers[:peak_count]
                    if len(peak_centers) < peak_count:
                        grid = np.linspace(float(np.min(x_ref)), float(np.max(x_ref)), peak_count + 2)[1:-1]
                        for v in grid:
                            if len(peak_centers) >= peak_count:
                                break
                            peak_centers.append(float(v))
                    peak_centers = sorted(peak_centers)[:peak_count]
                    if not peak_centers:
                        peak_centers = [float(x_ref[int(np.argmax(y_ref))])]
                    shape_label = str(pred.get("peak_shape_label", "Pseudo-Voigt"))
                    if shape_label not in FIT_SHAPES:
                        shape_label = "Pseudo-Voigt"
                    fit_state = _run_multipeak_fit(
                        x_ref,
                        y_ref,
                        shape=FIT_SHAPES[shape_label],
                        shape_label=shape_label,
                        n_peaks=peak_count,
                        maxiter=int(refine_maxiter),
                        peak_guesses=peak_centers,
                        x_col_name=x_col,
                        y_col_name=fit_y_col,
                    )

                fit_state.update({
                    "source_file": fit_file_name,
                    "source_curve_key": fit_curve_key,
                    "curve_label": fit_selected_label,
                    "ml_seed_model": pred.get("model_key"),
                    "ml_seed_confidence": pred.get("confidence"),
                })
                st.session_state['fit_results_csv'][fit_curve_key] = fit_state
                return fit_state

            with st.expander("📚 ML Model Library", expanded=False):
                st.dataframe(pd.DataFrame(ml_models), use_container_width=True, hide_index=True)

            with st.expander("🤖 ML-Assisted Setup", expanded=True):
                ml_col1, ml_col2 = st.columns([1.2, 1.0])
                with ml_col1:
                    ml_model_label = st.selectbox(
                        "ML model",
                        list(ml_model_by_label.keys()),
                        key=_safe_widget_key("ml_model_select", fit_curve_key),
                    )
                    ml_model_info = ml_model_by_label[ml_model_label]
                    st.caption(
                        f"Status: {ml_model_info.get('status', 'n/a')} | "
                        f"Target backend: {ml_model_info.get('target_backend', 'n/a')}"
                    )
                    st.caption(ml_model_info.get("description", ""))
                with ml_col2:
                    ml_target_mode = st.radio(
                        "Target backend override",
                        ["Auto", "General Peaks", "SAXS Physics"],
                        horizontal=True,
                        key=_safe_widget_key("ml_target_backend", fit_curve_key),
                    )

                pcol1, pcol2, pcol3 = st.columns(3)
                with pcol1:
                    ml_use_range = st.checkbox(
                        "Use x-range",
                        value=False,
                        key=_safe_widget_key("ml_use_range", fit_curve_key),
                    )
                with pcol2:
                    ml_log_y = st.checkbox(
                        "Log-transform y",
                        value=False,
                        key=_safe_widget_key("ml_log_y", fit_curve_key),
                    )
                with pcol3:
                    ml_normalize_y = st.checkbox(
                        "Normalize y",
                        value=True,
                        key=_safe_widget_key("ml_norm_y", fit_curve_key),
                    )

                ml_x_min, ml_x_max = None, None
                if ml_use_range:
                    x_min_full = float(np.min(x_full))
                    x_max_full = float(np.max(x_full))
                    if x_max_full > x_min_full:
                        ml_x_min, ml_x_max = st.slider(
                            "ML x-range",
                            min_value=x_min_full,
                            max_value=x_max_full,
                            value=(x_min_full, x_max_full),
                            key=_safe_widget_key("ml_range_slider", fit_curve_key),
                        )
                    else:
                        st.warning("X range is degenerate; using full range.")

                pcol4, pcol5, pcol6 = st.columns(3)
                with pcol4:
                    ml_smooth_window = st.slider(
                        "Smoothing window",
                        min_value=1,
                        max_value=31,
                        value=5,
                        step=2,
                        key=_safe_widget_key("ml_smooth_window", fit_curve_key),
                    )
                with pcol5:
                    ml_resample_points = st.select_slider(
                        "Resample points",
                        options=[64, 128, 256, 512, 1024],
                        value=256,
                        key=_safe_widget_key("ml_resample_points", fit_curve_key),
                    )
                with pcol6:
                    ml_refine_maxiter = st.number_input(
                        "Refine max iterations",
                        min_value=100,
                        max_value=10000,
                        value=2000,
                        step=100,
                        key=_safe_widget_key("ml_refine_maxiter", fit_curve_key),
                    )

                ml_auto_refine = st.checkbox(
                    "Auto-run suggested refinement after prediction",
                    value=False,
                    key=_safe_widget_key("ml_auto_refine", fit_curve_key),
                )

                run_ml_predict = st.button(
                    "🔮 Predict Initial Guess",
                    key=_safe_widget_key("ml_predict_button", fit_curve_key),
                )
                if run_ml_predict:
                    try:
                        ml_bundle = run_ml_prediction(
                            x_full,
                            y_full,
                            model_key=ml_model_info["key"],
                            x_min=ml_x_min,
                            x_max=ml_x_max,
                            log_y=bool(ml_log_y),
                            normalize_y=bool(ml_normalize_y),
                            smooth_window=int(ml_smooth_window),
                            resample_points=int(ml_resample_points),
                        )
                        pred = ml_bundle["prediction"]
                        if ml_target_mode == "General Peaks":
                            pred["recommended_backend"] = FIT_BACKENDS["General Peaks"]
                        elif ml_target_mode == "SAXS Physics":
                            pred["recommended_backend"] = FIT_BACKENDS["SAXS Physics"]
                            pred["saxs_shape"] = pred.get("saxs_shape") or "sphere"
                            pred["saxs_polydisperse"] = bool(pred.get("saxs_polydisperse", False))
                            pred["saxs_use_porod"] = bool(pred.get("saxs_use_porod", False))
                        ml_bundle["prediction"] = pred
                        ml_bundle["ui"] = {
                            "curve_key": fit_curve_key,
                            "curve_label": fit_selected_label,
                            "model_label": ml_model_label,
                            "target_override": ml_target_mode,
                            "auto_refine": bool(ml_auto_refine),
                            "refine_maxiter": int(ml_refine_maxiter),
                        }
                        st.session_state['fit_ml_predictions'][fit_curve_key] = ml_bundle
                        st.success("ML prediction completed.")

                        if ml_auto_refine:
                            fit_state = _run_ml_refinement(ml_bundle, refine_maxiter=int(ml_refine_maxiter))
                            if fit_state.get("success"):
                                st.success("Auto-refinement completed.")
                            else:
                                st.warning(f"Auto-refinement finished with warning: {fit_state.get('message')}")
                    except Exception as ml_error:
                        st.error(f"ML prediction failed: {ml_error}")

            ml_bundle = st.session_state['fit_ml_predictions'].get(fit_curve_key)
            with st.expander("🧠 ML Prediction + Refinement", expanded=bool(ml_bundle)):
                if ml_bundle:
                    pred = ml_bundle.get("prediction", {})
                    pre = ml_bundle.get("preprocessed", {})
                    model_name = ml_bundle.get("ui", {}).get("model_label", pred.get("model_key", "n/a"))
                    backend_name = str(pred.get("recommended_backend", "n/a"))
                    confidence = float(pred.get("confidence", np.nan))

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Model", model_name)
                    c2.metric("Recommended Backend", backend_name)
                    c3.metric("Confidence", _format_metric(confidence, ".3f"))
                    st.caption(pred.get("notes", ""))

                    if backend_name == FIT_BACKENDS["General Peaks"]:
                        peak_centers = pred.get("peak_centers", [])
                        peak_count = int(pred.get("peak_count", len(peak_centers) or 1))
                        st.caption(f"Predicted peaks: {peak_count} | shape: {pred.get('peak_shape_label', 'Pseudo-Voigt')}")
                        if peak_centers:
                            st.dataframe(
                                pd.DataFrame({
                                    "peak_index": np.arange(1, len(peak_centers) + 1),
                                    "x_guess": peak_centers,
                                }),
                                use_container_width=True,
                                hide_index=True,
                            )
                    elif backend_name == FIT_BACKENDS["SAXS Physics"]:
                        st.caption(
                            f"Predicted SAXS shape: {SAXS_MODEL_KEY_TO_LABEL.get(pred.get('saxs_shape', 'sphere'), pred.get('saxs_shape', 'sphere'))} | "
                            f"polydisperse: {bool(pred.get('saxs_polydisperse', False))} | "
                            f"Porod: {bool(pred.get('saxs_use_porod', False))}"
                        )
                        init_over = pred.get("initial_overrides", {})
                        if init_over:
                            st.dataframe(
                                pd.DataFrame([{"parameter": k, "value": v} for k, v in init_over.items()]),
                                use_container_width=True,
                                hide_index=True,
                            )

                    x_proc = np.asarray(pre.get("x_proc", []), dtype=float)
                    y_proc = np.asarray(pre.get("y_proc", []), dtype=float)
                    x_res = np.asarray(pre.get("x_resampled", []), dtype=float)
                    y_res = np.asarray(pre.get("y_resampled", []), dtype=float)
                    if len(x_proc) and len(y_proc):
                        ml_fig = go.Figure()
                        ml_fig.add_trace(
                            go.Scatter(
                                x=x_proc,
                                y=y_proc,
                                mode='lines',
                                name='Preprocessed',
                                line=dict(color='#1f77b4', width=1.4),
                            )
                        )
                        if len(x_res) and len(y_res):
                            ml_fig.add_trace(
                                go.Scatter(
                                    x=x_res,
                                    y=y_res,
                                    mode='lines',
                                    name='Resampled',
                                    line=dict(color='#ff7f0e', width=1.2, dash='dot'),
                                )
                            )
                        ml_fig.update_layout(height=320, title="ML Preprocessing Preview")
                        ml_fig.update_xaxes(title=x_col)
                        ml_fig.update_yaxes(title=f"{fit_y_col} (processed)")
                        st.plotly_chart(ml_fig, use_container_width=True)

                    action_col1, action_col2, action_col3 = st.columns(3)
                    with action_col1:
                        if st.button("📌 Apply Predicted Peaks", key=_safe_widget_key("ml_apply_peaks", fit_curve_key)):
                            peak_centers = [float(v) for v in pred.get("peak_centers", [])]
                            if peak_centers:
                                st.session_state['fit_peak_guesses'][fit_curve_key] = _dedupe_sorted(
                                    peak_centers,
                                    max(float(np.ptp(x_full)) / 5000.0, 1e-9),
                                )
                                st.session_state['fit_last_single_setup'] = {
                                    "shape": FIT_SHAPES.get(pred.get("peak_shape_label", "Pseudo-Voigt"), "pseudo_voigt"),
                                    "shape_label": pred.get("peak_shape_label", "Pseudo-Voigt"),
                                    "n_peaks": int(pred.get("peak_count", len(peak_centers))),
                                    "maxiter": int(ml_bundle.get("ui", {}).get("refine_maxiter", 2000)),
                                    "use_range": False,
                                    "x_range": None,
                                    "curve_key": fit_curve_key,
                                    "curve_label": fit_selected_label,
                                }
                                st.success("Predicted peaks applied to General Peaks backend.")
                                st.rerun()
                            else:
                                st.warning("No peak centers available in prediction.")
                    with action_col2:
                        if st.button("⚡ Run Suggested Refinement", key=_safe_widget_key("ml_run_refine", fit_curve_key)):
                            try:
                                fit_state = _run_ml_refinement(
                                    ml_bundle,
                                    refine_maxiter=int(ml_bundle.get("ui", {}).get("refine_maxiter", 2000)),
                                )
                                if fit_state.get("success"):
                                    st.success("Refinement fit completed.")
                                else:
                                    st.warning(f"Refinement fit finished with warning: {fit_state.get('message')}")
                            except Exception as ml_refine_error:
                                st.error(f"Refinement failed: {ml_refine_error}")
                    with action_col3:
                        current_fit = st.session_state['fit_results_csv'].get(fit_curve_key)
                        if current_fit:
                            st.caption(
                                f"Latest fit backend: {current_fit.get('backend', 'unknown')} | "
                                f"shape: {current_fit.get('shape_label', current_fit.get('shape', 'n/a'))}"
                            )
                else:
                    st.info("Run ML prediction first to view suggestions and refinement actions.")
    else:
        st.info("Select a supported backend to start fitting.")

# Export
st.divider()

if use_plotly:
    # Plotly export
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        html_buffer = io.StringIO()
        fig.write_html(html_buffer)
        html_bytes = html_buffer.getvalue().encode()
        st.download_button(
            label="💾 Download HTML",
            data=html_bytes,
            file_name=f"plot_{x_col}_vs_{y_col}.html",
            mime="text/html",
            help="Interactive plot - preserves hover functionality!"
        )

    with col2:
        try:
            img_bytes = fig.to_image(format="png", width=1200, height=800)
            st.download_button(
                label="💾 Download PNG",
                data=img_bytes,
                file_name=f"plot_{x_col}_vs_{y_col}.png",
                mime="image/png"
            )
        except:
            st.info("Install kaleido for PNG export")

    with col3:
        try:
            img_bytes = fig.to_image(format="svg")
            st.download_button(
                label="💾 Download SVG",
                data=img_bytes,
                file_name=f"plot_{x_col}_vs_{y_col}.svg",
                mime="image/svg+xml"
            )
        except:
            st.info("Install kaleido for SVG export")

    with col4:
        st.metric("Curves", plotted_count)

else:
    # Matplotlib export
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        buf = _save_fig_to_bytes(fig, dpi=300)
        st.download_button(
            label="💾 Download Plot (PNG, 300 DPI)",
            data=buf,
            file_name=f"plot_{x_col}_vs_{y_col}.png",
            mime="image/png"
        )

    with col2:
        buf_svg = _save_fig_to_bytes(fig, format='svg')
        st.download_button(
            label="💾 Download (SVG)",
            data=buf_svg,
            file_name=f"plot_{x_col}_vs_{y_col}.svg",
            mime="image/svg+xml"
        )

    with col3:
        st.metric("Curves", plotted_count)

    plt.close(fig)

# ---------------------------------------------------------------------------
# Data Preview & Statistics
# ---------------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    with st.expander("📄 Data Preview"):
        for file_name, df in dataframes.items():
            st.subheader(shorten_path(file_name, 50))
            st.dataframe(df.head(10))
            st.text(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            st.divider()

with col2:
    with st.expander("📊 Statistics"):
        for file_name, df in dataframes.items():
            y_cols_for_file = file_y_columns.get(file_name, [])
            if x_col in df.columns and y_cols_for_file:
                st.subheader(shorten_path(file_name, 50))
                if x_col in df.columns:
                    st.markdown(f"**{x_col}**")
                    st.text(f"  Min: {df[x_col].min():.4f}  Max: {df[x_col].max():.4f}  Mean: {df[x_col].mean():.4f}")
                for yc in y_cols_for_file:
                    if yc in df.columns:
                        st.markdown(f"**{yc}**")
                        st.text(f"  Min: {df[yc].min():.4f}  Max: {df[yc].max():.4f}  Mean: {df[yc].mean():.4f}")
                st.divider()
