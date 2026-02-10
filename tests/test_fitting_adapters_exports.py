import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from NanoOrganizer.web_app.components import fitting_adapters as fa


def _build_demo_fit_state():
    if not fa.PYFITTING_AVAILABLE:
        pytest.skip("pyFitting is required for fitting adapter export tests.")

    x = np.linspace(-2.0, 2.0, 220)
    y = (
        1.4 * np.exp(-0.5 * ((x + 0.45) / 0.2) ** 2)
        + 0.9 * np.exp(-0.5 * ((x - 0.55) / 0.25) ** 2)
        + 0.04
    )
    state = fa.run_general_peak_fit(
        x,
        y,
        shape="gaussian",
        shape_label="Gaussian",
        n_peaks=2,
        maxiter=400,
        peak_guesses=[-0.45, 0.55],
        x_col_name="q",
        y_col_name="I",
    )
    state.update(
        {
            "source_file": "demo.csv",
            "source_curve_key": "curve_001",
            "curve_label": "demo_curve",
        }
    )
    return state


def _pseudo_voigt_profile(x, center, width, eta):
    x = np.asarray(x, dtype=float)
    width = max(float(width), 1e-12)
    eta = float(np.clip(eta, 0.0, 1.0))
    gaussian = np.exp(-0.5 * ((x - center) / width) ** 2)
    lorentzian = 1.0 / (1.0 + ((x - center) / width) ** 2)
    return eta * lorentzian + (1.0 - eta) * gaussian


def test_run_general_peak_fit_pseudo_voigt_schema_and_eta():
    if not fa.PYFITTING_AVAILABLE:
        pytest.skip("pyFitting is required for fitting adapter peak-fit tests.")

    x = np.linspace(-2.0, 2.0, 300)
    y = (
        1.6 * _pseudo_voigt_profile(x, center=-0.5, width=0.18, eta=0.30)
        + 1.1 * _pseudo_voigt_profile(x, center=0.65, width=0.24, eta=0.75)
        + 0.05
    )

    state = fa.run_general_peak_fit(
        x,
        y,
        shape="pseudo_voigt",
        shape_label="Pseudo-Voigt",
        n_peaks=2,
        maxiter=500,
        peak_guesses=[-0.5, -0.50001, 0.65],
        x_col_name="q",
        y_col_name="I",
    )

    assert state["backend"] == "general_peaks"
    assert state["shape"] == "pseudo_voigt"
    assert state["x_col"] == "q"
    assert state["y_col"] == "I"
    assert state["n_peaks"] == 2
    assert len(state["x"]) == len(x)
    assert len(state["y_fit"]) == len(x)
    assert isinstance(state.get("params"), dict)
    assert isinstance(state.get("metrics"), dict)

    peak_guesses = state.get("peak_guesses", [])
    assert len(peak_guesses) == 2
    assert float(peak_guesses[0]) < float(peak_guesses[1])

    component_rows = state.get("component_table", [])
    assert len(component_rows) == 2
    assert all("eta" in row for row in component_rows)
    assert all(0.0 <= float(row["eta"]) <= 1.0 for row in component_rows)
    assert "eta1" in state["params"]
    assert "eta2" in state["params"]


def test_run_general_peak_fit_gaussian_component_schema():
    if not fa.PYFITTING_AVAILABLE:
        pytest.skip("pyFitting is required for fitting adapter peak-fit tests.")

    x = np.linspace(-2.0, 2.0, 260)
    y = (
        1.5 * np.exp(-0.5 * ((x + 0.6) / 0.20) ** 2)
        + 0.9 * np.exp(-0.5 * ((x - 0.45) / 0.27) ** 2)
        + 0.04
    )

    state = fa.run_general_peak_fit(
        x,
        y,
        shape="gaussian",
        shape_label="Gaussian",
        n_peaks=2,
        maxiter=400,
        peak_guesses=[-0.6, 0.45],
    )

    assert state["backend"] == "general_peaks"
    assert state["shape"] == "gaussian"
    assert state["n_peaks"] == 2
    assert len(state.get("components", [])) == 2

    component_rows = state.get("component_table", [])
    assert len(component_rows) == 2
    assert all("peak" in row and "A" in row and "mu" in row and "w" in row for row in component_rows)
    assert all("eta" not in row for row in component_rows)


def test_simulate_saxs_curve_background_modes():
    if not fa.PYSAXS_AVAILABLE:
        pytest.skip("pySAXSFitting is required for SAXS simulation tests.")

    const_df, const_meta = fa.simulate_saxs_curve(
        shape="sphere",
        q_min=0.01,
        q_max=0.2,
        n_points=300,
        radius=45.0,
        scale=2.2,
        background_mode="constant",
        background_const=0.03,
        noise_level=0.0,
        seed=123,
    )
    assert const_meta["shape"] == "sphere"
    assert const_meta["background_mode"] == "constant"
    assert len(const_df) == 300
    assert {"q", "intensity", "intensity_clean", "intensity_shape", "intensity_background"}.issubset(
        set(const_df.columns)
    )
    assert np.allclose(const_df["intensity"], const_df["intensity_clean"])
    assert np.allclose(const_df["intensity_background"], 0.03)

    decay_df, decay_meta = fa.simulate_saxs_curve(
        shape="cube",
        q_min=0.01,
        q_max=0.2,
        n_points=300,
        radius=50.0,
        scale=2.2,
        background_mode="decay",
        background_const=0.01,
        background_decay_amp=0.1,
        background_decay_q0=0.03,
        background_decay_exp=3.0,
        noise_level=0.0,
        seed=124,
    )
    assert decay_meta["shape"] == "cube"
    assert decay_meta["background_mode"] == "decay"
    assert len(decay_df) == 300
    assert float(decay_df["intensity_background"].iloc[0]) > float(decay_df["intensity_background"].iloc[-1])


def test_single_fit_zip_contains_expected_entries(monkeypatch):
    monkeypatch.setenv("NANOORGANIZER_DISABLE_PNG_EXPORT", "1")
    state = _build_demo_fit_state()

    payload = fa.build_fit_zip_bytes(state)
    assert len(payload) > 0

    with zipfile.ZipFile(io.BytesIO(payload), "r") as zf:
        names = set(zf.namelist())
        required = {
            "fitted_curve.csv",
            "fit_parameters.csv",
            "fit_metrics.csv",
            "fit_peaks.csv",
            "fit_arrays.npz",
            "fit_summary.json",
            "fit_plot.html",
            "fit_plot_note.txt",
        }
        assert required.issubset(names)

        summary = json.loads(zf.read("fit_summary.json").decode("utf-8"))
        assert summary["backend"] == "general_peaks"
        assert summary["curve_label"] == "demo_curve"
        assert summary["source_curve_key"] == "curve_001"


def test_batch_zip_and_server_save_outputs(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("NANOORGANIZER_DISABLE_PNG_EXPORT", "1")
    state1 = _build_demo_fit_state()
    state2 = dict(state1)
    state2["source_curve_key"] = "curve_002"
    state2["curve_label"] = "demo_curve_2"

    batch_states = {
        "curve_001": state1,
        "curve_002": state2,
    }
    payload = fa.build_batch_fit_zip_bytes(batch_states)
    assert len(payload) > 0

    with zipfile.ZipFile(io.BytesIO(payload), "r") as zf:
        names = set(zf.namelist())
        assert "batch_summary.csv" in names
        assert "001_demo_curve/fitted_curve.csv" in names
        assert "002_demo_curve_2/fitted_curve.csv" in names
        assert "001_demo_curve/fit_plot.html" in names

    single_saved = fa.save_single_fit_to_server(state1, export_root=tmp_path)
    assert single_saved["run_dir"].exists()
    assert single_saved["zip_path"].exists()
    assert single_saved["npz_path"].exists()
    assert single_saved["png_path"] is None

    summary_rows = [
        {
            "curve_key": "curve_001",
            "curve": "demo_curve",
            "status": "success",
            "message": "",
            "shape": "Gaussian",
            "r2": 1.0,
            "rmse": 0.0,
        },
        {
            "curve_key": "curve_002",
            "curve": "demo_curve_2",
            "status": "success",
            "message": "",
            "shape": "Gaussian",
            "r2": 1.0,
            "rmse": 0.0,
        },
    ]
    batch_saved = fa.save_batch_fit_to_server(batch_states, summary_rows, export_root=tmp_path)
    assert batch_saved["run_dir"].exists()
    assert batch_saved["zip_path"].exists()
    assert batch_saved["summary_path"].exists()
    assert len(batch_saved["curve_dirs"]) == 2
