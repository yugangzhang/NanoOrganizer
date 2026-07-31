#!/usr/bin/env python3
"""UV-Vis 1-D Explorer — browse & organize fitted SPR spectra by recipe / quality.

Point it at a *UV-Vis explorer dataset* produced by
``UV_Vis.droplet_recognition.write_explorer_dataset`` — a pair of files:

* ``uvvis_explorer_manifest.parquet`` — one row per recipe: identity
  (``planned, run_num``), conditions (``NCit, acid, HAu``), SPR fit features
  (``np_center, np_fwhm, np_amp, np_area, peak_abs``) and quality
  (``r2, snr, reliability, fit_flag, tier``) plus the ``1L-quad`` fit parameters
  (``np_amp_p, np_center_p, np_fwhm_p, b0, b1, b2``) and a ``row`` index into …
* ``uvvis_explorer_spectra.npz`` — ``wavelength`` (W,) + ``absorbance`` (N, W),
  row-aligned to the manifest.

Unlike the SAXS explorer (which parses filenames), this browses on **rich scalar
metadata**: filter by quality *tier*, *fit_flag*, run, reliability threshold, and
by ranges on any condition / feature; overlay the raw spectra coloured by any
column; and (optionally) overlay each recipe's ``1L-quad`` fit, reconstructed
analytically from the stored parameters — no refit, no per-recipe files.

Runs as a page of the NanoOrganizer web app (auto-discovered from
``web_app/pages/``) or standalone::

    streamlit run NanoOrganizer/web_app/pages/9_UVVis_1D_Explorer.py
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- access gate (owner-only in secure mode; open locally) -----------------
try:
    from NanoOrganizer.web_app.components.uvvis_access import require_uvvis_owner
except Exception:  # pragma: no cover - standalone fallback
    def require_uvvis_owner():
        return None

# ---------------------------------------------------------------------------
# Optional integration with the app's security helpers (permissive standalone).
# ---------------------------------------------------------------------------
try:
    from NanoOrganizer.web_app.components.security import (
        initialize_security_context,
        require_authentication,
        is_path_allowed,
    )
    _HAVE_SECURITY = True
except Exception:  # pragma: no cover - standalone fallback
    _HAVE_SECURITY = False

    def initialize_security_context():
        return None

    def require_authentication():
        return None

    def is_path_allowed(path, allow_nonexistent: bool = False):
        return True

try:
    from NanoOrganizer.web_app.components.folder_browser import folder_picker
    _HAVE_BROWSER = True
except Exception:  # pragma: no cover - standalone fallback
    _HAVE_BROWSER = False


# Default dataset location (mGpBo droplet-recognition results).
DEFAULT_DIR = (
    "/home/yuzhang/Repos/mGpBo/results/600_Droplet_Recognition/nanoorganizer_uvvis"
)

# Baseline centering constants — MUST match
# UV_Vis.droplet_recognition.fitting_models (BASELINE_X0 / BASELINE_SCALE).
BASELINE_X0 = 600.0
BASELINE_SCALE = 100.0

#: Columns offered as colour / filter axes (numeric features) when present.
FEATURE_COLS = ["np_center", "np_fwhm", "np_amp", "np_area", "peak_abs",
                "r2", "rmse", "snr", "reliability", "NCit", "acid", "HAu"]
TIER_ORDER = ["excellent", "good", "marginal", "poor", "failed"]


# ---------------------------------------------------------------------------
# Dataset IO (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset(folder: str):
    """Load (manifest DataFrame, wavelength, absorbance matrix) from a folder."""
    base = Path(folder)
    man_p = base / "uvvis_explorer_manifest.parquet"
    spec_p = base / "uvvis_explorer_spectra.npz"
    if not man_p.exists() or not spec_p.exists():
        return None, None, None
    man = pd.read_parquet(man_p)
    with np.load(spec_p) as d:
        wl = d["wavelength"]
        absb = d["absorbance"]
    return man, wl, absb


def reconstruct_1L_quad(wl, amp, center, fwhm, b0, b1, b2):
    """Analytic ``1L-quad`` fit curve from stored parameters (peak + baseline)."""
    wl = np.asarray(wl, dtype=float)
    half = fwhm / 2.0
    peak = amp * half ** 2 / ((wl - center) ** 2 + half ** 2)
    u = (wl - BASELINE_X0) / BASELINE_SCALE
    return peak + (b0 + b1 * u + b2 * u * u)


def fig_to_png_bytes(fig):
    try:
        return fig.to_image(format="png", scale=2)  # needs kaleido
    except Exception:
        return None


# ===========================================================================
# Page
# ===========================================================================
st.set_page_config(page_title="UV-Vis 1-D Explorer", page_icon="🧫", layout="wide")
initialize_security_context()
require_authentication()
require_uvvis_owner()

st.title("🧫 UV-Vis 1-D Explorer")
st.caption("Browse fitted SPR spectra by recipe & fit quality — overlay raw data "
           "and the 1L-quad fit, filter by tier / conditions / features.")

# --- Sidebar: data source --------------------------------------------------
with st.sidebar:
    st.header("📁 Dataset")
    if _HAVE_BROWSER:
        folder = folder_picker(key="uvvis1d_folder",
                               label="explorer dataset folder", default=DEFAULT_DIR)
    else:
        folder = st.text_input("explorer dataset folder", value=DEFAULT_DIR)
        if _HAVE_SECURITY and folder and not is_path_allowed(folder, allow_nonexistent=True):
            st.error("This folder is outside the allowed roots (secure mode).")
            st.stop()
    if st.button("🔄 Reload dataset"):
        load_dataset.clear()
    if not folder:
        st.stop()

man, wl, absb = load_dataset(folder)
if man is None:
    st.warning("No `uvvis_explorer_manifest.parquet` + `uvvis_explorer_spectra.npz` "
               "found in that folder. Generate them with "
               "`write_explorer_dataset(fdf, combined, out_dir)`.")
    st.stop()

st.sidebar.success(f"{len(man)} recipes loaded.")

# --- Sidebar: filters ------------------------------------------------------
with st.sidebar:
    st.header("🔎 Filters")
    work = man.copy()

    if "tier" in work:
        tiers = [t for t in TIER_ORDER if t in set(work["tier"])]
        pick_tiers = st.multiselect("Quality tier", tiers, default=tiers)
        work = work[work["tier"].isin(pick_tiers)]

    if "fit_flag" in work:
        flags = sorted(set(man["fit_flag"]))
        pick_flags = st.multiselect("Fit flag", flags, default=flags)
        work = work[work["fit_flag"].isin(pick_flags)]

    if "run_num" in work:
        runs = sorted(set(man["run_num"]))
        pick_runs = st.multiselect("Run", runs, default=runs)
        work = work[work["run_num"].isin(pick_runs)]

    if "reliability" in work and man["reliability"].notna().any():
        rmin = st.slider("Min reliability", 0.0, 1.0, 0.0, 0.05)
        work = work[work["reliability"].fillna(0.0) >= rmin]

    only_success = st.checkbox("Successful fits only", value=True)
    if only_success and "success" in work:
        work = work[work["success"]]

    # numeric range filters on conditions / features
    with st.expander("📐 Range filters (conditions & features)", expanded=False):
        range_cols = [c for c in ("NCit", "acid", "HAu", "np_center", "np_fwhm",
                                   "np_amp", "peak_abs", "r2") if c in man]
        for c in st.multiselect("Add range filter on", range_cols, default=[]):
            col = man[c].astype(float)
            lo, hi = float(np.nanmin(col)), float(np.nanmax(col))
            if lo < hi:
                a, b = st.slider(c, lo, hi, (lo, hi))
                work = work[(work[c] >= a) & (work[c] <= b)]

st.markdown(f"**{len(work)} recipe(s)** match the filters (of {len(man)}).")
if work.empty:
    st.info("Everything filtered out — widen a filter in the sidebar.")
    st.stop()

# --- Plot controls ---------------------------------------------------------
pc1, pc2, pc3, pc4, pc5 = st.columns(5)
color_opts = [c for c in (["tier", "fit_flag"] + FEATURE_COLS) if c in work]
color_by = pc1.selectbox("Colour by", color_opts,
                         index=color_opts.index("tier") if "tier" in color_opts else 0)
show_fit = pc2.checkbox("Overlay 1L-quad fit", value=True,
                        help="Reconstructed analytically from the stored parameters.")
show_raw = pc3.checkbox("Show raw data", value=True)
waterfall = pc4.number_input("Waterfall +offset", value=0.0, min_value=0.0, step=0.05,
                             help="Add this × curve-index to stack spectra vertically.")
max_curves = pc5.number_input("Max curves", value=10, min_value=1, step=10)

# stable ordering: by tier (best first) then reliability, so the cap keeps the good ones
sort_cols = [c for c in ("tier", "reliability", "planned") if c in work]
if "tier" in work:
    work = work.assign(_torder=work["tier"].map({t: i for i, t in enumerate(TIER_ORDER)}))
    sort_cols = ["_torder"] + [c for c in ("reliability", "planned") if c in work]
    asc = [True] + [False if c == "reliability" else True for c in sort_cols[1:]]
    work = work.sort_values(sort_cols, ascending=asc)
sel = work.head(int(max_curves)).reset_index(drop=True)
if len(work) > max_curves:
    st.info(f"Showing the top {int(max_curves)} of {len(work)} "
            f"(sorted by tier then reliability — raise 'Max curves' for more).")

with st.expander("📐 Axis ranges & labels", expanded=False):
    ac1, ac2, ac3, ac4 = st.columns(4)
    xmin = ac1.text_input("λ min (nm)", value="450")
    xmax = ac2.text_input("λ max (nm)", value="750")
    ymin = ac3.text_input("A min", value="")
    ymax = ac4.text_input("A max", value="")
    lc1, lc2, lc3 = st.columns(3)
    plot_title = lc1.text_input("Title", value="UV-Vis spectra")
    x_label = lc2.text_input("X label", value="wavelength (nm)")
    y_label = lc3.text_input("Y label", value="absorbance")
    gc1, gc2 = st.columns(2)
    show_grid = gc1.checkbox("Grid", value=True)
    show_legend = gc2.checkbox("Legend", value=(len(sel) <= 20))


def _f(txt):
    try:
        return float(txt)
    except (TypeError, ValueError):
        return None


# --- Colour mapping --------------------------------------------------------
is_cat = color_by in ("tier", "fit_flag")
if is_cat:
    cats = sorted(set(sel[color_by]),
                  key=lambda t: TIER_ORDER.index(t) if t in TIER_ORDER else 99)
    pal = px.colors.qualitative.Safe
    cat_color = {c: pal[i % len(pal)] for i, c in enumerate(cats)}
    cvals = None
else:
    cvals = sel[color_by].to_numpy(float)
    finite = cvals[np.isfinite(cvals)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
    span = (vmax - vmin) or 1.0
    colors = px.colors.sample_colorscale(
        "Turbo", [(v - vmin) / span if np.isfinite(v) else 0.0 for v in cvals])

# --- Build figure ----------------------------------------------------------
lo, hi = _f(xmin) or float(wl.min()), _f(xmax) or float(wl.max())
wmask = (wl >= lo) & (wl <= hi)
xw = wl[wmask]
fig = go.Figure()
seen_cats = set()
for i, r in enumerate(sel.itertuples()):
    row = int(r.row)
    off = waterfall * i
    color = cat_color[getattr(r, color_by)] if is_cat else colors[i]
    label = f"planned={int(r.planned)}"
    tier = getattr(r, "tier", "")
    hov = (f"<b>{label}</b> [{tier}]"
           f"<br>NCit={getattr(r, 'NCit', float('nan')):.0f} "
           f"acid={getattr(r, 'acid', float('nan')):.0f} "
           f"HAu={getattr(r, 'HAu', float('nan')):.0f}"
           f"<br>λ0={getattr(r, 'np_center', float('nan')):.0f}nm "
           f"fwhm={getattr(r, 'np_fwhm', float('nan')):.0f} "
           f"amp={getattr(r, 'np_amp', float('nan')):.3f}"
           f"<br>R²={getattr(r, 'r2', float('nan')):.3f} "
           f"rel={getattr(r, 'reliability', float('nan')):.2f}"
           "<br>λ=%{x:.0f}nm  A=%{y:.4f}<extra></extra>")

    # legend grouping: one entry per category (categorical) or per curve (feature)
    if is_cat:
        cat = getattr(r, color_by)
        show_this = show_legend and (cat not in seen_cats)
        seen_cats.add(cat)
        legname, leggrp = cat, cat
    else:
        show_this = show_legend
        legname, leggrp = label, None

    if show_raw:
        fig.add_trace(go.Scatter(
            x=xw, y=absb[row][wmask] + off, mode="markers", name=legname,
            legendgroup=leggrp, marker=dict(color=color, size=3, opacity=0.55),
            hovertemplate=hov, showlegend=show_this))
    if show_fit and bool(getattr(r, "success", True)):
        yfit = reconstruct_1L_quad(xw, r.np_amp_p, r.np_center_p, r.np_fwhm_p,
                                   r.b0, r.b1, r.b2) + off
        fig.add_trace(go.Scatter(
            x=xw, y=yfit, mode="lines", name=legname,
            legendgroup=leggrp, line=dict(color=color, width=2),
            hovertemplate=hov, showlegend=(show_this and not show_raw)))

yr = None
if _f(ymin) is not None and _f(ymax) is not None:
    yr = [_f(ymin), _f(ymax)]
fig.update_layout(
    height=650, template="plotly_white", title=plot_title,
    xaxis_title=x_label,
    yaxis_title=y_label + (" (+offset)" if waterfall > 0 else ""),
    xaxis=dict(range=[lo, hi], showgrid=show_grid),
    yaxis=dict(range=yr, showgrid=show_grid),
    margin=dict(l=60, r=20, t=45, b=50), legend=dict(font=dict(size=9)),
)
# continuous colour reference when colouring by a feature
if not is_cat:
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(colorscale="Turbo", cmin=vmin, cmax=vmax, color=[vmin],
                    colorbar=dict(title=color_by), showscale=True),
        hoverinfo="none", showlegend=False))

st.plotly_chart(fig, use_container_width=True)

# --- Details table + exports -----------------------------------------------
with st.expander("📋 Selected recipes / metadata", expanded=False):
    show_cols = [c for c in ("planned", "run_num", "tier", "fit_flag", "reliability",
                             "NCit", "acid", "HAu", "np_center", "np_fwhm", "np_amp",
                             "peak_abs", "r2") if c in sel]
    st.dataframe(sel[show_cols], use_container_width=True, hide_index=True)

ec1, ec2 = st.columns(2)
buf = io.StringIO()
sel.drop(columns=[c for c in ("_torder",) if c in sel]).to_csv(buf, index=False)
ec1.download_button("⬇️ Download selected manifest (CSV)", data=buf.getvalue(),
                    file_name="uvvis_selection.csv", mime="text/csv")
png = fig_to_png_bytes(fig)
if png:
    ec2.download_button("⬇️ Download plot PNG", data=png,
                        file_name="uvvis_1d_plot.png", mime="image/png")
else:
    ec2.caption("PNG export needs `kaleido` (`pip install kaleido`). Use the plot's "
                "camera icon meanwhile.")
