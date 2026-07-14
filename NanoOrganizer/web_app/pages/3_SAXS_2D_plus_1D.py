#!/usr/bin/env python3
"""SAXS 2D + 1D combo — q–φ map beside its circular average.

For each reduced frame the auto-pipeline writes:

* ``qphi/qphi_<name>.npz``   → keys ``q (n_q,)``, ``phi (n_phi,)``,
  ``qphi (n_phi, n_q)``, ``qphi_mask (n_phi, n_q)`` — the 2D map.
* ``cir_avg/Cir_Avg_<name>.csv`` → columns ``q_ca, iq_ca`` — the 1D curve.

This page lets you pick a frame (by filename / keyword / time), shows the
q–φ heatmap and the I(q) curve side-by-side, and can overlay several 1D
curves for comparison. It auto-pairs the two folders by the shared
``<name>`` stem.

Run as an app page, or standalone::

    streamlit run NanoOrganizer/web_app/pages/3_SAXS_2D_plus_1D.py
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from NanoOrganizer.web_app.components.security import (
        initialize_security_context, require_authentication, is_path_allowed,
    )
    _HAVE_SECURITY = True
except Exception:  # pragma: no cover
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


DEFAULT_ANALYSIS = (
    "/nsls2/users/yuzhang/cms_proposal_link/2026-2/pass-316987/"
    "experiments/1_Flow/saxs/analysis"
)

_TS_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})")
_WELL_RE = re.compile(r"_([A-H]\d{1,2})_(?=\d{4}_\d{2}_\d{2})")


def stem_of(fname: str) -> str:
    """Shared <name> stem: strip Cir_Avg_/qphi_ prefixes and extensions."""
    s = Path(fname).name
    for pref in ("Cir_Avg_", "qphi_", "qimg_"):
        if s.startswith(pref):
            s = s[len(pref):]
    return re.sub(r"\.npz$|\.tiff\.csv$|\.csv$|\.tiff$", "", s)


def parse_meta(stem: str) -> dict:
    ts = None
    m = _TS_RE.search(stem)
    if m:
        try:
            ts = datetime(*[int(x) for x in m.groups()])
        except ValueError:
            ts = None
    well = None
    m = _WELL_RE.search(stem)
    if m:
        well = m.group(1)
    is_cal = bool(re.match(r"(AgBH|DirBeam|Empty|glassy|GC)", stem, re.I))
    return dict(timestamp=ts, well=well, is_calibration=is_cal)


@st.cache_data(show_spinner=False)
def index_frames(analysis_dir: str) -> pd.DataFrame:
    base = Path(analysis_dir)
    cir_dir, qphi_dir = base / "cir_avg", base / "qphi"
    cir = {stem_of(p.name): str(p) for p in cir_dir.glob("*.csv")} if cir_dir.is_dir() else {}
    qphi = {stem_of(p.name): str(p) for p in qphi_dir.glob("*.npz")} if qphi_dir.is_dir() else {}
    stems = sorted(set(cir) | set(qphi))
    rows = []
    for s in stems:
        meta = parse_meta(s)
        rows.append(dict(stem=s, label=s, cir=cir.get(s), qphi=qphi.get(s),
                         has_2d=s in qphi, has_1d=s in cir, **meta))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=[c for c in ("timestamp", "stem") if c in df],
                            na_position="last").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_1d(fpath: str):
    df = pd.read_csv(fpath)
    cols = {c.lower(): c for c in df.columns}
    qcol = cols.get("q_ca") or cols.get("q") or df.columns[-2]
    icol = cols.get("iq_ca") or cols.get("intensity") or df.columns[-1]
    return df[qcol].to_numpy(float), df[icol].to_numpy(float)


@st.cache_data(show_spinner=False)
def load_2d(fpath: str):
    d = np.load(fpath)
    q = d["q"] if "q" in d else None
    phi = d["phi"] if "phi" in d else None
    qphi = d["qphi"] if "qphi" in d else None
    mask = d["qphi_mask"] if "qphi_mask" in d else None
    return q, phi, qphi, mask


# ===========================================================================
st.set_page_config(page_title="SAXS 2D + 1D", page_icon="🗺️", layout="wide")
initialize_security_context()
require_authentication()

st.title("🗺️ SAXS 2D + 1D")
st.caption("q–φ map beside its circular average, auto-paired by filename.")

with st.sidebar:
    st.header("📁 Analysis folder")
    if _HAVE_BROWSER:
        analysis = folder_picker(key="saxs2d_analysis",
                                 label="analysis/ dir (has cir_avg/ + qphi/)",
                                 default=DEFAULT_ANALYSIS)
    else:
        analysis = st.text_input("analysis/ dir (has cir_avg/ + qphi/)",
                                 value=DEFAULT_ANALYSIS)
        if _HAVE_SECURITY and analysis and not is_path_allowed(
            analysis, allow_nonexistent=True
        ):
            st.error("Folder outside allowed roots (secure mode).")
            st.stop()
    if st.button("🔄 Rescan"):
        index_frames.clear()
    if not analysis:
        st.stop()

    df = index_frames(analysis)
    if df.empty:
        st.warning("No cir_avg/ or qphi/ files found under this folder.")
        st.stop()
    n2d, n1d = int(df["has_2d"].sum()), int(df["has_1d"].sum())
    st.success(f"{len(df)} frames — {n2d} with 2D, {n1d} with 1D.")

    hide_cal = st.checkbox("Hide calibration", value=True)
    kw = st.text_input("Filter by keyword(s), comma-sep", value="",
                       help="AND filter on the filename stem.")

work = df.copy()
if hide_cal:
    work = work[~work["is_calibration"]]
if kw.strip():
    for tok in [k.strip() for k in kw.split(",") if k.strip()]:
        work = work[work["stem"].str.contains(re.escape(tok))]
work = work.reset_index(drop=True)
if work.empty:
    st.warning("Nothing matches the filter.")
    st.stop()

# --- Frame picker (by time order or name) ----------------------------------
c1, c2 = st.columns([3, 1])
labels = work["stem"].tolist()
if len(labels) > 1:
    chosen_label = c1.selectbox("Frame", options=labels, index=0)
    idx = labels.index(chosen_label)
else:
    idx = 0
sel = work.iloc[int(idx)]
c2.metric("Frame", f"{int(idx) + 1}/{len(labels)}")

ts = sel["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if pd.notna(sel["timestamp"]) else "—"
st.markdown(f"**{sel['stem']}**  ·  well `{sel['well']}`  ·  t = {ts}")

# --- Display controls ------------------------------------------------------
dc1, dc2, dc3, dc4 = st.columns(4)
logI = dc1.checkbox("log I (2D)", value=True)
logq = dc2.checkbox("log q (1D)", value=True)
logiq = dc3.checkbox("log I (1D)", value=True)
cmap = dc4.selectbox("2D colormap", ["Viridis", "Turbo", "Inferno", "Jet", "Cividis"], index=0)

overlay_current = st.checkbox("Overlay 1D curves from other filtered frames", value=False)

# --- Build side-by-side figure ---------------------------------------------
fig = make_subplots(rows=1, cols=2, column_widths=[0.52, 0.48],
                    subplot_titles=("q–φ map", "I(q) circular average"),
                    horizontal_spacing=0.09)

# left: 2D q-phi
if sel["has_2d"]:
    q, phi, qphi, mask = load_2d(sel["qphi"])
    if qphi is not None:
        z = qphi.astype(float).copy()
        if mask is not None:
            z[~mask.astype(bool)] = np.nan if mask.dtype == bool else z[~mask.astype(bool)]
        if logI:
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.log10(np.clip(z, 1e-6, None))
        fig.add_trace(go.Heatmap(
            z=z, x=q, y=phi, colorscale=cmap, colorbar=dict(title="log I" if logI else "I", x=0.46),
            hovertemplate="q=%{x:.4f}<br>φ=%{y:.1f}°<br>I=%{z:.3g}<extra></extra>",
        ), row=1, col=1)
        fig.update_xaxes(title_text="q (Å⁻¹)", type="log" if logq else "linear", row=1, col=1)
        fig.update_yaxes(title_text="φ (deg)", row=1, col=1)
else:
    fig.add_annotation(text="no 2D (qphi) for this frame", xref="x1", yref="y1",
                       x=0.5, y=0.5, showarrow=False)

# right: 1D curve(s)
if overlay_current:
    import plotly.express as px
    sub = work[work["has_1d"]]
    cols = px.colors.sample_colorscale("Turbo", np.linspace(0, 1, max(1, len(sub))))
    for (row, col) in zip(sub.itertuples(), cols):
        qq, ii = load_1d(row.cir)
        fig.add_trace(go.Scatter(x=qq, y=ii, mode="lines", line=dict(width=1, color=col),
                                 name=row.stem, showlegend=False,
                                 hovertemplate=f"{row.stem}<br>q=%{{x:.4f}}<br>I=%{{y:.3g}}<extra></extra>"),
                      row=1, col=2)
if sel["has_1d"]:
    qq, ii = load_1d(sel["cir"])
    fig.add_trace(go.Scatter(x=qq, y=ii, mode="lines", line=dict(width=2.4, color="crimson"),
                             name=sel["stem"],
                             hovertemplate="q=%{x:.4f}<br>I=%{y:.3g}<extra></extra>"),
                  row=1, col=2)
fig.update_xaxes(title_text="q (Å⁻¹)", type="log" if logq else "linear", row=1, col=2)
fig.update_yaxes(title_text="I(q)", type="log" if logiq else "linear", row=1, col=2)
fig.update_layout(height=560, template="plotly_white", showlegend=False,
                  margin=dict(l=60, r=20, t=50, b=50))

st.plotly_chart(fig, use_container_width=True)

with st.expander("📋 Frame table", expanded=False):
    st.dataframe(work[["stem", "well", "timestamp", "has_2d", "has_1d"]],
                 use_container_width=True, hide_index=True)
