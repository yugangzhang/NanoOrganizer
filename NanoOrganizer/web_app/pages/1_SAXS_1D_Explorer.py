#!/usr/bin/env python3
"""SAXS 1-D Explorer — browse & overlay circular-average I(q) curves.

Point it at a ``cir_avg/`` folder of ``Cir_Avg_*.csv`` files (the CMS
auto-reduction output) and explore the 1-D data by:

* **filename** — pick individual files from the table,
* **keywords** — AND / OR / NOT substring filters on the filename,
* **time**     — slice by the acquisition timestamp parsed from the name.

Each CSV has columns ``q_ca, iq_ca`` (a leading unnamed index column is
tolerated). Curves are overlaid with Plotly; log/linear axes, per-curve or
time-colored, optional waterfall offset, and CSV/PNG export.

Runs as a page of the NanoOrganizer web app (auto-discovered from
``web_app/pages/``) or standalone::

    streamlit run NanoOrganizer/web_app/pages/1_SAXS_1D_Explorer.py
"""

from __future__ import annotations

import io
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Optional integration with the app's security / filter helpers. Fall back to
# permissive local versions so the page also runs standalone.
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
    from NanoOrganizer.web_app.components.folder_browser import filter_file_list
except Exception:  # pragma: no cover - standalone fallback
    def filter_file_list(file_list, and_list=[], or_list=[], no_list=[]):
        out, n_or = [], len(or_list)
        for f in file_list:
            name = Path(f).name
            ok = True
            if and_list and not all(p in name for p in and_list):
                ok = False
            if or_list and sum(p not in name for p in or_list) == n_or:
                ok = False
            if no_list and any(p in name for p in no_list):
                ok = False
            if ok:
                out.append(f)
        return out


DEFAULT_DIR = (
    "/nsls2/users/yuzhang/cms_proposal_link/2026-2/pass-316987/"
    "experiments/1_Flow/saxs/analysis/cir_avg"
)

# ---------------------------------------------------------------------------
# Filename metadata parsing
# ---------------------------------------------------------------------------
# Example: Cir_Avg_SY1_Au-TOP-I2-2_A1_2026_07_03_02_39_51_10.00s_2416749_000000_saxs.tiff.csv
_TS_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})")
_EXP_RE = re.compile(r"_(\d+\.\d+)s_")
_SCAN_RE = re.compile(r"_(\d{6,})_\d{6}_")
_WELL_RE = re.compile(r"_([A-H]\d{1,2})_(?=\d{4}_\d{2}_\d{2})")


def parse_name(fname: str) -> dict:
    """Best-effort metadata from a Cir_Avg filename."""
    name = Path(fname).name
    stem = name
    for pref in ("Cir_Avg_",):
        if stem.startswith(pref):
            stem = stem[len(pref):]
    stem = re.sub(r"\.tiff\.csv$|\.csv$", "", stem)

    ts = None
    m = _TS_RE.search(name)
    if m:
        try:
            ts = datetime(*[int(x) for x in m.groups()])
        except ValueError:
            ts = None

    exp = None
    m = _EXP_RE.search(name)
    if m:
        exp = float(m.group(1))

    scan = None
    m = _SCAN_RE.search(name)
    if m:
        scan = int(m.group(1))

    well = None
    m = _WELL_RE.search(name)
    if m:
        well = m.group(1)

    # sample = everything before the well or timestamp token
    sample = stem
    if well:
        sample = stem.split(f"_{well}_")[0]
    elif m := _TS_RE.search(stem):
        sample = stem[:m.start()].rstrip("_")

    is_cal = bool(re.match(r"(AgBH|DirBeam|Empty|glassy|GC)", stem, re.I))

    return dict(
        file=fname, name=name, label=stem, sample=sample, well=well,
        timestamp=ts, exposure_s=exp, scan_id=scan, is_calibration=is_cal,
    )


@st.cache_data(show_spinner=False)
def scan_folder(folder: str, recursive: bool) -> pd.DataFrame:
    base = Path(folder)
    if not base.is_dir():
        return pd.DataFrame()
    globber = base.rglob if recursive else base.glob
    files = sorted(str(p) for p in globber("*.csv"))
    rows = [parse_name(f) for f in files]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            by=[c for c in ("timestamp", "scan_id", "name") if c in df],
            na_position="last",
        ).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_curve(fpath: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (q, I) from a Cir_Avg CSV. Tolerates a leading index column."""
    df = pd.read_csv(fpath)
    cols = {c.lower(): c for c in df.columns}
    qcol = cols.get("q_ca") or cols.get("q") or df.columns[-2]
    icol = cols.get("iq_ca") or cols.get("intensity") or cols.get("i") or df.columns[-1]
    q = df[qcol].to_numpy(dtype=float)
    iq = df[icol].to_numpy(dtype=float)
    return q, iq


def fig_to_png_bytes(fig) -> bytes | None:
    try:
        return fig.to_image(format="png", scale=2)  # needs kaleido
    except Exception:
        return None


# ===========================================================================
# Page
# ===========================================================================
st.set_page_config(page_title="SAXS 1-D Explorer", page_icon="📈", layout="wide")
initialize_security_context()
require_authentication()

st.title("📈 SAXS 1-D Explorer")
st.caption("Overlay circular-average I(q) curves — filter by filename, keyword, or time.")

# --- Sidebar: data source + filters ---------------------------------------
with st.sidebar:
    st.header("📁 Data source")
    folder = st.text_input("cir_avg folder", value=DEFAULT_DIR)
    recursive = st.checkbox("Search subfolders", value=False)
    if st.button("🔄 Rescan folder"):
        scan_folder.clear()

    if _HAVE_SECURITY and folder and not is_path_allowed(folder, allow_nonexistent=True):
        st.error("This folder is outside the allowed roots (secure mode).")
        st.stop()

    df = scan_folder(folder, recursive)
    if df.empty:
        st.warning("No CSV files found. Check the folder path.")
        st.stop()
    st.success(f"{len(df)} files found.")

    st.header("🔎 Filter")
    mode = st.radio(
        "Select by", ["Keywords", "Time range", "Pick files"], index=0,
    )
    hide_cal = st.checkbox("Hide calibration (AgBH/DirBeam)", value=True)


work = df.copy()
if hide_cal:
    work = work[~work["is_calibration"]].reset_index(drop=True)
if work.empty:
    st.warning("Everything filtered out. Uncheck 'Hide calibration' or widen the filter.")
    st.stop()

# --- Apply the chosen filter mode ------------------------------------------
selected = work

if mode == "Keywords":
    c1, c2, c3 = st.columns(3)
    and_s = c1.text_input("Contains ALL (comma-sep)", value="")
    or_s = c2.text_input("Contains ANY (comma-sep)", value="")
    no_s = c3.text_input("Excludes (comma-sep)", value="")
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]
    keep = set(filter_file_list(
        work["file"].tolist(),
        and_list=split(and_s), or_list=split(or_s), no_list=split(no_s),
    ))
    selected = work[work["file"].isin(keep)].reset_index(drop=True)

elif mode == "Time range":
    tdf = work[work["timestamp"].notna()].reset_index(drop=True)
    if tdf.empty:
        st.warning("No parseable timestamps in these filenames.")
        st.stop()
    tmin, tmax = tdf["timestamp"].min().to_pydatetime(), tdf["timestamp"].max().to_pydatetime()
    if tmin == tmax:
        st.info(f"All files share one timestamp: {tmin}")
        selected = tdf
    else:
        lo, hi = st.slider(
            "Acquisition time window",
            min_value=tmin, max_value=tmax, value=(tmin, tmax), format="MM-DD HH:mm:ss",
        )
        selected = tdf[(tdf["timestamp"] >= lo) & (tdf["timestamp"] <= hi)].reset_index(drop=True)

else:  # Pick files
    samples = ["(all)"] + sorted(work["sample"].dropna().unique().tolist())
    pick_sample = st.selectbox("Sample", samples, index=0)
    pool = work if pick_sample == "(all)" else work[work["sample"] == pick_sample]
    default = pool["name"].tolist()[:8]
    chosen = st.multiselect(
        "Files to plot", options=pool["name"].tolist(), default=default,
    )
    selected = pool[pool["name"].isin(chosen)].reset_index(drop=True)

st.markdown(f"**{len(selected)} curve(s) selected** of {len(work)} (calibration hidden: {hide_cal}).")
if selected.empty:
    st.stop()

# --- Plot controls ---------------------------------------------------------
# Compact top row: scales, coloring, stacking, cap.
pc1, pc2, pc3, pc4, pc5 = st.columns(5)
logx = pc1.checkbox("log q", value=True)
logy = pc2.checkbox("log I", value=True)
color_by = pc3.selectbox("Color by", ["time", "index", "sample", "manual"], index=0,
                         help="'manual' unlocks per-curve color/marker/style below.")
waterfall = pc4.number_input("Waterfall ×", value=1.0, min_value=1.0, step=0.5,
                             help="Multiply each successive curve by this factor to stack them.")
max_curves = pc5.number_input("Max curves", value=60, min_value=1, step=10)

if len(selected) > max_curves:
    st.info(f"Showing first {int(max_curves)} of {len(selected)} curves (raise 'Max curves' to see more).")
    selected = selected.iloc[: int(max_curves)].reset_index(drop=True)

# --- Axis ranges + labels (xlim / ylim / titles) ---------------------------
with st.expander("📐 Axis ranges & labels", expanded=False):
    ac1, ac2, ac3, ac4 = st.columns(4)
    xmin = ac1.text_input("q min", value="", help="blank = auto")
    xmax = ac2.text_input("q max", value="")
    ymin = ac3.text_input("I min", value="", help="blank = auto (see 'auto I' below)")
    ymax = ac4.text_input("I max", value="")
    auto_y = st.checkbox(
        "Auto I range from visible q-window", value=True,
        help="Rescale I to the data inside [q min, q max]. Overridden by any "
             "I min / I max you type above.",
    )
    lc1, lc2, lc3 = st.columns(3)
    plot_title = lc1.text_input("Title", value="SAXS I(q)")
    x_label = lc2.text_input("X label", value="q (Å⁻¹)")
    y_label = lc3.text_input("Y label", value="I(q)")
    gc1, gc2, gc3 = st.columns(3)
    show_grid = gc1.checkbox("Grid", value=True)
    show_legend = gc2.checkbox("Legend", value=(len(selected) <= 20))
    marker_mode = gc3.selectbox("Draw as", ["lines", "lines+markers", "markers"], index=0)

def _f(txt):
    try:
        return float(txt)
    except (TypeError, ValueError):
        return None

# For a log axis Plotly wants log10 of the bound.
def _axrange(lo, hi, is_log):
    lo, hi = _f(lo), _f(hi)
    if lo is None and hi is None:
        return None
    if is_log:
        import math
        lo = math.log10(lo) if (lo and lo > 0) else None
        hi = math.log10(hi) if (hi and hi > 0) else None
    return [lo, hi] if (lo is not None and hi is not None) else None

# --- Global marker/line defaults (apply to all curves) ---------------------
import plotly.express as px

MARKERS = ["circle", "square", "triangle-up", "triangle-down", "diamond",
           "cross", "x", "star", "hexagon", "pentagon"]
DASHES = {"solid": "solid", "dash": "dash", "dot": "dot", "dashdot": "dashdot"}

with st.expander("🎨 Global style (line width / marker size / opacity)", expanded=False):
    sc1, sc2, sc3, sc4 = st.columns(4)
    g_lw = sc1.slider("Line width", 0.5, 6.0, 1.6, 0.1)
    g_ms = sc2.slider("Marker size", 2, 16, 6, 1)
    g_alpha = sc3.slider("Opacity", 0.1, 1.0, 0.9, 0.05)
    g_dash = sc4.selectbox("Line style", list(DASHES.keys()), index=0)

# --- Coloring: auto colormap vs. per-curve manual --------------------------
times = selected["timestamp"]
manual = (color_by == "manual")
per_curve = {}   # file -> dict(color, marker, dash, lw, ms, alpha)

if manual:
    st.caption("Per-curve styling — expand each to override color / marker / line / width / opacity.")
    NAMED = {"Blue": "#1f77b4", "Orange": "#ff7f0e", "Green": "#2ca02c",
             "Red": "#d62728", "Purple": "#9467bd", "Brown": "#8c564b",
             "Pink": "#e377c2", "Gray": "#7f7f7f", "Olive": "#bcbd22",
             "Cyan": "#17becf", "Black": "#000000", "Magenta": "#FF00FF"}
    names = list(NAMED.keys())
    for i, row in enumerate(selected.itertuples()):
        with st.expander(f"🔧 {row.label}", expanded=(len(selected) <= 4)):
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            cname = m1.selectbox("Color", names, index=i % len(names), key=f"c_{row.file}")
            mk = m2.selectbox("Marker", ["none"] + MARKERS, index=0, key=f"m_{row.file}")
            ds = m3.selectbox("Line", ["none"] + list(DASHES.keys()), index=1, key=f"d_{row.file}")
            lw = m4.slider("Width", 0.5, 6.0, g_lw, 0.1, key=f"lw_{row.file}")
            ms = m5.slider("Size", 2, 16, g_ms, 1, key=f"ms_{row.file}")
            al = m6.slider("Opacity", 0.1, 1.0, g_alpha, 0.05, key=f"al_{row.file}")
            per_curve[row.file] = dict(color=NAMED[cname], marker=mk, dash=ds,
                                       lw=lw, ms=ms, alpha=al)
    cvals = list(range(len(selected)))
    cbar_title = None
else:
    if color_by == "time" and times.notna().any():
        t0 = times.min()
        cvals = [(t - t0).total_seconds() if pd.notna(t) else None for t in times]
        if all(v is None for v in cvals):
            cvals = list(range(len(selected)))
        cbar_title = "t − t₀ (s)"
    elif color_by == "sample":
        cats = {s: i for i, s in enumerate(sorted(selected["sample"].unique()))}
        cvals = [cats[s] for s in selected["sample"]]
        cbar_title = "sample #"
    else:
        cvals = list(range(len(selected)))
        cbar_title = "index"

_valid = [v for v in cvals if v is not None]
vmin, vmax = (min(_valid), max(_valid)) if _valid else (0, 1)
span = (vmax - vmin) or 1.0
palette = px.colors.sample_colorscale(
    "Turbo", [((v - vmin) / span) if v is not None else 0.0 for v in cvals]
)

def _rgba(hexc, alpha):
    hexc = hexc.lstrip("#")
    if len(hexc) != 6:
        return hexc
    r, g, b = (int(hexc[j:j + 2], 16) for j in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

# --- Build figure ----------------------------------------------------------
# q-window used both for the x-axis and (when auto_y) to pick the I range.
q_lo, q_hi = _f(xmin), _f(xmax)
y_in_window = []   # collect plotted I values that fall inside [q_lo, q_hi]

fig = go.Figure()
for i, (row, auto_col) in enumerate(zip(selected.itertuples(), palette)):
    q, iq = load_curve(row.file)
    scale = waterfall ** i if waterfall > 1.0 else 1.0
    y = iq * scale
    ts = row.timestamp.strftime("%m-%d %H:%M:%S") if pd.notna(row.timestamp) else "—"

    # Gather y-values within the visible q-window for auto I-scaling.
    m = np.isfinite(q) & np.isfinite(y)
    if q_lo is not None:
        m &= q >= q_lo
    if q_hi is not None:
        m &= q <= q_hi
    if logy:
        m &= y > 0     # log axis: ignore non-positive
    if m.any():
        y_in_window.append(y[m])

    if manual:
        s = per_curve[row.file]
        color, alpha, lw, ms = s["color"], s["alpha"], s["lw"], s["ms"]
        has_line = s["dash"] != "none"
        has_mark = s["marker"] != "none"
        mode = "lines+markers" if (has_line and has_mark) else ("markers" if has_mark else "lines")
        dash = DASHES.get(s["dash"], "solid")
        msym = s["marker"] if has_mark else "circle"
    else:
        color, alpha, lw, ms = auto_col, g_alpha, g_lw, g_ms
        mode = marker_mode
        dash = DASHES[g_dash]
        msym = MARKERS[i % len(MARKERS)]

    fig.add_trace(go.Scatter(
        x=q, y=y, mode=mode, name=row.label,
        line=dict(color=color, width=lw, dash=dash),
        marker=dict(color=color, size=ms, symbol=msym),
        opacity=alpha,
        hovertemplate=(
            f"<b>{row.label}</b><br>well={row.well} t={ts}"
            "<br>q=%{x:.4f} Å⁻¹<br>I=%{y:.3g}<extra></extra>"
        ),
        showlegend=show_legend,
    ))

xr = _axrange(xmin, xmax, logx)
yr = _axrange(ymin, ymax, logy)

# Auto I-range: fit the data inside the visible q-window, unless the user
# typed explicit I min/max (those always win).
if auto_y and (ymin.strip() == "" and ymax.strip() == "") and y_in_window:
    allv = np.concatenate(y_in_window)
    lo_v, hi_v = float(np.nanmin(allv)), float(np.nanmax(allv))
    if np.isfinite(lo_v) and np.isfinite(hi_v) and hi_v > lo_v:
        import math
        if logy:
            lo_l, hi_l = math.log10(lo_v), math.log10(hi_v)
            pad = 0.05 * (hi_l - lo_l) or 0.1
            yr = [lo_l - pad, hi_l + pad]
        else:
            pad = 0.05 * (hi_v - lo_v)
            yr = [lo_v - pad, hi_v + pad]
fig.update_layout(
    height=650, template="plotly_white", title=plot_title,
    xaxis_title=x_label,
    yaxis_title=y_label + (" × waterfall" if waterfall > 1 else ""),
    xaxis=dict(type="log" if logx else "linear", range=xr, showgrid=show_grid),
    yaxis=dict(type="log" if logy else "linear", range=yr, showgrid=show_grid),
    margin=dict(l=60, r=20, t=45, b=50),
    legend=dict(font=dict(size=9)),
)
# color reference when a colormap is used and legend is hidden
if (not manual) and (not show_legend) and _valid and cbar_title:
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(colorscale="Turbo", cmin=vmin, cmax=vmax, color=[vmin],
                    colorbar=dict(title=cbar_title), showscale=True),
        hoverinfo="none", showlegend=False,
    ))

st.plotly_chart(fig, use_container_width=True)

# --- Details table + exports -----------------------------------------------
with st.expander("📋 Selected files / metadata", expanded=False):
    show_cols = ["name", "sample", "well", "timestamp", "exposure_s", "scan_id"]
    st.dataframe(selected[show_cols], use_container_width=True, hide_index=True)

ec1, ec2 = st.columns(2)

# combined long-form CSV of all plotted curves
buf = io.StringIO()
frames = []
for row in selected.itertuples():
    q, iq = load_curve(row.file)
    frames.append(pd.DataFrame({"label": row.label, "q": q, "I": iq}))
pd.concat(frames, ignore_index=True).to_csv(buf, index=False)
ec1.download_button(
    "⬇️ Download combined CSV (long form)",
    data=buf.getvalue(), file_name="saxs_1d_selection.csv", mime="text/csv",
)

png = fig_to_png_bytes(fig)
if png:
    ec2.download_button("⬇️ Download plot PNG", data=png,
                        file_name="saxs_1d_plot.png", mime="image/png")
else:
    ec2.caption("PNG export needs `kaleido` (`pip install kaleido`). Use the plot's camera icon meanwhile.")
