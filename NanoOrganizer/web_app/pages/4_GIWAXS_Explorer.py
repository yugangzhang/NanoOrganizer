#!/usr/bin/env python3
"""GIWAXS / GISAXS Explorer — four panels per frame, plus interactive line-cuts.

Point it at a MAXS/GIWAXS ``analysis/`` folder (the CMS auto-reduction output)
that contains, per reduced frame:

* ``stitched/<name>.tiff``            → the stitched raw detector image (A).
* ``q_image/qimg_<name>.tiff.npz``    → keys ``qimg (nz, nx)``, ``qx (nx,)``,
  ``qz (nz,)``, ``qimg_mask (nz, nx)`` — the remeshed q-space image (B).
* ``qphi/qphi_<name>.tiff.npz``       → keys ``q (nq,)``, ``phi (nphi,)``,
  ``qphi (nphi, nq)``, ``qphi_mask`` — the q–φ caking map (C).
* ``cir_avg/Cir_Avg_<name>.tiff.csv`` → columns ``q_ca, iq_ca`` — the circular
  average (D).

Pick a frame (by filename / keyword / time) and the page shows all four panels
at once. You can also take **line-cuts**:

* on the **q_image** — a *qr-cut* (I vs qr at a fixed qz band) or a *qz-cut*
  (I vs qz at a fixed qr band),
* on the **q–φ** map — a *q-cut* (I vs φ at a fixed q band) or a *φ-cut*
  (I vs q at a fixed φ band).

Each cut is defined by one or more **centers** (comma-separated) and a single
band **width**; the integration band is overlaid on the map and the extracted
1-D profiles are plotted together (and exportable as CSV).

Runs as a page of the NanoOrganizer web app (auto-discovered from
``web_app/pages/``) or standalone::

    streamlit run NanoOrganizer/web_app/pages/4_GIWAXS_Explorer.py
"""

from __future__ import annotations

import io
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Optional integration with the app's security helpers. Fall back to permissive
# local versions so the page also runs standalone.
# ---------------------------------------------------------------------------
try:
    from NanoOrganizer.web_app.components.security import (
        initialize_security_context, require_authentication, is_path_allowed,
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


DEFAULT_ANALYSIS = (
    "/nsls2/auto-storage/cms/proposals/2026-2/pass-320306/"
    "experiments/0_Static/maxs/analysis"
)

# Max pixels to keep for the raw-image heatmap; larger images are downsampled.
_RAW_MAX_PIXELS = 500_000

_TS_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})")
_TH_RE = re.compile(r"_th(-?\d+\.\d+)_")
_SCAN_RE = re.compile(r"_(\d{6,})_\d{6}_")
_WELL_RE = re.compile(r"_([A-H]\d{1,2})_(?=\d{4}_\d{2}_\d{2})")


# ---------------------------------------------------------------------------
# Filename ↔ frame indexing
# ---------------------------------------------------------------------------
def stem_of(fname: str) -> str:
    """Shared <name> stem: strip known prefixes / extensions from a file name.

    Handles the mixed conventions used by the auto-reduction, e.g.
    ``stitched/<name>.tiff``, ``q_image/qimg_<name>.tiff.npz`` and
    ``cir_avg/Cir_Avg_<name>.tiff.csv`` all reduce to the same ``<name>``.
    """
    s = Path(fname).name
    for pref in ("Cir_Avg_", "qphi_", "qimg_", "qc_"):
        if s.startswith(pref):
            s = s[len(pref):]
    # Peel trailing extensions repeatedly (e.g. ".tiff.npz" → ".tiff" → "").
    while True:
        new = re.sub(r"\.(npz|csv|png|tiff)$", "", s)
        if new == s:
            return s
        s = new


def parse_meta(stem: str) -> dict:
    ts = None
    m = _TS_RE.search(stem)
    if m:
        try:
            ts = datetime(*[int(x) for x in m.groups()])
        except ValueError:
            ts = None
    th = None
    m = _TH_RE.search(stem)
    if m:
        th = float(m.group(1))
    scan = None
    m = _SCAN_RE.search(stem)
    if m:
        scan = int(m.group(1))
    well = None
    m = _WELL_RE.search(stem)
    if m:
        well = m.group(1)
    is_cal = bool(re.match(r"(AgBH|DirBeam|Empty|glassy|GC)", stem, re.I))
    return dict(timestamp=ts, th=th, scan=scan, well=well, is_calibration=is_cal)


@st.cache_data(show_spinner=False)
def index_frames(analysis_dir: str) -> pd.DataFrame:
    base = Path(analysis_dir)
    dirs = {
        "raw": (base / "stitched", "*.tiff"),
        "qimg": (base / "q_image", "*.npz"),
        "qphi": (base / "qphi", "*.npz"),
        "cir": (base / "cir_avg", "*.csv"),
    }
    maps = {}
    for key, (d, pat) in dirs.items():
        maps[key] = ({stem_of(p.name): str(p) for p in d.glob(pat)}
                     if d.is_dir() else {})
    stems = sorted(set().union(*[set(m) for m in maps.values()]))
    rows = []
    for s in stems:
        meta = parse_meta(s)
        rows.append(dict(
            stem=s, label=s,
            raw=maps["raw"].get(s), qimg=maps["qimg"].get(s),
            qphi=maps["qphi"].get(s), cir=maps["cir"].get(s),
            has_raw=s in maps["raw"], has_qimg=s in maps["qimg"],
            has_qphi=s in maps["qphi"], has_cir=s in maps["cir"], **meta,
        ))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=[c for c in ("timestamp", "th", "stem") if c in df],
                            na_position="last").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw(fpath: str):
    from PIL import Image
    return np.asarray(Image.open(fpath)).astype(float)


@st.cache_data(show_spinner=False)
def load_qimg(fpath: str):
    d = np.load(fpath)
    return (d.get("qimg"), d.get("qx"), d.get("qz"), d.get("qimg_mask"))


@st.cache_data(show_spinner=False)
def load_qphi(fpath: str):
    d = np.load(fpath)
    return (d.get("q"), d.get("phi"), d.get("qphi"), d.get("qphi_mask"))


@st.cache_data(show_spinner=False)
def load_cir(fpath: str):
    df = pd.read_csv(fpath)
    cols = {c.lower(): c for c in df.columns}
    qcol = cols.get("q_ca") or cols.get("q") or df.columns[-2]
    icol = cols.get("iq_ca") or cols.get("intensity") or df.columns[-1]
    return df[qcol].to_numpy(float), df[icol].to_numpy(float)


def _apply_mask(z, mask):
    """Return a float copy with no-data entries set to NaN.

    The auto-reduction marks the masked-out region (beamstop, detector gaps,
    off-detector remesh pixels) two ways, and we honour both:

    * ``qimg_mask == True`` flags the *no-data* region (it coincides exactly
      with ``qimg == 0``); those pixels are blanked when the mask shape matches.
    * Remeshed / caked maps store no-data as literal ``0`` (and any non-finite
      value), so non-positive pixels are blanked as well.

    Blanking (rather than clipping to a floor) lets the panels render gaps as
    dark "no data" and lets line-cut ``nanmean`` ignore them.
    """
    z = np.asarray(z, float).copy()
    if mask is not None and getattr(mask, "shape", None) == z.shape:
        z[mask.astype(bool)] = np.nan          # mask True == masked-out
    z[~np.isfinite(z)] = np.nan
    z[z <= 0] = np.nan                          # 0 == no data in remeshed maps
    return z


def _log_scale(z):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(np.clip(z, 1e-6, None))


def _downsample(z, x, y, max_pixels=_RAW_MAX_PIXELS):
    """Stride-decimate a 2D array (+ optional axes) to <= max_pixels."""
    ny, nx = z.shape
    step = max(1, int(np.ceil(np.sqrt(ny * nx / max_pixels))))
    if step == 1:
        return z, x, y
    z = z[::step, ::step]
    if x is not None:
        x = np.asarray(x)[::step]
    if y is not None:
        y = np.asarray(y)[::step]
    return z, x, y


def _parse_centers(text: str):
    out = []
    for tok in re.split(r"[,\s]+", text.strip()):
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def _band_profile(z, coord_along, coord_band, center, width, mask=None):
    """Average ``z`` over a band ``center ± width/2`` along ``coord_band``.

    ``z`` is 2D with axis-0 varying ``coord0`` and axis-1 varying ``coord1``.
    Here ``coord_band`` selects which axis defines the integration band and
    ``coord_along`` is the axis the resulting 1-D profile runs along.

    Returns ``(x, y)`` where ``x = coord_along`` and ``y`` is the band mean, or
    ``None`` when the band is empty. ``axis_band`` is inferred: whichever axis's
    length matches ``coord_band``.
    """
    zc = _apply_mask(z, mask)
    lo, hi = center - width / 2.0, center + width / 2.0
    band = (coord_band >= lo) & (coord_band <= hi)
    if not band.any():
        return None
    with warnings.catch_warnings():  # all-NaN columns → nan, not a scary warning
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if len(coord_band) == zc.shape[0]:  # band runs down the rows → mean over rows
            prof = np.nanmean(zc[band, :], axis=0)
        else:                               # band runs across the cols → mean over cols
            prof = np.nanmean(zc[:, band], axis=1)
    return np.asarray(coord_along, float), prof


# ===========================================================================
st.set_page_config(page_title="GIWAXS Explorer", page_icon="🧭", layout="wide")
initialize_security_context()
require_authentication()

st.title("🧭 GIWAXS / GISAXS Explorer")
st.caption("Raw image · q-image · q–φ map · circular average — with q-image / "
           "q–φ line-cuts (qr, qz, q, φ).")

with st.sidebar:
    st.header("📁 Analysis folder")
    analysis = st.text_input(
        "analysis/ dir (has stitched/ q_image/ qphi/ cir_avg/)",
        value=DEFAULT_ANALYSIS)
    if st.button("🔄 Rescan"):
        index_frames.clear()
    if _HAVE_SECURITY and analysis and not is_path_allowed(analysis, allow_nonexistent=True):
        st.error("Folder outside allowed roots (secure mode).")
        st.stop()

    df = index_frames(analysis)
    if df.empty:
        st.warning("No stitched/ q_image/ qphi/ or cir_avg/ files found here.")
        st.stop()
    st.success(
        f"{len(df)} frames — "
        f"{int(df['has_raw'].sum())} raw · {int(df['has_qimg'].sum())} q-img · "
        f"{int(df['has_qphi'].sum())} q–φ · {int(df['has_cir'].sum())} 1D.")

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

# --- Frame picker -----------------------------------------------------------
c1, c2 = st.columns([4, 1])
labels = work["stem"].tolist()
chosen = c1.selectbox("Frame", options=labels, index=0) if len(labels) > 1 else labels[0]
idx = labels.index(chosen)
sel = work.iloc[int(idx)]
c2.metric("Frame", f"{int(idx) + 1}/{len(labels)}")

ts = sel["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if pd.notna(sel["timestamp"]) else "—"
th = f"{sel['th']:.3f}°" if pd.notna(sel["th"]) else "—"
st.markdown(f"**{sel['stem']}**  ·  θ = {th}  ·  well `{sel['well']}`  ·  t = {ts}")

# --- Display controls -------------------------------------------------------
dc1, dc2, dc3, dc4 = st.columns(4)
logI = dc1.checkbox("log I (2D panels)", value=True)
logq = dc2.checkbox("log q (1D)", value=False)
logiq = dc3.checkbox("log I (1D)", value=True)
cmap = dc4.selectbox("2D colormap", ["Viridis", "Turbo", "Inferno", "Jet", "Cividis"], index=0)

_PANEL_H = 380
# Colour shown for NaN / no-data pixels (heatmap gaps let the plot bg show).
_NODATA_BG = "#101010"


def _rng(col, label, key, lo_val=None, hi_val=None, fmt="%.4g"):
    """Two side-by-side optional number inputs → (min, max); None means auto."""
    a, b = col.columns(2)
    lo = a.number_input(f"{label} min", value=lo_val, key=f"{key}_lo", format=fmt)
    hi = b.number_input(f"{label} max", value=hi_val, key=f"{key}_hi", format=fmt)
    return lo, hi


with st.expander("🎛️ Ranges & colour scaling (blank = auto)", expanded=False):
    st.caption("Colour limits are in **intensity** units (pre-log). "
               "Auto colour uses robust percentiles of each panel.")
    ap, bp, cp = st.columns(3)
    ap.markdown("**A · raw**")
    a_vmin, a_vmax = _rng(ap, "I", "a_v")
    bp.markdown("**B · q-image**")
    b_vmin, b_vmax = _rng(bp, "I", "b_v")
    b_qxr = _rng(bp, "qx", "b_qx")
    b_qzr = _rng(bp, "qz", "b_qz")
    cp.markdown("**C · q–φ**")
    c_vmin, c_vmax = _rng(cp, "I", "c_v")
    c_qr = _rng(cp, "q", "c_q")
    c_phir = _rng(cp, "φ", "c_phi")
    st.markdown("**D · circular average**")
    dp1, dp2 = st.columns(2)
    d_qr = _rng(dp1, "q", "d_q")
    d_ir = _rng(dp2, "I", "d_i")


def _color_limits(z, vmin_I, vmax_I):
    """Map optional intensity limits into display (log) space; auto = 1/99.5 pct."""
    finite = z[np.isfinite(z)]
    if vmin_I is None or vmax_I is None:
        if finite.size:
            p_lo, p_hi = np.nanpercentile(finite, [1.0, 99.5])
        else:
            p_lo, p_hi = None, None
        vmin_I = p_lo if vmin_I is None else vmin_I
        vmax_I = p_hi if vmax_I is None else vmax_I
    if vmin_I is None or vmax_I is None:
        return None, None
    if logI:
        lo = float(np.log10(max(vmin_I, 1e-6)))
        hi = float(np.log10(max(vmax_I, 1e-6)))
        return lo, hi
    return float(vmin_I), float(vmax_I)


def _heatmap_fig(title, z, x, y, xlab, ylab, xlog=False, shapes=None,
                 y_reverse=False, vmin_I=None, vmax_I=None,
                 x_range=None, y_range=None):
    zz = _log_scale(z) if logI else z
    zmin, zmax = _color_limits(z, vmin_I, vmax_I)
    fig = go.Figure(go.Heatmap(
        z=zz, x=x, y=y, colorscale=cmap, zmin=zmin, zmax=zmax,
        colorbar=dict(title="log I" if logI else "I"),
        hovertemplate=f"{xlab}=%{{x:.4g}}<br>{ylab}=%{{y:.4g}}<br>I=%{{z:.3g}}<extra></extra>",
    ))
    xr = list(x_range) if x_range and None not in x_range else None
    fig.update_xaxes(title_text=xlab, type="log" if xlog else "linear", range=xr)
    if y_range and None not in y_range:
        yr = list(y_range)
        if y_reverse:
            yr = yr[::-1]
        fig.update_yaxes(title_text=ylab, range=yr)
    else:
        fig.update_yaxes(title_text=ylab, autorange="reversed" if y_reverse else True)
    if shapes:
        for sh in shapes:
            fig.add_shape(**sh)
    fig.update_layout(title=title, height=_PANEL_H, template="plotly_white",
                      plot_bgcolor=_NODATA_BG,
                      margin=dict(l=55, r=10, t=40, b=45))
    return fig


# ===========================================================================
# Line-cut controls (built first so we can overlay bands on the panels below)
# ===========================================================================
st.divider()
st.subheader("✂️ Line-cuts")

lc1, lc2, lc3, lc4 = st.columns([1.3, 1.6, 1.3, 1])
cut_source = lc1.selectbox("Cut on", ["q_image (qx–qz)", "q–φ map"], index=0)

if cut_source.startswith("q_image"):
    cut_dir = lc2.selectbox(
        "Direction",
        ["qr-cut  (I vs qr, fixed qz band)", "qz-cut  (I vs qz, fixed qr band)"],
        index=0)
    _is_qr = cut_dir.startswith("qr")
    centers_lab = "qz center(s)" if _is_qr else "qr center(s)"
    width_lab = "qz width" if _is_qr else "qr width"
    def_centers, def_width = ("0.0", 0.05)
else:
    cut_dir = lc2.selectbox(
        "Direction",
        ["q-cut  (I vs φ, fixed q band)", "φ-cut  (I vs q, fixed φ band)"],
        index=0)
    _is_qcut = cut_dir.startswith("q-cut")
    centers_lab = "q center(s)" if _is_qcut else "φ center(s)"
    width_lab = "q width" if _is_qcut else "φ width"
    def_centers, def_width = ("1.0", 0.05) if _is_qcut else ("0", 10.0)

centers_txt = lc3.text_input(centers_lab, value=def_centers,
                             help="Comma / space separated; one profile per center.")
width = lc4.number_input(width_lab, value=float(def_width), min_value=0.0,
                         step=0.01, format="%.3f")
centers = _parse_centers(centers_txt)

# Compute the cut profiles and the band rectangles to overlay on the map.
cut_curves = []          # list of (name, xarr, yarr)
qimg_shapes, qphi_shapes = [], []
_band_color = "rgba(255,0,0,0.15)"
_line_color = "crimson"

if centers:
    if cut_source.startswith("q_image") and sel["has_qimg"]:
        qimg, qx, qz, qmask = load_qimg(sel["qimg"])
        if qimg is not None:
            for c in centers:
                if _is_qr:   # band in qz → profile along qx (qr)
                    res = _band_profile(qimg, qx, qz, c, width, qmask)
                    if res:
                        cut_curves.append((f"qz={c:g}", res[0], res[1]))
                    qimg_shapes.append(dict(
                        type="rect", xref="x", yref="y",
                        x0=float(qx.min()), x1=float(qx.max()),
                        y0=c - width / 2, y1=c + width / 2,
                        fillcolor=_band_color, line=dict(color=_line_color, width=1)))
                else:        # band in qx (qr) → profile along qz
                    res = _band_profile(qimg, qz, qx, c, width, qmask)
                    if res:
                        cut_curves.append((f"qr={c:g}", res[0], res[1]))
                    qimg_shapes.append(dict(
                        type="rect", xref="x", yref="y",
                        y0=float(qz.min()), y1=float(qz.max()),
                        x0=c - width / 2, x1=c + width / 2,
                        fillcolor=_band_color, line=dict(color=_line_color, width=1)))
    elif cut_source.startswith("q–φ") and sel["has_qphi"]:
        q, phi, qphi, pmask = load_qphi(sel["qphi"])
        # qphi_mask is stored on the raw-detector grid, not (nphi, nq); skip it.
        pmask = pmask if getattr(pmask, "shape", None) == getattr(qphi, "shape", None) else None
        if qphi is not None:
            for c in centers:
                if _is_qcut:  # band in q → profile along phi
                    res = _band_profile(qphi, phi, q, c, width, pmask)
                    if res:
                        cut_curves.append((f"q={c:g}", res[0], res[1]))
                    qphi_shapes.append(dict(
                        type="rect", xref="x", yref="y",
                        y0=float(phi.min()), y1=float(phi.max()),
                        x0=c - width / 2, x1=c + width / 2,
                        fillcolor=_band_color, line=dict(color=_line_color, width=1)))
                else:         # band in phi → profile along q
                    res = _band_profile(qphi, q, phi, c, width, pmask)
                    if res:
                        cut_curves.append((f"φ={c:g}", res[0], res[1]))
                    qphi_shapes.append(dict(
                        type="rect", xref="x", yref="y",
                        x0=float(q.min()), x1=float(q.max()),
                        y0=c - width / 2, y1=c + width / 2,
                        fillcolor=_band_color, line=dict(color=_line_color, width=1)))

# ===========================================================================
# Four panels: A raw · B q-image · C q–φ · D circular average
# ===========================================================================
st.divider()
rowA = st.columns(2)
rowB = st.columns(2)

# A) stitched raw image ------------------------------------------------------
with rowA[0]:
    if sel["has_raw"]:
        raw = load_raw(sel["raw"])
        z = raw.astype(float).copy()
        z[~np.isfinite(z)] = np.nan
        z[z <= 0] = np.nan                     # negatives / gaps → dark
        z, xx, yy = _downsample(z, None, None)
        fig = _heatmap_fig("A · stitched raw", z, None, None,
                           "x (px)", "y (px)", y_reverse=True,
                           vmin_I=a_vmin, vmax_I=a_vmax)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No stitched raw image for this frame.")

# B) q-image (qx–qz) ---------------------------------------------------------
with rowA[1]:
    if sel["has_qimg"]:
        qimg, qx, qz, qmask = load_qimg(sel["qimg"])
        z = _apply_mask(qimg, qmask)
        z, xx, yy = _downsample(z, qx, qz)
        # Overlay shapes are in data coords, so downsampling doesn't affect them.
        fig = _heatmap_fig("B · q-image", z, xx, yy, "qx (Å⁻¹)", "qz (Å⁻¹)",
                           shapes=qimg_shapes, vmin_I=b_vmin, vmax_I=b_vmax,
                           x_range=b_qxr, y_range=b_qzr)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No q_image for this frame.")

# C) q–φ map -----------------------------------------------------------------
with rowB[0]:
    if sel["has_qphi"]:
        q, phi, qphi, pmask = load_qphi(sel["qphi"])
        pmask = pmask if getattr(pmask, "shape", None) == getattr(qphi, "shape", None) else None
        z = _apply_mask(qphi, pmask)
        fig = _heatmap_fig("C · q–φ map", z, q, phi, "q (Å⁻¹)", "φ (deg)",
                           xlog=logq, shapes=qphi_shapes,
                           vmin_I=c_vmin, vmax_I=c_vmax,
                           x_range=c_qr, y_range=c_phir)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No qphi map for this frame.")

# D) circular average --------------------------------------------------------
with rowB[1]:
    if sel["has_cir"]:
        qq, ii = load_cir(sel["cir"])
        fig = go.Figure(go.Scatter(
            x=qq, y=ii, mode="lines", line=dict(width=2.2, color="crimson"),
            hovertemplate="q=%{x:.4f}<br>I=%{y:.3g}<extra></extra>"))
        def _axrange(lo, hi, is_log):
            if lo is None and hi is None:
                return None
            if is_log:
                lo = np.log10(lo) if lo and lo > 0 else None
                hi = np.log10(hi) if hi and hi > 0 else None
            return [lo, hi] if lo is not None and hi is not None else None
        fig.update_xaxes(title_text="q (Å⁻¹)", type="log" if logq else "linear",
                         range=_axrange(d_qr[0], d_qr[1], logq))
        fig.update_yaxes(title_text="I(q)", type="log" if logiq else "linear",
                         range=_axrange(d_ir[0], d_ir[1], logiq))
        fig.update_layout(title="D · circular average", height=_PANEL_H,
                          template="plotly_white", margin=dict(l=60, r=15, t=40, b=45))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No circular average for this frame.")

# ===========================================================================
# Line-cut result plot + export
# ===========================================================================
if centers:
    st.divider()
    st.markdown("#### Line-cut profiles")
    if not cut_curves:
        st.warning("No data in the chosen band(s) — check centers / width / source panel.")
    else:
        along_is_q = (cut_source.startswith("q_image") and _is_qr) or \
                     (cut_source.startswith("q–φ") and not _is_qcut)
        if cut_source.startswith("q_image"):
            xlab = "qr (Å⁻¹)" if _is_qr else "qz (Å⁻¹)"
        else:
            xlab = "φ (deg)" if _is_qcut else "q (Å⁻¹)"
        xlog = logq and along_is_q

        import plotly.express as px
        colors = px.colors.sample_colorscale("Turbo", np.linspace(0, 1, max(1, len(cut_curves))))
        fig = go.Figure()
        for (name, xa, ya), col in zip(cut_curves, colors):
            fig.add_trace(go.Scatter(x=xa, y=ya, mode="lines", name=name,
                                     line=dict(width=2, color=col)))
        fig.update_xaxes(title_text=xlab, type="log" if xlog else "linear")
        fig.update_yaxes(title_text="I (band mean)", type="log" if logiq else "linear")
        fig.update_layout(height=420, template="plotly_white",
                          margin=dict(l=60, r=15, t=25, b=50),
                          legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, use_container_width=True)

        # CSV export: outer-join all profiles on their common x-axis.
        buf = io.StringIO()
        frames = []
        for name, xa, ya in cut_curves:
            frames.append(pd.DataFrame({xlab: xa, f"I[{name}]": ya}))
        out = frames[0]
        for f in frames[1:]:
            out = out.merge(f, on=xlab, how="outer")
        out = out.sort_values(xlab)
        out.to_csv(buf, index=False)
        st.download_button("⬇️ Download line-cuts (CSV)", buf.getvalue(),
                           file_name=f"linecuts_{sel['stem']}.csv", mime="text/csv")

# --- Frame table ------------------------------------------------------------
with st.expander("📋 Frame table", expanded=False):
    st.dataframe(
        work[["stem", "th", "well", "timestamp",
              "has_raw", "has_qimg", "has_qphi", "has_cir"]],
        use_container_width=True, hide_index=True)
