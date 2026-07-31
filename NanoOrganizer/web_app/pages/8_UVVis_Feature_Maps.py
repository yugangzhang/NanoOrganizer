#!/usr/bin/env python3
"""UV-Vis Feature Maps — recipe-space maps of the *reduced* SPR fit features.

Fast, paper-ready views of the fitted features (``peak_abs, np_center, np_fwhm,
np_area, np_amp``) over the ``(NCit, acid, HAu)`` recipe cube — the reduced-data
counterpart to the (slower) 1-D spectra browser.  Reads only the lightweight
manifest parquet written by ``UV_Vis.droplet_recognition.write_explorer_dataset``
(no spectra load), and reuses the mGpBo analysis + plotting functions directly so
the maps match notebook 604 exactly:

* **3-D recipe space** — rotatable Plotly scatter, colour = feature, marker size
  ∝ reliability (``recipe_space_points`` + ``plot_recipe_space_3d_plotly``).
* **2-D condition slices** — bin ``acid`` into N ranges, weighted-median feature
  over the ``NCit × HAu`` grid, with a companion reliability map
  (``condition_slices`` + ``plot_condition_slices``).

Private to its owner (see ``components.uvvis_access``).  Runs as a page of the
NanoOrganizer web app or standalone::

    streamlit run NanoOrganizer/web_app/pages/8_UVVis_Feature_Maps.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

# --- access gate (owner-only in secure mode; open locally) -----------------
try:
    from NanoOrganizer.web_app.components.uvvis_access import require_uvvis_owner
except Exception:  # pragma: no cover - standalone fallback
    def require_uvvis_owner():
        return None

# --- optional app integration (security + folder browser) ------------------
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
except Exception:  # pragma: no cover
    _HAVE_BROWSER = False

# --- mGpBo analysis + plotting (single source of truth = notebook 604) -----
try:
    from UV_Vis.droplet_recognition import (
        recipe_space_points, plot_recipe_space_3d_plotly,
        condition_slices, plot_condition_slices,
    )
    _HAVE_MGPBO = True
    _MGPBO_ERR = ""
except Exception as e:  # pragma: no cover - mGpBo not importable
    _HAVE_MGPBO = False
    _MGPBO_ERR = str(e)


DEFAULT_DIR = (
    "/home/yuzhang/Repos/mGpBo/results/600_Droplet_Recognition/nanoorganizer_uvvis"
)
FEATURES = ["peak_abs", "np_center", "np_fwhm", "np_area", "np_amp"]
FEATURE_HELP = {
    "peak_abs": "absorbance at the SPR peak (yield proxy)",
    "np_center": "SPR peak wavelength λ₀ (size proxy)",
    "np_fwhm": "SPR peak width (dispersity proxy)",
    "np_area": "integrated SPR area (total product)",
    "np_amp": "Lorentzian peak height",
}


@st.cache_data(show_spinner=False)
def load_manifest(folder: str) -> pd.DataFrame | None:
    p = Path(folder) / "uvvis_explorer_manifest.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


# ===========================================================================
# Page
# ===========================================================================
st.set_page_config(page_title="UV-Vis Feature Maps", page_icon="🗺️", layout="wide")
initialize_security_context()
require_authentication()
require_uvvis_owner()

st.title("🗺️ UV-Vis Feature Maps")
st.caption("Recipe-space maps of the reduced SPR fit features — 3-D scatter + "
           "reliability-weighted condition slices.")

if not _HAVE_MGPBO:
    st.error("Could not import `UV_Vis.droplet_recognition` (mGpBo). "
             "Install mGpBo into this environment.\n\n" + _MGPBO_ERR)
    st.stop()

# --- Sidebar: data source --------------------------------------------------
with st.sidebar:
    st.header("📁 Dataset")
    if _HAVE_BROWSER:
        folder = folder_picker(key="uvvis_maps_folder",
                               label="explorer dataset folder", default=DEFAULT_DIR)
    else:
        folder = st.text_input("explorer dataset folder", value=DEFAULT_DIR)
        if _HAVE_SECURITY and folder and not is_path_allowed(folder, allow_nonexistent=True):
            st.error("This folder is outside the allowed roots (secure mode).")
            st.stop()
    if st.button("🔄 Reload"):
        load_manifest.clear()

man = load_manifest(folder) if folder else None
if man is None:
    st.warning("No `uvvis_explorer_manifest.parquet` in that folder. Generate it "
               "with `write_explorer_dataset(fdf, combined, out_dir)` (notebook 604 §9).")
    st.stop()

n_ok = int(man["success"].sum()) if "success" in man else len(man)
st.sidebar.success(f"{len(man)} recipes ({n_ok} successful).")

with st.sidebar:
    st.header("🔎 Filters")
    work = man.copy()
    if "success" in work and st.checkbox("Successful fits only", value=True):
        work = work[work["success"]]
    if "reliability" in work and work["reliability"].notna().any():
        rmin = st.slider("Min reliability", 0.0, 1.0, 0.0, 0.05,
                         help="Rows below this are dropped before mapping.")
        work = work[work["reliability"].fillna(0.0) >= rmin]
    if "tier" in work:
        order = ["excellent", "good", "marginal", "poor", "failed"]
        tiers = [t for t in order if t in set(man["tier"])]
        pick = st.multiselect("Quality tier", tiers, default=tiers)
        work = work[work["tier"].isin(pick)]

st.markdown(f"**{len(work)} recipe(s)** in the maps (of {len(man)}).")
if work.empty:
    st.info("Everything filtered out — widen a filter in the sidebar.")
    st.stop()

has_rel = "reliability" in work and work["reliability"].notna().any()

tab3d, tab2d = st.tabs(["🌐 3-D recipe space", "🔲 2-D condition slices"])

# ---------------------------------------------------------------------------
# 3-D recipe-space scatter
# ---------------------------------------------------------------------------
with tab3d:
    c1, c2, c3 = st.columns(3)
    feat = c1.selectbox("Feature (colour)", FEATURES, index=0,
                        help=FEATURE_HELP.get(FEATURES[0], ""), key="f3d")
    st.caption(FEATURE_HELP.get(feat, ""))
    cmap = c2.selectbox("Colourscale", ["Turbo", "Viridis", "Cividis", "RdBu"], index=0)
    size_by_rel = c3.checkbox("Marker size ∝ reliability", value=has_rel,
                              disabled=not has_rel)

    d = work.dropna(subset=["NCit", "acid", "HAu", feat]).copy()
    if d.empty:
        st.info(f"No finite `{feat}` in the current selection.")
    else:
        pts = recipe_space_points(d, feat, success_only=False)
        size_by = d["reliability"].to_numpy() if (size_by_rel and has_rel) else None
        hover = d["planned"].to_numpy() if "planned" in d else None
        fig = plot_recipe_space_3d_plotly(
            pts, colorscale=cmap, clip_pct=(2, 98),
            size_by=size_by, size_range=(3, 11), hover=hover,
            title=f"{feat} in recipe space"
            + (" (size ∝ reliability)" if size_by is not None else ""))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{len(d)} recipes · drag to rotate · hover for value + planned id")

# ---------------------------------------------------------------------------
# 2-D condition slices (weighted median over NCit × HAu, sliced by acid)
# ---------------------------------------------------------------------------
with tab2d:
    c1, c2, c3, c4 = st.columns(4)
    feat2 = c1.selectbox("Feature", FEATURES, index=0, key="f2d")
    st.caption(FEATURE_HELP.get(feat2, ""))
    axes = ["NCit", "acid", "HAu"]
    xax = c2.selectbox("X axis", axes, index=0)
    yax = c3.selectbox("Y axis", [a for a in axes if a != xax], index=1)
    slice_by = [a for a in axes if a not in (xax, yax)][0]
    n_bins = c4.slider("Slice bins", 2, 6, 4, help=f"Bins along {slice_by}.")

    o1, o2 = st.columns(2)
    weighted = o1.checkbox("Reliability-weighted", value=has_rel, disabled=not has_rel)
    fade = o2.checkbox("Fade untrusted cells", value=has_rel, disabled=not has_rel)

    d = work.dropna(subset=[feat2, xax, yax, slice_by])
    if d.empty:
        st.info(f"No finite `{feat2}` in the current selection.")
    else:
        sl = condition_slices(
            d, feat2, x=xax, y=yax, slice_by=slice_by, n_bins=int(n_bins),
            success_only=False,
            weight_col=("reliability" if (weighted and has_rel) else None))
        fig2 = plot_condition_slices(
            sl, show_reliability=(has_rel and weighted),
            fade_by_reliability=(fade and has_rel and weighted))
        st.pyplot(fig2)
        st.caption(f"weighted-median {feat2} over ({xax} × {yax}), sliced by "
                   f"{slice_by} into {int(n_bins)} bins" +
                   (" · brightness = reliability" if (has_rel and weighted) else ""))

# --- export the filtered manifest ------------------------------------------
import io
buf = io.StringIO()
work.to_csv(buf, index=False)
st.download_button("⬇️ Download filtered manifest (CSV)", data=buf.getvalue(),
                   file_name="uvvis_feature_maps_selection.csv", mime="text/csv")
