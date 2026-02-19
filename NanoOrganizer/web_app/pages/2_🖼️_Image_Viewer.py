#!/usr/bin/env python3
"""
2D Image Viewer - Dedicated tool for viewing 2D images and stacks.

Features:
- Load image stacks (NPY, PNG, TIF, etc.)
- Browse through frames
- Interactive (Plotly) and Static (Matplotlib) modes
- Log scale display
- Colormap selection
- Intensity adjustment
- Side-by-side comparison
- Export: PNG, SVG, interactive HTML
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402

import streamlit as st  # noqa: E402
import numpy as np  # noqa: E402
from pathlib import Path  # noqa: E402
import io  # noqa: E402
import sys  # noqa: E402
import copy  # noqa: E402
import pprint  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from components.folder_browser import folder_browser  # noqa: E402
from components.floating_button import floating_sidebar_toggle  # noqa: E402

# User-mode restriction (set by nanoorganizer_user)
_user_mode = st.session_state.get("user_mode", False)
_start_dir = st.session_state.get("user_start_dir", None)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLORMAPS = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
             'turbo', 'jet', 'hot', 'cool', 'gray', 'bone', 'seismic',
             'RdYlBu', 'RdBu', 'coolwarm']
PLOT_MODE_OPTIONS = ["Interactive (Plotly)", "Static (Matplotlib)"]
VIEW_MODE_OPTIONS = ["Single image", "Side-by-side comparison", "Grid view"]
ASPECT_OPTIONS = ["equal", "auto"]
INTERPOLATION_OPTIONS = ["nearest", "bilinear", "bicubic", "gaussian"]

# Matplotlib colormap -> Plotly colorscale name mapping
_PLOTLY_CMAP = {
    'viridis': 'Viridis', 'plasma': 'Plasma', 'inferno': 'Inferno',
    'magma': 'Magma', 'cividis': 'Cividis', 'turbo': 'Turbo',
    'jet': 'Jet', 'hot': 'Hot', 'cool': 'ice', 'gray': 'Greys',
    'bone': 'Greys', 'seismic': 'RdBu', 'RdYlBu': 'RdYlBu',
    'RdBu': 'RdBu', 'coolwarm': 'RdBu',
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_image(file_path):
    """Load image from various formats."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    try:
        if suffix == '.npy':
            return np.load(file_path)
        elif suffix == '.npz':
            data = np.load(file_path)
            return data[data.files[0]]
        elif suffix in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            from PIL import Image
            img = Image.open(file_path)
            return np.array(img, dtype=np.float64)
        else:
            st.error(f"Unsupported format: {suffix}")
            return None
    except Exception as e:
        st.error(f"Error loading {path.name}: {e}")
        return None


def _save_fig_to_bytes(fig, format='png', dpi=300):
    """Save matplotlib figure to bytes buffer."""
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf


def prepare_display_data(img, use_log):
    """Apply log10 transform if requested. Returns (display_data, label_suffix)."""
    if not use_log:
        return img.astype(np.float64), ""
    # log10 with floor for zero/negative values
    floor_val = np.min(img[img > 0]) if np.any(img > 0) else 1e-10
    clipped = np.clip(img.astype(np.float64), floor_val, None)
    return np.log10(clipped), " (log₁₀)"


def calc_intensity_range(img, auto_contrast, vmin_pct, vmax_pct):
    """Calculate vmin/vmax from percentiles."""
    if auto_contrast:
        return np.nanpercentile(img, [1, 99])
    return np.nanpercentile(img, vmin_pct), np.nanpercentile(img, vmax_pct)


def get_frame(img, frame_idx=None):
    """Extract single frame from stack or return 2D image as-is."""
    if img.ndim == 3:
        return img[frame_idx if frame_idx is not None else 0]
    return img


def make_plotly_heatmap(display_data, original_data, title, cmap, vmin, vmax,
                        use_log, equal_aspect=True):
    """Create a Plotly heatmap figure with hover showing original intensity."""
    colorscale = _PLOTLY_CMAP.get(cmap, 'Viridis')
    colorbar_title = "log₁₀(I)" if use_log else "Intensity"

    # For Heatmap, customdata must be 3D (M x N x K) — access via %{customdata[0]}
    custom_3d = original_data[:, :, np.newaxis]

    hover_parts = ['x: %{x}', 'y: %{y}']
    if use_log:
        hover_parts.append('Intensity: %{customdata[0]:.4g}')
        hover_parts.append('log₁₀(I): %{z:.3f}')
    else:
        hover_parts.append('Intensity: %{customdata[0]:.4g}')
    hover = '<br>'.join(hover_parts) + '<extra></extra>'

    fig = go.Figure(go.Heatmap(
        z=display_data,
        zmin=vmin,
        zmax=vmax,
        colorscale=colorscale,
        colorbar=dict(title=colorbar_title),
        customdata=custom_3d,
        hovertemplate=hover,
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=700,
        hovermode='closest',
    )

    # y-axis: row 0 at top (like imshow origin='upper')
    fig.update_yaxes(autorange='reversed')

    if equal_aspect:
        fig.update_yaxes(scaleanchor='x', scaleratio=1)

    return fig


def make_plotly_grid(image_list, name_list, cmap, use_log, auto_contrast,
                     vmin_pct, vmax_pct, n_cols):
    """Create Plotly subplots grid of heatmaps."""
    n = len(image_list)
    n_rows = (n + n_cols - 1) // n_cols
    colorscale = _PLOTLY_CMAP.get(cmap, 'Viridis')

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=name_list,
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
    )

    for idx, (img, name) in enumerate(zip(image_list, name_list)):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        display_data, _ = prepare_display_data(img, use_log)
        vmin, vmax = calc_intensity_range(display_data, auto_contrast, vmin_pct, vmax_pct)

        # customdata must be 3D for Heatmap
        custom_3d = img.astype(np.float64)[:, :, np.newaxis]

        fig.add_trace(
            go.Heatmap(
                z=display_data,
                zmin=vmin, zmax=vmax,
                colorscale=colorscale,
                customdata=custom_3d,
                hovertemplate=(
                    f'<b>{name}</b><br>'
                    'x: %{x}  y: %{y}<br>'
                    'Intensity: %{customdata[0]:.4g}<extra></extra>'
                ),
                showscale=(idx == 0),
            ),
            row=row, col=col,
        )

        # y-axis reversed for image convention
        fig.update_yaxes(autorange='reversed', row=row, col=col)
        fig.update_yaxes(scaleanchor=f'x{idx+1 if idx > 0 else ""}',
                         scaleratio=1, row=row, col=col)

    fig.update_layout(
        height=max(400, 350 * n_rows),
        hovermode='closest',
    )

    return fig


def export_section_plotly(fig, file_stem):
    """Render Plotly export buttons."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        html_buffer = io.StringIO()
        fig.write_html(html_buffer)
        html_bytes = html_buffer.getvalue().encode()
        st.download_button(
            label="💾 HTML (interactive)",
            data=html_bytes,
            file_name=f"{file_stem}.html",
            mime="text/html",
            help="Interactive plot — zoom, pan, hover preserved!"
        )

    with col2:
        try:
            img_bytes = fig.to_image(format="png", width=1200, height=900)
            st.download_button(
                label="💾 PNG",
                data=img_bytes,
                file_name=f"{file_stem}.png",
                mime="image/png"
            )
        except Exception:
            st.info("Install kaleido for PNG export")

    with col3:
        try:
            img_bytes = fig.to_image(format="svg")
            st.download_button(
                label="💾 SVG",
                data=img_bytes,
                file_name=f"{file_stem}.svg",
                mime="image/svg+xml"
            )
        except Exception:
            st.info("Install kaleido for SVG export")

    with col4:
        st.metric("Mode", "Interactive")


def export_section_matplotlib(mpl_fig, file_stem):
    """Render Matplotlib export buttons."""
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        buf = _save_fig_to_bytes(mpl_fig, dpi=300)
        st.download_button(
            label="💾 PNG (300 DPI)",
            data=buf,
            file_name=f"{file_stem}.png",
            mime="image/png"
        )

    with col2:
        buf_svg = _save_fig_to_bytes(mpl_fig, format='svg')
        st.download_button(
            label="💾 SVG",
            data=buf_svg,
            file_name=f"{file_stem}.svg",
            mime="image/svg+xml"
        )

    with col3:
        st.metric("Mode", "Static")


def _to_bool(value, default=False):
    """Convert arbitrary values to bool with fallback."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _to_int(value, default, min_value=None, max_value=None):
    """Convert arbitrary values to int with optional clamp."""
    fallback = 0 if default is None else default
    try:
        parsed = int(value)
    except Exception:
        try:
            parsed = int(fallback)
        except Exception:
            parsed = 0
    if min_value is not None:
        parsed = max(parsed, min_value)
    if max_value is not None:
        parsed = min(parsed, max_value)
    return parsed


def _to_float(value, default, min_value=None, max_value=None):
    """Convert arbitrary values to float with optional clamp."""
    fallback = 0.0 if default is None else default
    try:
        parsed = float(value)
    except Exception:
        try:
            parsed = float(fallback)
        except Exception:
            parsed = 0.0
    if min_value is not None:
        parsed = max(parsed, min_value)
    if max_value is not None:
        parsed = min(parsed, max_value)
    return parsed


def _build_image_plot_spec(images):
    """Build serializable image-viewer config from current UI state."""
    frame_by_file = {}
    for file_name, img in images.items():
        if img.ndim == 3:
            frame_key = f"img_frame_{file_name}"
            frame_by_file[file_name] = _to_int(
                st.session_state.get(frame_key, 0),
                0,
                0,
                img.shape[0] - 1
            )

    selected_compare = st.session_state.get("img_selected_compare_files", [])
    if isinstance(selected_compare, str):
        selected_compare = [selected_compare]
    if not isinstance(selected_compare, (list, tuple, set)):
        selected_compare = []

    return {
        "version": 1,
        "plot_mode": st.session_state.get("img_plot_mode", PLOT_MODE_OPTIONS[0]),
        "view_mode": st.session_state.get("img_view_mode", VIEW_MODE_OPTIONS[0]),
        "cmap": st.session_state.get("img_cmap", COLORMAPS[0]),
        "use_log_scale": _to_bool(st.session_state.get("img_use_log_scale", False), False),
        "intensity": {
            "auto_contrast": _to_bool(st.session_state.get("img_auto_contrast", True), True),
            "vmin_pct": _to_float(st.session_state.get("img_vmin_pct", 0.0), 0.0, 0.0, 50.0),
            "vmax_pct": _to_float(st.session_state.get("img_vmax_pct", 100.0), 100.0, 50.0, 100.0),
        },
        "matplotlib": {
            "aspect": st.session_state.get("img_aspect", ASPECT_OPTIONS[0]),
            "interpolation": st.session_state.get("img_interpolation", INTERPOLATION_OPTIONS[0]),
        },
        "selection": {
            "single_file": st.session_state.get("img_selected_file"),
            "compare_files": [str(f) for f in selected_compare],
            "grid_cols": _to_int(st.session_state.get("img_grid_cols", 3), 3, 1, 4),
            "frame_by_file": frame_by_file,
        },
    }


def _generate_image_editor_python(plot_spec):
    """Generate editable Python code for image-viewer config."""
    spec_text = pprint.pformat(plot_spec, sort_dicts=False, width=100, compact=False)
    return (
        "# NanoOrganizer image-viewer editor (experimental)\n"
        "# Edit plot_spec and click 'Run Edited Python'.\n"
        "# The app uses `result` (if defined) otherwise `plot_spec`.\n\n"
        f"plot_spec = {spec_text}\n\n"
        "# Example tweaks:\n"
        "# plot_spec['cmap'] = 'magma'\n"
        "# plot_spec['selection']['grid_cols'] = 4\n\n"
        "result = plot_spec\n"
    )


def _sanitize_image_plot_spec(candidate, fallback_spec, images):
    """Sanitize user-edited image-viewer spec and return (spec, warnings)."""
    warnings = []
    if not isinstance(candidate, dict):
        warnings.append("Edited code did not return a dict; keeping previous settings.")
        return copy.deepcopy(fallback_spec), warnings

    normalized = copy.deepcopy(fallback_spec)
    image_names = list(images.keys())

    plot_mode = candidate.get("plot_mode", normalized.get("plot_mode", PLOT_MODE_OPTIONS[0]))
    if plot_mode in PLOT_MODE_OPTIONS:
        normalized["plot_mode"] = plot_mode
    elif "plot_mode" in candidate:
        warnings.append(f"Ignored unsupported plot_mode '{plot_mode}'.")

    view_mode = candidate.get("view_mode", normalized.get("view_mode", VIEW_MODE_OPTIONS[0]))
    if view_mode in VIEW_MODE_OPTIONS:
        normalized["view_mode"] = view_mode
    elif "view_mode" in candidate:
        warnings.append(f"Ignored unsupported view_mode '{view_mode}'.")

    cmap = candidate.get("cmap", normalized.get("cmap", COLORMAPS[0]))
    if cmap in COLORMAPS:
        normalized["cmap"] = cmap
    elif "cmap" in candidate:
        warnings.append(f"Ignored unsupported colormap '{cmap}'.")

    normalized["use_log_scale"] = _to_bool(candidate.get("use_log_scale", normalized.get("use_log_scale", False)), False)

    intensity = candidate.get("intensity", {})
    if not isinstance(intensity, dict):
        intensity = {}
        warnings.append("intensity must be a dict; keeping previous intensity settings.")
    fallback_intensity = normalized.get("intensity", {})
    auto_contrast = _to_bool(intensity.get("auto_contrast", fallback_intensity.get("auto_contrast", True)), True)
    vmin_pct = _to_float(intensity.get("vmin_pct", fallback_intensity.get("vmin_pct", 0.0)), 0.0, 0.0, 50.0)
    vmax_pct = _to_float(intensity.get("vmax_pct", fallback_intensity.get("vmax_pct", 100.0)), 100.0, 50.0, 100.0)
    if vmin_pct > vmax_pct:
        warnings.append("vmin_pct > vmax_pct; swapped values.")
        vmin_pct, vmax_pct = vmax_pct, vmin_pct
    normalized["intensity"] = {
        "auto_contrast": auto_contrast,
        "vmin_pct": vmin_pct,
        "vmax_pct": vmax_pct,
    }

    mpl_cfg = candidate.get("matplotlib", {})
    if not isinstance(mpl_cfg, dict):
        mpl_cfg = {}
        warnings.append("matplotlib must be a dict; keeping previous settings.")
    fallback_mpl = normalized.get("matplotlib", {})
    aspect = mpl_cfg.get("aspect", fallback_mpl.get("aspect", ASPECT_OPTIONS[0]))
    interpolation = mpl_cfg.get("interpolation", fallback_mpl.get("interpolation", INTERPOLATION_OPTIONS[0]))
    normalized["matplotlib"] = {
        "aspect": aspect if aspect in ASPECT_OPTIONS else ASPECT_OPTIONS[0],
        "interpolation": interpolation if interpolation in INTERPOLATION_OPTIONS else INTERPOLATION_OPTIONS[0],
    }

    selection = candidate.get("selection", {})
    if not isinstance(selection, dict):
        selection = {}
        warnings.append("selection must be a dict; keeping previous selection settings.")
    fallback_selection = normalized.get("selection", {})

    single_file = selection.get("single_file", fallback_selection.get("single_file"))
    if single_file not in image_names and image_names:
        single_file = image_names[0]

    compare_files = selection.get("compare_files", fallback_selection.get("compare_files", []))
    if isinstance(compare_files, str):
        compare_files = [compare_files]
    if not isinstance(compare_files, (list, tuple, set)):
        compare_files = []
    compare_valid = []
    for file_name in compare_files:
        file_name = str(file_name)
        if file_name in image_names and file_name not in compare_valid:
            compare_valid.append(file_name)
    if not compare_valid:
        compare_valid = image_names[:min(4, len(image_names))]
    compare_valid = compare_valid[:4]

    grid_cols = _to_int(selection.get("grid_cols", fallback_selection.get("grid_cols", 3)), 3, 1, 4)

    incoming_frames = selection.get("frame_by_file", fallback_selection.get("frame_by_file", {}))
    if not isinstance(incoming_frames, dict):
        incoming_frames = {}
    frame_by_file = {}
    for file_name, img in images.items():
        if img.ndim == 3:
            frame_by_file[file_name] = _to_int(incoming_frames.get(file_name, 0), 0, 0, img.shape[0] - 1)

    normalized["selection"] = {
        "single_file": single_file,
        "compare_files": compare_valid,
        "grid_cols": grid_cols,
        "frame_by_file": frame_by_file,
    }

    return normalized, warnings


def _apply_image_plot_spec_to_state(plot_spec, images):
    """Apply normalized image-viewer config to session state before widgets render."""
    if plot_spec.get("plot_mode") in PLOT_MODE_OPTIONS:
        st.session_state["img_plot_mode"] = plot_spec["plot_mode"]
    if plot_spec.get("view_mode") in VIEW_MODE_OPTIONS:
        st.session_state["img_view_mode"] = plot_spec["view_mode"]
    if plot_spec.get("cmap") in COLORMAPS:
        st.session_state["img_cmap"] = plot_spec["cmap"]
    st.session_state["img_use_log_scale"] = _to_bool(plot_spec.get("use_log_scale", False), False)

    intensity = plot_spec.get("intensity", {})
    st.session_state["img_auto_contrast"] = _to_bool(intensity.get("auto_contrast", True), True)
    st.session_state["img_vmin_pct"] = _to_float(intensity.get("vmin_pct", 0.0), 0.0, 0.0, 50.0)
    st.session_state["img_vmax_pct"] = _to_float(intensity.get("vmax_pct", 100.0), 100.0, 50.0, 100.0)

    mpl_cfg = plot_spec.get("matplotlib", {})
    aspect = mpl_cfg.get("aspect", ASPECT_OPTIONS[0])
    interpolation = mpl_cfg.get("interpolation", INTERPOLATION_OPTIONS[0])
    st.session_state["img_aspect"] = aspect if aspect in ASPECT_OPTIONS else ASPECT_OPTIONS[0]
    st.session_state["img_interpolation"] = interpolation if interpolation in INTERPOLATION_OPTIONS else INTERPOLATION_OPTIONS[0]

    selection = plot_spec.get("selection", {})
    image_names = list(images.keys())
    single_file = selection.get("single_file")
    if single_file not in image_names and image_names:
        single_file = image_names[0]
    st.session_state["img_selected_file"] = single_file

    compare_files = selection.get("compare_files", [])
    if isinstance(compare_files, str):
        compare_files = [compare_files]
    if not isinstance(compare_files, (list, tuple, set)):
        compare_files = []
    compare_valid = []
    for file_name in compare_files:
        file_name = str(file_name)
        if file_name in image_names and file_name not in compare_valid:
            compare_valid.append(file_name)
    if not compare_valid:
        compare_valid = image_names[:min(4, len(image_names))]
    st.session_state["img_selected_compare_files"] = compare_valid[:4]
    st.session_state["img_grid_cols"] = _to_int(selection.get("grid_cols", 3), 3, 1, 4)

    frame_by_file = selection.get("frame_by_file", {})
    if not isinstance(frame_by_file, dict):
        frame_by_file = {}
    for file_name, img in images.items():
        if img.ndim == 3:
            st.session_state[f"img_frame_{file_name}"] = _to_int(
                frame_by_file.get(file_name, 0),
                0,
                0,
                img.shape[0] - 1
            )


def _execute_image_editor(code_text, base_plot_spec, images):
    """Execute editor code and sanitize resulting plot spec."""
    execution_locals = {
        "plot_spec": copy.deepcopy(base_plot_spec),
        "result": None,
        "image_files": list(images.keys()),
        "image_shapes": {k: tuple(v.shape) for k, v in images.items()},
        "copy": copy,
        "np": np,
    }
    exec(code_text, {"__builtins__": __builtins__}, execution_locals)
    candidate = execution_locals.get("result")
    if candidate is None:
        candidate = execution_locals.get("plot_spec")
    return _sanitize_image_plot_spec(candidate, base_plot_spec, images)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

st.title("🖼️ 2D Image Viewer")
st.markdown("Dedicated viewer for 2D images, detector data, and image stacks")

# Floating sidebar toggle button (bottom-left)
floating_sidebar_toggle()

# Session state for persistent data
if 'images_viewer' not in st.session_state:
    st.session_state['images_viewer'] = {}
if 'image_paths_viewer' not in st.session_state:
    st.session_state['image_paths_viewer'] = {}

st.session_state.setdefault("img_plot_mode", PLOT_MODE_OPTIONS[0])
st.session_state.setdefault("img_view_mode", VIEW_MODE_OPTIONS[0])
st.session_state.setdefault("img_cmap", COLORMAPS[0])
st.session_state.setdefault("img_use_log_scale", False)
st.session_state.setdefault("img_auto_contrast", True)
st.session_state.setdefault("img_vmin_pct", 0.0)
st.session_state.setdefault("img_vmax_pct", 100.0)
st.session_state.setdefault("img_aspect", ASPECT_OPTIONS[0])
st.session_state.setdefault("img_interpolation", INTERPOLATION_OPTIONS[0])
st.session_state.setdefault("img_selected_file", None)
st.session_state.setdefault("img_selected_compare_files", [])
st.session_state.setdefault("img_grid_cols", 3)
st.session_state.setdefault("img_editor_code", "")
st.session_state.setdefault("img_editor_status", "")
st.session_state.setdefault("img_editor_warnings", [])

# ---------------------------------------------------------------------------
# Sidebar: Load Images
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📁 Load Images")

    data_source = st.radio(
        "Data source",
        ["Upload files", "Browse server"],
        key="img_data_source",
        help="Upload images or browse server filesystem"
    )

    if data_source == "Upload files":
        uploaded_files = st.file_uploader(
            "Upload image files",
            type=['npy', 'npz', 'png', 'jpg', 'jpeg', 'tif', 'tiff'],
            accept_multiple_files=True,
            help="Upload NPY, NPZ, PNG, TIFF, or JPG files"
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    temp_path = Path(f"/tmp/{uploaded_file.name}")
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())

                    img = load_image(str(temp_path))
                    if img is not None:
                        st.session_state['images_viewer'][uploaded_file.name] = img
                        st.session_state['image_paths_viewer'][uploaded_file.name] = uploaded_file.name
                except Exception as e:
                    st.error(f"Error: {e}")

    else:  # Browse server
        st.markdown("**🗂️ Interactive Folder Browser**")
        st.markdown("Click folders to navigate, select files with checkboxes:")

        # File pattern selector
        st.markdown("**📋 File Type Filter:**")
        pattern = st.selectbox(
            "Extension pattern",
            ["*.*", "*.npy", "*.npz", "*.png", "*.tif", "*.tiff", "*.jpg"],
            help="Filter files by extension",
            label_visibility="collapsed"
        )

        st.info("💡 Tip: Use '🔍 Advanced Filters' below for name-based filtering")

        # Use folder browser component
        selected_files = folder_browser(
            key="image_viewer_browser",
            show_files=True,
            file_pattern=pattern,
            multi_select=True,
            initial_path=_start_dir if _user_mode else None,
            restrict_to_start_dir=_user_mode,
        )

        # Load button
        if selected_files and st.button("📥 Load Selected Files", key="img_load_btn"):
            for full_path in selected_files:
                img = load_image(full_path)
                if img is not None:
                    file_name = Path(full_path).name
                    st.session_state['images_viewer'][file_name] = img
                    st.session_state['image_paths_viewer'][file_name] = full_path
                    st.success(f"✅ Loaded {file_name}")

    # Get images from session state
    images = st.session_state['images_viewer']
    file_paths = st.session_state['image_paths_viewer']

    # Clear button
    if images:
        if st.button("🗑️ Clear All Images", key="clear_img_data"):
            st.session_state['images_viewer'] = {}
            st.session_state['image_paths_viewer'] = {}
            st.session_state.pop("img_pending_plot_spec", None)
            st.rerun()

    if not images:
        st.info("👆 Upload or select images to get started")
        st.stop()

    st.success(f"✅ Loaded {len(images)} image(s)")

    if "img_pending_plot_spec" in st.session_state:
        pending_spec = st.session_state.pop("img_pending_plot_spec")
        _apply_image_plot_spec_to_state(pending_spec, images)

    image_names = list(images.keys())
    if st.session_state.get("img_selected_file") not in image_names:
        st.session_state["img_selected_file"] = image_names[0]
    selected_compare = st.session_state.get("img_selected_compare_files", [])
    if not isinstance(selected_compare, (list, tuple, set)):
        selected_compare = []
    selected_compare = [f for f in selected_compare if f in image_names]
    if not selected_compare:
        selected_compare = image_names[:min(4, len(image_names))]
    st.session_state["img_selected_compare_files"] = list(selected_compare)[:4]
    st.session_state["img_grid_cols"] = _to_int(st.session_state.get("img_grid_cols", 3), 3, 1, 4)

    # -----------------------------------------------------------------------
    # Display Controls
    # -----------------------------------------------------------------------

    st.header("🎨 Display Controls")

    # Plot mode
    plot_mode = st.radio(
        "Plot mode",
        PLOT_MODE_OPTIONS,
        horizontal=True,
        key="img_plot_mode",
        help="Plotly: hover values, zoom, pan. Matplotlib: publication-ready static."
    )
    use_plotly = plot_mode.startswith("Interactive")

    # View mode
    view_mode = st.radio(
        "View mode",
        VIEW_MODE_OPTIONS,
        key="img_view_mode",
        help="How to display images"
    )

    # Colormap
    cmap = st.selectbox("Colormap", COLORMAPS, key="img_cmap")

    # Log scale
    use_log_scale = st.checkbox(
        "Log scale",
        key="img_use_log_scale",
        help="Display log₁₀(intensity). Useful for data with large dynamic range (e.g. diffraction)."
    )

    # Intensity controls
    with st.expander("🔆 Intensity", expanded=True):
        auto_contrast = st.checkbox("Auto contrast", key="img_auto_contrast",
                                    help="Automatically adjust intensity range")

        if not auto_contrast:
            vmin_pct = st.slider("Min percentile", 0.0, 50.0, 0.0, 0.5, key="img_vmin_pct",
                                help="Lower intensity cutoff (percentile)")
            vmax_pct = st.slider("Max percentile", 50.0, 100.0, 100.0, 0.5, key="img_vmax_pct",
                                help="Upper intensity cutoff (percentile)")
        else:
            vmin_pct = 0.0
            vmax_pct = 100.0

    # Matplotlib-only controls
    if not use_plotly:
        aspect = st.radio("Aspect ratio", ASPECT_OPTIONS, horizontal=True, key="img_aspect")
        interpolation = st.selectbox(
            "Interpolation",
            INTERPOLATION_OPTIONS,
            key="img_interpolation",
            help="Image interpolation method"
        )
    else:
        aspect = st.session_state.get("img_aspect", ASPECT_OPTIONS[0])
        interpolation = st.session_state.get("img_interpolation", INTERPOLATION_OPTIONS[0])

# ---------------------------------------------------------------------------
# Python Editor (Two-way GUI <-> Code)
# ---------------------------------------------------------------------------

current_image_plot_spec = _build_image_plot_spec(images)
st.session_state["img_current_plot_spec"] = current_image_plot_spec

if not st.session_state.get("img_editor_code"):
    st.session_state["img_editor_code"] = _generate_image_editor_python(current_image_plot_spec)

st.divider()
st.header("🧠 Python Plot Editor (Experimental)")
st.caption("Two-way control: GUI -> Python script -> GUI.")
st.caption("Code runs in the app process. Only run trusted code.")

pending_editor_code = st.session_state.pop("img_editor_code_pending", None)
if pending_editor_code is not None:
    st.session_state["img_editor_code"] = pending_editor_code

editor_code = st.text_area(
    "Editable Python script",
    key="img_editor_code",
    height=280,
    help="Edit plot_spec and click 'Run Edited Python' to update image controls."
)

ed_col1, ed_col2 = st.columns(2)
with ed_col1:
    if st.button("🧾 Show Python from Current GUI", key="img_editor_generate"):
        st.session_state["img_editor_code_pending"] = _generate_image_editor_python(current_image_plot_spec)
        st.session_state["img_editor_status"] = "Editor refreshed from current GUI state."
        st.session_state["img_editor_warnings"] = []
        st.rerun()

with ed_col2:
    if st.button("▶️ Run Edited Python", key="img_editor_run", type="primary"):
        try:
            normalized_spec, warnings = _execute_image_editor(
                editor_code,
                current_image_plot_spec,
                images,
            )
            st.session_state["img_pending_plot_spec"] = normalized_spec
            st.session_state["img_editor_status"] = "Script applied. GUI synced from edited plot_spec."
            st.session_state["img_editor_warnings"] = warnings
            st.rerun()
        except Exception as exc:
            st.session_state["img_editor_status"] = f"Script execution failed: {exc}"
            st.session_state["img_editor_warnings"] = []

if st.session_state.get("img_editor_status"):
    status_text = st.session_state["img_editor_status"]
    if status_text.lower().startswith("script execution failed"):
        st.error(status_text)
    else:
        st.success(status_text)
for warning_msg in st.session_state.get("img_editor_warnings", []):
    st.warning(warning_msg)

# ---------------------------------------------------------------------------
# Main Area: Display Images
# ---------------------------------------------------------------------------

st.header("📊 Image Display")

if view_mode == "Single image":
    # ----- Single image view -----
    selected_file = st.selectbox("Select image", list(images.keys()), key="img_selected_file")
    img_raw = images[selected_file]

    # Handle stacks
    if img_raw.ndim == 3:
        st.info(f"Image stack: {img_raw.shape[0]} frames, "
                f"{img_raw.shape[1]}×{img_raw.shape[2]} pixels")
        frame_key = f"img_frame_{selected_file}"
        if frame_key not in st.session_state:
            st.session_state[frame_key] = 0
        frame_idx = st.slider("Frame", 0, img_raw.shape[0] - 1, 0, key=frame_key)
        img_frame = img_raw[frame_idx]
    else:
        img_frame = img_raw

    # Prepare display data
    display_data, label_suffix = prepare_display_data(img_frame, use_log_scale)
    vmin, vmax = calc_intensity_range(display_data, auto_contrast, vmin_pct, vmax_pct)
    file_stem = Path(selected_file).stem

    if use_plotly:
        fig = make_plotly_heatmap(
            display_data, img_frame.astype(np.float64),
            title=selected_file,
            cmap=cmap, vmin=vmin, vmax=vmax,
            use_log=use_log_scale,
            equal_aspect=True,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.success("**Hover** to see intensity values. **Scroll** to zoom. **Drag** to pan.")

        # Export
        st.divider()
        export_section_plotly(fig, f"{file_stem}_display")

    else:
        mpl_fig, ax = plt.subplots(figsize=(12, 10))
        if use_log_scale:
            im = ax.imshow(img_frame.astype(np.float64), cmap=cmap,
                           norm=mcolors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax),
                           aspect=aspect, interpolation=interpolation)
        else:
            im = ax.imshow(display_data, cmap=cmap, vmin=vmin, vmax=vmax,
                           aspect=aspect, interpolation=interpolation)
        ax.set_title(selected_file, fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label=f"Intensity{label_suffix}")

        st.pyplot(mpl_fig)

        # Export
        st.divider()
        export_section_matplotlib(mpl_fig, f"{file_stem}_display")
        plt.close(mpl_fig)

    # Statistics
    with st.expander("📊 Statistics"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Shape", f"{img_frame.shape[0]}×{img_frame.shape[1]}")
        with col2:
            st.metric("Min", f"{img_frame.min():.2f}")
        with col3:
            st.metric("Max", f"{img_frame.max():.2f}")
        with col4:
            st.metric("Mean", f"{img_frame.mean():.2f}")

elif view_mode == "Side-by-side comparison":
    # ----- Side-by-side comparison -----
    if len(images) < 2:
        st.warning("Need at least 2 images for comparison")
        st.stop()

    selected_files = st.multiselect(
        "Select images to compare",
        list(images.keys()),
        default=st.session_state.get("img_selected_compare_files", list(images.keys())[:min(4, len(images))]),
        key="img_selected_compare_files",
        help="Select 2-4 images"
    )

    if len(selected_files) < 2:
        st.info("Select at least 2 images")
        st.stop()

    if len(selected_files) > 4:
        st.warning("Maximum 4 images for side-by-side. Showing first 4.")
        selected_files = selected_files[:4]

    n_images = len(selected_files)

    # Collect frames (with optional per-image frame slider)
    frames = []
    names = []
    frame_cols = st.columns(n_images)
    for idx, file_name in enumerate(selected_files):
        img_raw = images[file_name]
        with frame_cols[idx]:
            if img_raw.ndim == 3:
                frame_key = f"img_frame_{file_name}"
                if frame_key not in st.session_state:
                    st.session_state[frame_key] = 0
                fi = st.slider(f"Frame: {Path(file_name).stem}",
                               0, img_raw.shape[0] - 1, 0,
                               key=frame_key)
                frames.append(img_raw[fi])
            else:
                frames.append(img_raw)
        names.append(Path(file_name).stem)

    if use_plotly:
        fig = make_plotly_grid(frames, names, cmap, use_log_scale,
                               auto_contrast, vmin_pct, vmax_pct, n_cols=n_images)
        st.plotly_chart(fig, use_container_width=True)
        st.success("**Hover** for intensity. **Scroll** to zoom. **Drag** to pan.")

        st.divider()
        export_section_plotly(fig, "comparison")

    else:
        cols = st.columns(n_images)
        for idx, (col, img_frame, name) in enumerate(zip(cols, frames, names)):
            with col:
                display_data, _ = prepare_display_data(img_frame, use_log_scale)
                vmin, vmax = calc_intensity_range(display_data, auto_contrast,
                                                  vmin_pct, vmax_pct)
                mpl_fig, ax = plt.subplots(figsize=(6, 6))
                if use_log_scale:
                    im = ax.imshow(img_frame.astype(np.float64), cmap=cmap,
                                   norm=mcolors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax),
                                   aspect=aspect, interpolation=interpolation)
                else:
                    im = ax.imshow(display_data, cmap=cmap, vmin=vmin, vmax=vmax,
                                   aspect=aspect, interpolation=interpolation)
                ax.set_title(name, fontsize=10)
                plt.colorbar(im, ax=ax)
                st.pyplot(mpl_fig)
                plt.close(mpl_fig)

else:  # Grid view
    # ----- Grid view of all images -----
    n_images = len(images)
    n_cols_grid = st.slider("Columns", 1, 4, min(3, n_images), key="img_grid_cols")

    # Collect frames (middle frame for stacks)
    grid_frames = []
    grid_names = []
    for file_name, img_raw in images.items():
        if img_raw.ndim == 3:
            grid_frames.append(img_raw[img_raw.shape[0] // 2])
        else:
            grid_frames.append(img_raw)
        grid_names.append(Path(file_name).stem)

    if use_plotly:
        fig = make_plotly_grid(grid_frames, grid_names, cmap, use_log_scale,
                               auto_contrast, vmin_pct, vmax_pct, n_cols=n_cols_grid)
        st.plotly_chart(fig, use_container_width=True)
        st.success("**Hover** for intensity. **Scroll** to zoom. **Drag** to pan.")

        st.divider()
        export_section_plotly(fig, "image_grid")

    else:
        n_rows = (n_images + n_cols_grid - 1) // n_cols_grid
        mpl_fig, axes = plt.subplots(n_rows, n_cols_grid,
                                     figsize=(4 * n_cols_grid, 4 * n_rows))
        if n_images == 1:
            axes = np.array([axes])
        axes = np.array(axes).flatten()

        for idx, (img_frame, name) in enumerate(zip(grid_frames, grid_names)):
            ax = axes[idx]
            display_data, _ = prepare_display_data(img_frame, use_log_scale)
            vmin, vmax = calc_intensity_range(display_data, auto_contrast,
                                              vmin_pct, vmax_pct)
            if use_log_scale:
                im = ax.imshow(img_frame.astype(np.float64), cmap=cmap,
                               norm=mcolors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax),
                               aspect=aspect, interpolation=interpolation)
            else:
                im = ax.imshow(display_data, cmap=cmap, vmin=vmin, vmax=vmax,
                               aspect=aspect, interpolation=interpolation)
            ax.set_title(name, fontsize=9)
            ax.axis('off')

        for idx in range(n_images, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        st.pyplot(mpl_fig)

        st.divider()
        export_section_matplotlib(mpl_fig, "image_grid")
        plt.close(mpl_fig)

# ---------------------------------------------------------------------------
# Image Info
# ---------------------------------------------------------------------------

with st.expander("📄 Image Information"):
    for file_name, img_raw in images.items():
        st.subheader(file_name)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.text(f"Shape: {img_raw.shape}")
            st.text(f"Dimensions: {img_raw.ndim}D")

        with col2:
            st.text(f"Data type: {img_raw.dtype}")
            st.text(f"Size: {img_raw.size} pixels")

        with col3:
            st.text(f"Min: {img_raw.min():.4f}")
            st.text(f"Max: {img_raw.max():.4f}")
            st.text(f"Mean: {img_raw.mean():.4f}")

        st.divider()
