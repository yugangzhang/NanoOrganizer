#!/usr/bin/env python3
"""
3D Plotter - Interactive 3D visualization with Plotly.

Much better than matplotlib - fully interactive, rotatable, zoomable!
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import griddata
import sys
import copy
import pprint

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from components.folder_browser import folder_browser
from components.floating_button import floating_sidebar_toggle
from components.security import (
    initialize_security_context,
    require_authentication,
)

# User-mode restriction (set by nanoorganizer_user)
initialize_security_context()
require_authentication()
_user_mode = st.session_state.get("user_mode", False)
_start_dir = st.session_state.get("user_start_dir", None)

DATA_SOURCE_OPTIONS = ["Upload CSV", "Browse server", "Generate synthetic"]
PLOT_TYPE_OPTIONS = ["Surface", "Scatter 3D", "Contour 3D", "Wireframe", "Mesh"]
FUNCTION_OPTIONS = ["Gaussian", "Ripple", "Saddle", "Volcano", "Waves", "Mexican Hat"]
COLORSCALE_OPTIONS = [
    'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',
    'Turbo', 'Rainbow', 'Jet', 'Hot', 'Cool',
    'Portland', 'Electric', 'Picnic', 'RdBu', 'Earth'
]

st.title("📈 Interactive 3D Plotter (Plotly)")
st.markdown("XYZ + Color dimension - Fully interactive and rotatable!")

# Floating sidebar toggle button (bottom-left)
floating_sidebar_toggle()

st.session_state.setdefault("d3_data_source", DATA_SOURCE_OPTIONS[0])
st.session_state.setdefault("d3_function", FUNCTION_OPTIONS[0])
st.session_state.setdefault("d3_grid_size", 50)
st.session_state.setdefault("d3_x_col", None)
st.session_state.setdefault("d3_y_col", None)
st.session_state.setdefault("d3_z_col", None)
st.session_state.setdefault("d3_use_color", True)
st.session_state.setdefault("d3_color_col", None)
st.session_state.setdefault("d3_plot_type", PLOT_TYPE_OPTIONS[0])
st.session_state.setdefault("d3_colorscale", COLORSCALE_OPTIONS[0])
st.session_state.setdefault("d3_opacity", 0.9)
st.session_state.setdefault("d3_editor_code", "")
st.session_state.setdefault("d3_editor_status", "")
st.session_state.setdefault("d3_editor_warnings", [])

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def load_data_file(file_path):
    """Load CSV or NPZ file."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    try:
        if suffix == '.npz':
            data = np.load(file_path)
            df_dict = {}
            for key in data.files:
                arr = data[key]
                if arr.ndim == 1:
                    df_dict[key] = arr
                elif arr.ndim == 2:
                    df_dict[key] = arr.flatten()
            return pd.DataFrame(df_dict)
        else:
            df = pd.read_csv(file_path, sep=',')
            if len(df.columns) == 1:
                df = pd.read_csv(file_path, sep='\t')
            if len(df.columns) == 1:
                df = pd.read_csv(file_path, sep=r'\s+')
            return df
    except Exception as e:
        st.error(f"Error loading {path.name}: {e}")
        return None


def browse_directory(base_dir, pattern="*.csv"):
    """Browse directory and find files."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    files = list(base_path.rglob(pattern))
    return [str(f) for f in sorted(files)]


def create_meshgrid_from_data(x, y, z, grid_size=100):
    """Create meshgrid from scattered or gridded data."""
    unique_x = np.unique(x)
    unique_y = np.unique(y)

    if len(unique_x) * len(unique_y) == len(z):
        # Data is already gridded
        X, Y = np.meshgrid(unique_x, unique_y)
        Z = z.reshape(len(unique_y), len(unique_x))
    else:
        # Data is scattered - interpolate to grid
        X, Y = np.meshgrid(
            np.linspace(x.min(), x.max(), grid_size),
            np.linspace(y.min(), y.max(), grid_size)
        )
        Z = griddata((x, y), z, (X, Y), method='cubic')

    return X, Y, Z


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


def _build_3d_plot_spec(x_col, y_col, z_col, use_color, color_col, plot_type, colorscale, opacity):
    """Build serializable 3D plot config from current UI state."""
    return {
        "version": 1,
        "x_col": str(x_col),
        "y_col": str(y_col),
        "z_col": str(z_col),
        "use_color": bool(use_color),
        "color_col": str(color_col) if color_col else None,
        "plot_type": str(plot_type),
        "colorscale": str(colorscale),
        "opacity": float(opacity),
    }


def _generate_3d_editor_python(plot_spec):
    """Generate editable Python code for the 3D plotter."""
    spec_text = pprint.pformat(plot_spec, sort_dicts=False, width=100, compact=False)
    return (
        "# NanoOrganizer 3D plot editor (experimental)\n"
        "# Edit plot_spec and click 'Run Edited Python'.\n"
        "# The app uses `result` (if defined) otherwise `plot_spec`.\n\n"
        f"plot_spec = {spec_text}\n\n"
        "# Example tweaks:\n"
        "# plot_spec['plot_type'] = 'Scatter 3D'\n"
        "# plot_spec['opacity'] = 0.6\n\n"
        "result = plot_spec\n"
    )


def _sanitize_3d_plot_spec(candidate, fallback_spec, columns):
    """Sanitize user-edited 3D plot spec and return (spec, warnings)."""
    warnings = []
    if not isinstance(candidate, dict):
        warnings.append("Edited code did not return a dict; keeping previous settings.")
        return copy.deepcopy(fallback_spec), warnings

    normalized = copy.deepcopy(fallback_spec)

    for axis_key in ["x_col", "y_col", "z_col"]:
        proposed = str(candidate.get(axis_key, normalized[axis_key]))
        if proposed in columns:
            normalized[axis_key] = proposed
        elif axis_key in candidate:
            warnings.append(f"Ignored unknown {axis_key} '{proposed}'.")

    proposed_use_color = _to_bool(candidate.get("use_color", normalized.get("use_color", False)), False)
    normalized["use_color"] = proposed_use_color

    proposed_color_col = candidate.get("color_col", normalized.get("color_col"))
    if proposed_color_col is None:
        normalized["color_col"] = None
    else:
        proposed_color_col = str(proposed_color_col)
        if proposed_color_col in columns:
            normalized["color_col"] = proposed_color_col
        else:
            warnings.append(f"Ignored unknown color_col '{proposed_color_col}'.")

    if normalized["use_color"] and not normalized["color_col"]:
        non_axis_cols = [c for c in columns if c not in {normalized["x_col"], normalized["y_col"], normalized["z_col"]}]
        normalized["color_col"] = non_axis_cols[0] if non_axis_cols else normalized["z_col"]

    plot_type = candidate.get("plot_type", normalized.get("plot_type"))
    if plot_type in PLOT_TYPE_OPTIONS:
        normalized["plot_type"] = plot_type
    elif "plot_type" in candidate:
        warnings.append(f"Ignored unsupported plot_type '{plot_type}'.")

    colorscale = candidate.get("colorscale", normalized.get("colorscale"))
    if colorscale in COLORSCALE_OPTIONS:
        normalized["colorscale"] = colorscale
    elif "colorscale" in candidate:
        warnings.append(f"Ignored unsupported colorscale '{colorscale}'.")

    normalized["opacity"] = _to_float(candidate.get("opacity", normalized.get("opacity", 0.9)), 0.9, 0.1, 1.0)

    return normalized, warnings


def _apply_3d_plot_spec_to_state(plot_spec, columns):
    """Apply normalized 3D plot spec to session state before widget creation."""
    if plot_spec.get("x_col") in columns:
        st.session_state["d3_x_col"] = plot_spec["x_col"]
    if plot_spec.get("y_col") in columns:
        st.session_state["d3_y_col"] = plot_spec["y_col"]
    if plot_spec.get("z_col") in columns:
        st.session_state["d3_z_col"] = plot_spec["z_col"]

    if plot_spec.get("plot_type") in PLOT_TYPE_OPTIONS:
        st.session_state["d3_plot_type"] = plot_spec["plot_type"]
    if plot_spec.get("colorscale") in COLORSCALE_OPTIONS:
        st.session_state["d3_colorscale"] = plot_spec["colorscale"]
    st.session_state["d3_opacity"] = _to_float(plot_spec.get("opacity", 0.9), 0.9, 0.1, 1.0)
    st.session_state["d3_use_color"] = _to_bool(plot_spec.get("use_color", False), False)

    color_col = plot_spec.get("color_col")
    if color_col in columns:
        st.session_state["d3_color_col"] = color_col
    elif columns:
        st.session_state["d3_color_col"] = columns[min(3, len(columns) - 1)]


def _execute_3d_editor(code_text, base_plot_spec, columns):
    """Execute editor code and sanitize returned plot spec."""
    execution_locals = {
        "plot_spec": copy.deepcopy(base_plot_spec),
        "result": None,
        "columns": list(columns),
        "copy": copy,
        "np": np,
        "pd": pd,
    }
    exec(code_text, {"__builtins__": __builtins__}, execution_locals)
    candidate = execution_locals.get("result")
    if candidate is None:
        candidate = execution_locals.get("plot_spec")
    return _sanitize_3d_plot_spec(candidate, base_plot_spec, columns)


# ---------------------------------------------------------------------------
# Sidebar: Data Loading
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📁 Data Input")

    data_source = st.radio(
        "Data source",
        DATA_SOURCE_OPTIONS,
        key="d3_data_source"
    )

    df = None

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv', 'txt', 'dat', 'npz']
        )

        if uploaded_file:
            temp_path = Path(f"/tmp/{uploaded_file.name}")
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            df = load_data_file(str(temp_path))

    elif data_source == "Browse server":
        st.markdown("**🗂️ Interactive Folder Browser**")
        st.markdown("Click folders to navigate, select file:")

        # File pattern selector
        st.markdown("**📋 File Type Filter:**")
        pattern = st.selectbox(
            "Extension pattern",
            ["*.*", "*.csv", "*.npz", "*.txt", "*.dat"],
            help="Filter files by extension",
            label_visibility="collapsed"
        )

        st.info("💡 Tip: Use '🔍 Advanced Filters' below for name-based filtering")

        # Use folder browser component (single select for 3D)
        selected_files = folder_browser(
            key="3d_plotter_browser",
            show_files=True,
            file_pattern=pattern,
            multi_select=False,
            initial_path=_start_dir if _user_mode else None,
            restrict_to_start_dir=_user_mode,
        )

        # Load button
        if selected_files and st.button("📥 Load Selected File", key="3d_load_btn"):
            df = load_data_file(selected_files[0])
            if df is not None:
                st.session_state['loaded_3d_df'] = df
                st.success(f"✅ Loaded {Path(selected_files[0]).name}")

        if 'loaded_3d_df' in st.session_state:
            df = st.session_state['loaded_3d_df']

    else:  # Generate synthetic
        st.subheader("Synthetic Data")

        func_type = st.selectbox(
            "Function",
            FUNCTION_OPTIONS,
            key="d3_function"
        )

        grid_size = st.slider("Grid size", 20, 200, 50, 10, key="d3_grid_size")

        if st.button("🎲 Generate"):
            x = np.linspace(-5, 5, grid_size)
            y = np.linspace(-5, 5, grid_size)
            X, Y = np.meshgrid(x, y)

            if func_type == "Gaussian":
                Z = np.exp(-(X**2 + Y**2) / 10)
                H = Z * (X**2 + Y**2)  # Color based on position * height
            elif func_type == "Ripple":
                Z = np.sin(np.sqrt(X**2 + Y**2))
                H = np.cos(np.sqrt(X**2 + Y**2))
            elif func_type == "Saddle":
                Z = X**2 - Y**2
                H = np.abs(X) + np.abs(Y)
            elif func_type == "Volcano":
                Z = -np.exp(-(X**2 + Y**2) / 10) + 0.1 * (X**2 + Y**2)
                H = X**2 + Y**2
            elif func_type == "Waves":
                Z = np.sin(X) * np.cos(Y)
                H = np.cos(X) * np.sin(Y)
            else:  # Mexican Hat
                r = np.sqrt(X**2 + Y**2)
                Z = (1 - r**2) * np.exp(-r**2 / 2)
                H = r

            df = pd.DataFrame({
                'X': X.flatten(),
                'Y': Y.flatten(),
                'Z': Z.flatten(),
                'H': H.flatten()
            })
            st.session_state['synth_df_3d'] = df
            st.success(f"Generated {len(df)} points")

        if 'synth_df_3d' in st.session_state:
            df = st.session_state['synth_df_3d']

    if df is None:
        st.info("👆 Load or generate data")
        st.stop()

    # Column selection
    st.header("📐 Column Selection")

    columns = list(df.columns)

    if "d3_pending_plot_spec" in st.session_state:
        pending_spec = st.session_state.pop("d3_pending_plot_spec")
        _apply_3d_plot_spec_to_state(pending_spec, columns)

    if not st.session_state.get("d3_x_col") or st.session_state.get("d3_x_col") not in columns:
        st.session_state["d3_x_col"] = columns[0]
    if not st.session_state.get("d3_y_col") or st.session_state.get("d3_y_col") not in columns:
        st.session_state["d3_y_col"] = columns[min(1, len(columns) - 1)]
    if not st.session_state.get("d3_z_col") or st.session_state.get("d3_z_col") not in columns:
        st.session_state["d3_z_col"] = columns[min(2, len(columns) - 1)]
    if not st.session_state.get("d3_color_col") or st.session_state.get("d3_color_col") not in columns:
        st.session_state["d3_color_col"] = columns[min(3, len(columns) - 1)] if columns else None

    x_col = st.selectbox("X-axis", columns, key="d3_x_col")
    y_col = st.selectbox("Y-axis", columns, key="d3_y_col")
    z_col = st.selectbox("Z-axis", columns, key="d3_z_col")

    use_color = st.checkbox("Use 4th dimension for color", key="d3_use_color", value=len(columns) > 3)
    if use_color and len(columns) > 3:
        color_col = st.selectbox("Color column", columns, key="d3_color_col")
    else:
        color_col = None

    # Plot type
    st.header("📊 Plot Type")

    plot_type = st.radio(
        "Type",
        PLOT_TYPE_OPTIONS,
        key="d3_plot_type",
        help="All plots are fully interactive!"
    )

    # Colorscale
    st.header("🎨 Style")

    colorscale = st.selectbox("Colorscale", COLORSCALE_OPTIONS, key="d3_colorscale")

    # Opacity
    opacity = st.slider("Opacity", 0.1, 1.0, 0.9, 0.1, key="d3_opacity")

# ---------------------------------------------------------------------------
# Main Area: 3D Plot
# ---------------------------------------------------------------------------

current_3d_plot_spec = _build_3d_plot_spec(
    x_col=x_col,
    y_col=y_col,
    z_col=z_col,
    use_color=use_color,
    color_col=color_col,
    plot_type=plot_type,
    colorscale=colorscale,
    opacity=opacity,
)
st.session_state["d3_current_plot_spec"] = current_3d_plot_spec

if not st.session_state.get("d3_editor_code"):
    st.session_state["d3_editor_code"] = _generate_3d_editor_python(current_3d_plot_spec)

st.header("🧠 Python Plot Editor (Experimental)")
st.caption("Two-way control: GUI -> Python script -> GUI.")
st.caption("Code runs in the app process. Only run trusted code.")

pending_editor_code = st.session_state.pop("d3_editor_code_pending", None)
if pending_editor_code is not None:
    st.session_state["d3_editor_code"] = pending_editor_code

editor_code = st.text_area(
    "Editable Python script",
    key="d3_editor_code",
    height=280,
    help="Edit plot_spec and click 'Run Edited Python' to update controls."
)

ed_col1, ed_col2 = st.columns(2)
with ed_col1:
    if st.button("🧾 Show Python from Current GUI", key="d3_editor_generate"):
        st.session_state["d3_editor_code_pending"] = _generate_3d_editor_python(current_3d_plot_spec)
        st.session_state["d3_editor_status"] = "Editor refreshed from current GUI state."
        st.session_state["d3_editor_warnings"] = []
        st.rerun()

with ed_col2:
    if st.button("▶️ Run Edited Python", key="d3_editor_run", type="primary"):
        try:
            normalized_spec, warnings = _execute_3d_editor(
                editor_code,
                current_3d_plot_spec,
                columns,
            )
            st.session_state["d3_pending_plot_spec"] = normalized_spec
            st.session_state["d3_editor_status"] = "Script applied. GUI synced from edited plot_spec."
            st.session_state["d3_editor_warnings"] = warnings
            st.rerun()
        except Exception as exc:
            st.session_state["d3_editor_status"] = f"Script execution failed: {exc}"
            st.session_state["d3_editor_warnings"] = []

if st.session_state.get("d3_editor_status"):
    status_text = st.session_state["d3_editor_status"]
    if status_text.lower().startswith("script execution failed"):
        st.error(status_text)
    else:
        st.success(status_text)
for warning_msg in st.session_state.get("d3_editor_warnings", []):
    st.warning(warning_msg)

st.divider()
st.header(f"📊 Interactive 3D Plot: {plot_type}")

# Get data
x = df[x_col].values
y = df[y_col].values
z = df[z_col].values
c = df[color_col].values if color_col else z

# Create figure
fig = go.Figure()

if plot_type == "Surface":
    # Create mesh grid
    X, Y, Z = create_meshgrid_from_data(x, y, z, grid_size=100)

    if color_col:
        C, _, _ = create_meshgrid_from_data(x, y, c, grid_size=100)
    else:
        C = Z

    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=C,
        colorscale=colorscale,
        opacity=opacity,
        colorbar=dict(title=color_col if color_col else z_col)
    ))

elif plot_type == "Scatter 3D":
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=c,
            colorscale=colorscale,
            opacity=opacity,
            colorbar=dict(title=color_col if color_col else z_col)
        )
    ))

elif plot_type == "Contour 3D":
    fig.add_trace(go.Isosurface(
        x=x, y=y, z=z,
        value=c,
        colorscale=colorscale,
        opacity=opacity,
        caps=dict(x_show=False, y_show=False),
        colorbar=dict(title=color_col if color_col else z_col)
    ))

elif plot_type == "Wireframe":
    X, Y, Z = create_meshgrid_from_data(x, y, z, grid_size=50)

    # Create wireframe by adding lines
    for i in range(0, X.shape[0], 2):
        fig.add_trace(go.Scatter3d(
            x=X[i, :], y=Y[i, :], z=Z[i, :],
            mode='lines',
            line=dict(color='blue', width=1),
            showlegend=False
        ))

    for j in range(0, X.shape[1], 2):
        fig.add_trace(go.Scatter3d(
            x=X[:, j], y=Y[:, j], z=Z[:, j],
            mode='lines',
            line=dict(color='blue', width=1),
            showlegend=False
        ))

else:  # Mesh
    X, Y, Z = create_meshgrid_from_data(x, y, z, grid_size=100)

    fig.add_trace(go.Mesh3d(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        intensity=c,
        colorscale=colorscale,
        opacity=opacity,
        colorbar=dict(title=color_col if color_col else z_col)
    ))

# Update layout
fig.update_layout(
    title=f"{plot_type} Plot: {z_col}",
    scene=dict(
        xaxis_title=x_col,
        yaxis_title=y_col,
        zaxis_title=z_col,
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    ),
    width=1000,
    height=700,
    font=dict(size=12)
)

# Show plot (fully interactive!)
st.plotly_chart(fig, use_container_width=True)

st.info("""
🖱️ **Interactive Controls:**
- **Rotate**: Click and drag
- **Zoom**: Scroll wheel
- **Pan**: Right-click and drag
- **Reset**: Double-click
- **Export**: Click camera icon in toolbar (PNG, SVG, HTML)
""")

# Download options
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    # Save as HTML (interactive!)
    html_str = fig.to_html(include_plotlyjs='cdn')
    st.download_button(
        label="💾 Download Interactive HTML",
        data=html_str,
        file_name="3d_plot_interactive.html",
        mime="text/html"
    )

with col2:
    # Save as PNG
    try:
        img_bytes = fig.to_image(format="png", width=1200, height=900)
        st.download_button(
            label="💾 Download PNG",
            data=img_bytes,
            file_name="3d_plot.png",
            mime="image/png"
        )
    except:
        st.info("Install kaleido for PNG export: `pip install kaleido`")

with col3:
    st.metric("Data Points", len(df))

# Statistics
with st.expander("📊 Data Statistics"):
    col1, col2, col3 = st.columns(3)

    for col, col_name in zip([col1, col2, col3], [x_col, y_col, z_col]):
        with col:
            st.markdown(f"**{col_name}**")
            vals = df[col_name].values
            st.text(f"Min:  {vals.min():.4f}")
            st.text(f"Max:  {vals.max():.4f}")
            st.text(f"Mean: {vals.mean():.4f}")
            st.text(f"Std:  {vals.std():.4f}")
