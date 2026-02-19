#!/usr/bin/env python3
"""
Multi-Axes Plotter - Create complex multi-panel figures.

Features:
- Multiple subplots with flexible layouts
- Assign specific data to specific axes
- Dynamic layout adjustment
- Publication-ready multi-panel figures
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402

import streamlit as st  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pathlib import Path  # noqa: E402
import io  # noqa: E402
import sys  # noqa: E402
import copy  # noqa: E402
import pprint  # noqa: E402
import math  # noqa: E402

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from components.folder_browser import folder_browser  # noqa: E402
from components.floating_button import floating_sidebar_toggle  # noqa: E402
from components.security import (  # noqa: E402
    initialize_security_context,
    require_authentication,
)

# User-mode restriction (set by nanoorganizer_user)
initialize_security_context()
require_authentication()
_user_mode = st.session_state.get("user_mode", False)
_start_dir = st.session_state.get("user_start_dir", None)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
MARKERS = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']
LAYOUT_OPTIONS = ["Grid (rows × cols)", "Custom positions"]
SCALE_OPTIONS = ["linear", "log"]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def load_data_file(file_path):
    """Load CSV, TXT, or NPZ file."""
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
        st.error(f"Error loading {Path(file_path).name}: {e}")
        return None


def browse_directory(base_dir, pattern="*.csv"):
    """Browse directory and find files."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    files = list(base_path.rglob(pattern))
    return [str(f) for f in sorted(files)]


def _save_fig_to_bytes(fig, format='png', dpi=300):
    """Save matplotlib figure to bytes buffer."""
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf


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


def _sanitize_multi_plot_spec(candidate, fallback_spec, dataframes):
    """Sanitize user-edited multi-axes spec and return (spec, warnings)."""
    warnings = []
    if not isinstance(candidate, dict):
        warnings.append("Edited code did not return a dict; keeping previous settings.")
        return copy.deepcopy(fallback_spec), warnings

    normalized = copy.deepcopy(fallback_spec)

    layout = candidate.get("layout", {})
    if not isinstance(layout, dict):
        layout = {}
        warnings.append("layout must be a dict; keeping previous layout.")

    layout_type = layout.get("type", normalized["layout"]["type"])
    if layout_type in LAYOUT_OPTIONS:
        normalized["layout"]["type"] = layout_type
    elif "type" in layout:
        warnings.append(f"Ignored unsupported layout type '{layout_type}'.")

    normalized["layout"]["n_rows"] = _to_int(layout.get("n_rows", normalized["layout"]["n_rows"]), 2, 1, 5)
    normalized["layout"]["n_cols"] = _to_int(layout.get("n_cols", normalized["layout"]["n_cols"]), 2, 1, 5)
    normalized["layout"]["n_axes"] = _to_int(layout.get("n_axes", normalized["layout"]["n_axes"]), 4, 1, 9)

    figure = candidate.get("figure", {})
    if not isinstance(figure, dict):
        figure = {}
        warnings.append("figure must be a dict; keeping previous figure size.")
    normalized["figure"] = {
        "width": _to_int(figure.get("width", normalized["figure"]["width"]), 12, 6, 20),
        "height": _to_int(figure.get("height", normalized["figure"]["height"]), 8, 4, 16),
    }

    if normalized["layout"]["type"] == "Grid (rows × cols)":
        effective_axes = normalized["layout"]["n_rows"] * normalized["layout"]["n_cols"]
    else:
        effective_axes = normalized["layout"]["n_axes"]

    raw_assignments = candidate.get("axes_assignments", {})
    if not isinstance(raw_assignments, dict):
        raw_assignments = {}
        warnings.append("axes_assignments must be a dict; keeping previous axis settings.")

    fallback_assignments = fallback_spec.get("axes_assignments", {})
    normalized_assignments = {}

    dataset_names = list(dataframes.keys())
    for ax_idx in range(effective_axes):
        ax_key = str(ax_idx)
        incoming = raw_assignments.get(ax_key, raw_assignments.get(ax_idx))
        if incoming is None:
            incoming = fallback_assignments.get(ax_key, {})
        if not isinstance(incoming, dict):
            incoming = {}

        fallback_axis = fallback_assignments.get(ax_key, {})

        datasets = incoming.get("datasets", fallback_axis.get("datasets", []))
        if isinstance(datasets, str):
            datasets = [datasets]
        if not isinstance(datasets, (list, tuple, set)):
            datasets = []
        valid_datasets = []
        for ds in datasets:
            ds = str(ds)
            if ds in dataset_names and ds not in valid_datasets:
                valid_datasets.append(ds)

        axis_cfg = {"datasets": valid_datasets}
        if valid_datasets:
            first_df = dataframes[valid_datasets[0]]
            all_cols = list(first_df.columns)
            default_x = all_cols[0]
            default_y = all_cols[min(1, len(all_cols) - 1)]

            x_col = str(incoming.get("x_col", fallback_axis.get("x_col", default_x)))
            y_col = str(incoming.get("y_col", fallback_axis.get("y_col", default_y)))
            if x_col not in all_cols:
                x_col = default_x
            if y_col not in all_cols:
                y_col = default_y

            x_scale = str(incoming.get("x_scale", fallback_axis.get("x_scale", "linear")))
            y_scale = str(incoming.get("y_scale", fallback_axis.get("y_scale", "linear")))
            if x_scale not in SCALE_OPTIONS:
                x_scale = "linear"
            if y_scale not in SCALE_OPTIONS:
                y_scale = "linear"

            title_default = f"Axis {ax_idx+1}"
            axis_cfg.update({
                "x_col": x_col,
                "y_col": y_col,
                "x_scale": x_scale,
                "y_scale": y_scale,
                "show_legend": _to_bool(incoming.get("show_legend", fallback_axis.get("show_legend", True)), True),
                "title": str(incoming.get("title", fallback_axis.get("title", title_default))),
                "xlabel": str(incoming.get("xlabel", fallback_axis.get("xlabel", x_col))),
                "ylabel": str(incoming.get("ylabel", fallback_axis.get("ylabel", y_col))),
            })
        normalized_assignments[ax_key] = axis_cfg

    normalized["axes_assignments"] = normalized_assignments
    return normalized, warnings


def _apply_multi_plot_spec_to_state(plot_spec, dataframes):
    """Apply normalized multi-axes spec into session state before widgets render."""
    layout = plot_spec.get("layout", {})
    st.session_state["multi_layout_type"] = layout.get("type", LAYOUT_OPTIONS[0])
    st.session_state["multi_n_rows"] = _to_int(layout.get("n_rows", 2), 2, 1, 5)
    st.session_state["multi_n_cols"] = _to_int(layout.get("n_cols", 2), 2, 1, 5)
    st.session_state["multi_n_axes"] = _to_int(layout.get("n_axes", 4), 4, 1, 9)

    figure = plot_spec.get("figure", {})
    st.session_state["multi_fig_width"] = _to_int(figure.get("width", 12), 12, 6, 20)
    st.session_state["multi_fig_height"] = _to_int(figure.get("height", 8), 8, 4, 16)

    if st.session_state["multi_layout_type"] == "Grid (rows × cols)":
        n_axes = st.session_state["multi_n_rows"] * st.session_state["multi_n_cols"]
    else:
        n_axes = st.session_state["multi_n_axes"]

    axes_assignments = plot_spec.get("axes_assignments", {})
    st.session_state["axes_assignments"] = {}
    for ax_idx in range(n_axes):
        ax_key = str(ax_idx)
        assignment = axes_assignments.get(ax_key, {})
        if not isinstance(assignment, dict):
            assignment = {}
        datasets = assignment.get("datasets", [])
        if isinstance(datasets, str):
            datasets = [datasets]
        if not isinstance(datasets, (list, tuple, set)):
            datasets = []
        datasets = [str(ds) for ds in datasets if str(ds) in dataframes]

        st.session_state[f"datasets_{ax_idx}"] = datasets
        st.session_state["axes_assignments"][ax_idx] = {"datasets": datasets}

        if datasets:
            all_cols = list(dataframes[datasets[0]].columns)
            default_x = all_cols[0]
            default_y = all_cols[min(1, len(all_cols) - 1)]
            x_col = assignment.get("x_col", default_x)
            y_col = assignment.get("y_col", default_y)
            if x_col not in all_cols:
                x_col = default_x
            if y_col not in all_cols:
                y_col = default_y

            x_scale = assignment.get("x_scale", "linear")
            y_scale = assignment.get("y_scale", "linear")
            if x_scale not in SCALE_OPTIONS:
                x_scale = "linear"
            if y_scale not in SCALE_OPTIONS:
                y_scale = "linear"

            st.session_state[f"x_{ax_idx}"] = x_col
            st.session_state[f"y_{ax_idx}"] = y_col
            st.session_state[f"xscale_{ax_idx}"] = x_scale
            st.session_state[f"yscale_{ax_idx}"] = y_scale
            st.session_state[f"legend_{ax_idx}"] = _to_bool(assignment.get("show_legend", True), True)
            st.session_state[f"title_{ax_idx}"] = str(assignment.get("title", f"Axis {ax_idx+1}"))
            st.session_state[f"xlabel_{ax_idx}"] = str(assignment.get("xlabel", x_col))
            st.session_state[f"ylabel_{ax_idx}"] = str(assignment.get("ylabel", y_col))


def _build_multi_plot_spec(layout_type, n_rows, n_cols, n_axes, fig_width, fig_height, axes_assignments):
    """Build serializable multi-axes spec from current UI state."""
    assignments = {}
    for ax_idx, cfg in axes_assignments.items():
        assignments[str(ax_idx)] = copy.deepcopy(cfg)
    return {
        "version": 1,
        "layout": {
            "type": layout_type,
            "n_rows": int(n_rows),
            "n_cols": int(n_cols),
            "n_axes": int(n_axes),
        },
        "figure": {
            "width": int(fig_width),
            "height": int(fig_height),
        },
        "axes_assignments": assignments,
    }


def _generate_multi_editor_python(plot_spec):
    """Generate editable Python code for multi-axes plot config."""
    spec_text = pprint.pformat(plot_spec, sort_dicts=False, width=100, compact=False)
    return (
        "# NanoOrganizer multi-axes editor (experimental)\n"
        "# Edit plot_spec and click 'Run Edited Python'.\n"
        "# The app uses `result` (if defined) otherwise `plot_spec`.\n\n"
        f"plot_spec = {spec_text}\n\n"
        "# Example tweaks:\n"
        "# plot_spec['figure']['width'] = 16\n"
        "# plot_spec['axes_assignments']['0']['x_scale'] = 'log'\n\n"
        "result = plot_spec\n"
    )


def _execute_multi_editor(code_text, base_plot_spec, dataframes):
    """Execute editor code and sanitize returned multi-axes spec."""
    execution_locals = {
        "plot_spec": copy.deepcopy(base_plot_spec),
        "result": None,
        "data_files": list(dataframes.keys()),
        "columns_by_file": {k: list(v.columns) for k, v in dataframes.items()},
        "copy": copy,
        "np": np,
        "pd": pd,
    }
    exec(code_text, {"__builtins__": __builtins__}, execution_locals)
    candidate = execution_locals.get("result")
    if candidate is None:
        candidate = execution_locals.get("plot_spec")
    return _sanitize_multi_plot_spec(candidate, base_plot_spec, dataframes)


# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------

if 'axes_assignments' not in st.session_state:
    st.session_state['axes_assignments'] = {}

if 'dataframes_multi' not in st.session_state:
    st.session_state['dataframes_multi'] = {}

if 'file_paths_multi' not in st.session_state:
    st.session_state['file_paths_multi'] = {}

st.session_state.setdefault("multi_layout_type", LAYOUT_OPTIONS[0])
st.session_state.setdefault("multi_n_rows", 2)
st.session_state.setdefault("multi_n_cols", 2)
st.session_state.setdefault("multi_n_axes", 4)
st.session_state.setdefault("multi_fig_width", 12)
st.session_state.setdefault("multi_fig_height", 8)
st.session_state.setdefault("multi_editor_code", "")
st.session_state.setdefault("multi_editor_status", "")
st.session_state.setdefault("multi_editor_warnings", [])

# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

st.title("📐 Multi-Axes Plotter")
st.markdown("Create complex multi-panel figures with flexible layouts")

# Floating sidebar toggle button (bottom-left)
floating_sidebar_toggle()

# ---------------------------------------------------------------------------
# Sidebar: Load Data
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📁 Load Data")

    data_source = st.radio(
        "Data source",
        ["Upload files", "Browse server"],
        key="multi_data_source"
    )

    if data_source == "Upload files":
        uploaded_files = st.file_uploader(
            "Upload data files",
            type=['csv', 'txt', 'dat', 'npz'],
            accept_multiple_files=True
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    temp_path = Path(f"/tmp/{uploaded_file.name}")
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())

                    df = load_data_file(str(temp_path))
                    if df is not None:
                        st.session_state['dataframes_multi'][uploaded_file.name] = df
                        st.session_state['file_paths_multi'][uploaded_file.name] = uploaded_file.name
                except Exception as e:
                    st.error(f"Error: {e}")

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

        st.info("💡 Tip: Use '🔍 Advanced Filters' below for name-based filtering")

        # Use folder browser component
        selected_files = folder_browser(
            key="multi_axes_browser",
            show_files=True,
            file_pattern=pattern,
            multi_select=True,
            initial_path=_start_dir if _user_mode else None,
            restrict_to_start_dir=_user_mode,
        )

        # Load button
        if selected_files and st.button("📥 Load Selected Files", key="multi_load_btn"):
            for full_path in selected_files:
                df = load_data_file(full_path)
                if df is not None:
                    file_name = Path(full_path).name
                    st.session_state['dataframes_multi'][file_name] = df
                    st.session_state['file_paths_multi'][file_name] = full_path
                    st.success(f"✅ Loaded {file_name}")

    # Get dataframes from session state
    dataframes = st.session_state['dataframes_multi']
    file_paths = st.session_state['file_paths_multi']

    # Clear button
    if dataframes:
        if st.button("🗑️ Clear All Data", key="clear_multi_data"):
            st.session_state['dataframes_multi'] = {}
            st.session_state['file_paths_multi'] = {}
            st.session_state['axes_assignments'] = {}
            st.session_state.pop("multi_pending_plot_spec", None)
            st.rerun()

    if not dataframes:
        st.info("👆 Upload or select data files to get started")
        st.stop()

    st.success(f"✅ Loaded {len(dataframes)} file(s)")

    if "multi_pending_plot_spec" in st.session_state:
        pending_spec = st.session_state.pop("multi_pending_plot_spec")
        _apply_multi_plot_spec_to_state(pending_spec, dataframes)

    # ---------------------------------------------------------------------------
    # Layout Configuration
    # ---------------------------------------------------------------------------

    st.header("📐 Layout")

    layout_type = st.radio(
        "Layout type",
        LAYOUT_OPTIONS,
        key="multi_layout_type",
        help="Grid for regular layouts, custom for complex arrangements"
    )

    if layout_type == "Grid (rows × cols)":
        col1, col2 = st.columns(2)
        with col1:
            n_rows = st.number_input("Rows", 1, 5, 2, 1, key="multi_n_rows")
        with col2:
            n_cols = st.number_input("Columns", 1, 5, 2, 1, key="multi_n_cols")

        n_axes = n_rows * n_cols
        axes_labels = [f"({i//n_cols+1},{i%n_cols+1})" for i in range(n_axes)]

    else:  # Custom
        n_axes = st.number_input("Number of axes", 1, 9, 4, 1, key="multi_n_axes")
        axes_labels = [f"Axis {i+1}" for i in range(n_axes)]
        n_cols = int(max(1, math.ceil(math.sqrt(n_axes))))
        n_rows = int(max(1, math.ceil(n_axes / n_cols)))

    # Figure size
    st.subheader("📏 Figure Size")
    col1, col2 = st.columns(2)
    with col1:
        fig_width = st.slider("Width (inches)", 6, 20, 12, 1, key="multi_fig_width")
    with col2:
        fig_height = st.slider("Height (inches)", 4, 16, 8, 1, key="multi_fig_height")

# ---------------------------------------------------------------------------
# Main Area: Data Assignment
# ---------------------------------------------------------------------------

st.header("🎯 Data Assignment")

st.markdown("Assign data files to specific axes for plotting")

# Create tabs for each axis
tabs = st.tabs(axes_labels)

for ax_idx, (tab, ax_label) in enumerate(zip(tabs, axes_labels)):
    with tab:
        st.subheader(f"Configure {ax_label}")

        # Seed widget state from stored assignments when available.
        stored_cfg = st.session_state['axes_assignments'].get(ax_idx, {})
        if isinstance(stored_cfg, dict):
            if f"datasets_{ax_idx}" not in st.session_state and stored_cfg.get("datasets"):
                st.session_state[f"datasets_{ax_idx}"] = [
                    ds for ds in stored_cfg.get("datasets", [])
                    if ds in dataframes
                ]

        # Select which datasets to plot on this axis
        selected_datasets = st.multiselect(
            "Select datasets",
            list(dataframes.keys()),
            key=f"datasets_{ax_idx}",
            help="Select one or more datasets to plot on this axis"
        )

        if not selected_datasets:
            st.session_state['axes_assignments'][ax_idx] = {'datasets': []}
            st.info("No datasets selected for this axis")
            continue

        # Get columns from first dataset
        first_df = dataframes[selected_datasets[0]]
        all_cols = list(first_df.columns)

        default_x = all_cols[0]
        default_y = all_cols[min(1, len(all_cols)-1)]
        if st.session_state.get(f"x_{ax_idx}") not in all_cols:
            st.session_state[f"x_{ax_idx}"] = stored_cfg.get('x_col', default_x) if stored_cfg else default_x
        if st.session_state.get(f"y_{ax_idx}") not in all_cols:
            st.session_state[f"y_{ax_idx}"] = stored_cfg.get('y_col', default_y) if stored_cfg else default_y
        if st.session_state.get(f"xscale_{ax_idx}") not in SCALE_OPTIONS:
            st.session_state[f"xscale_{ax_idx}"] = stored_cfg.get('x_scale', 'linear') if stored_cfg else 'linear'
        if st.session_state.get(f"yscale_{ax_idx}") not in SCALE_OPTIONS:
            st.session_state[f"yscale_{ax_idx}"] = stored_cfg.get('y_scale', 'linear') if stored_cfg else 'linear'
        if f"legend_{ax_idx}" not in st.session_state:
            st.session_state[f"legend_{ax_idx}"] = _to_bool(stored_cfg.get('show_legend', True), True) if stored_cfg else True
        if f"title_{ax_idx}" not in st.session_state:
            st.session_state[f"title_{ax_idx}"] = stored_cfg.get('title', ax_label) if stored_cfg else ax_label
        if f"xlabel_{ax_idx}" not in st.session_state:
            st.session_state[f"xlabel_{ax_idx}"] = stored_cfg.get('xlabel', st.session_state[f"x_{ax_idx}"]) if stored_cfg else st.session_state[f"x_{ax_idx}"]
        if f"ylabel_{ax_idx}" not in st.session_state:
            st.session_state[f"ylabel_{ax_idx}"] = stored_cfg.get('ylabel', st.session_state[f"y_{ax_idx}"]) if stored_cfg else st.session_state[f"y_{ax_idx}"]

        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox(
                "X-axis column",
                all_cols,
                key=f"x_{ax_idx}",
                help="Column to use for X-axis"
            )

        with col2:
            y_col = st.selectbox(
                "Y-axis column",
                all_cols,
                index=min(1, len(all_cols)-1),
                key=f"y_{ax_idx}",
                help="Column to use for Y-axis"
            )

        # Plot style
        col1, col2, col3 = st.columns(3)
        with col1:
            x_scale = st.radio(
                "X scale",
                SCALE_OPTIONS,
                key=f"xscale_{ax_idx}",
                horizontal=True
            )
        with col2:
            y_scale = st.radio(
                "Y scale",
                SCALE_OPTIONS,
                key=f"yscale_{ax_idx}",
                horizontal=True
            )
        with col3:
            show_legend = st.checkbox(
                "Show legend",
                value=True,
                key=f"legend_{ax_idx}"
            )

        # Labels
        col1, col2, col3 = st.columns(3)
        with col1:
            title = st.text_input(
                "Title",
                value=f"{ax_label}",
                key=f"title_{ax_idx}"
            )
        with col2:
            xlabel = st.text_input(
                "X label",
                value=x_col,
                key=f"xlabel_{ax_idx}"
            )
        with col3:
            ylabel = st.text_input(
                "Y label",
                value=y_col,
                key=f"ylabel_{ax_idx}"
            )

        # Store assignment
        st.session_state['axes_assignments'][ax_idx] = {
            'datasets': selected_datasets,
            'x_col': x_col,
            'y_col': y_col,
            'x_scale': x_scale,
            'y_scale': y_scale,
            'show_legend': show_legend,
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel
        }

# ---------------------------------------------------------------------------
# Python Editor (Two-way GUI <-> Code)
# ---------------------------------------------------------------------------

current_multi_plot_spec = _build_multi_plot_spec(
    layout_type=layout_type,
    n_rows=n_rows,
    n_cols=n_cols,
    n_axes=n_axes,
    fig_width=fig_width,
    fig_height=fig_height,
    axes_assignments=st.session_state['axes_assignments'],
)
st.session_state["multi_current_plot_spec"] = current_multi_plot_spec

if not st.session_state.get("multi_editor_code"):
    st.session_state["multi_editor_code"] = _generate_multi_editor_python(current_multi_plot_spec)

st.divider()
st.header("🧠 Python Plot Editor (Experimental)")
st.caption("Two-way control: GUI -> Python script -> GUI.")
st.caption("Code runs in the app process. Only run trusted code.")

pending_editor_code = st.session_state.pop("multi_editor_code_pending", None)
if pending_editor_code is not None:
    st.session_state["multi_editor_code"] = pending_editor_code

editor_code = st.text_area(
    "Editable Python script",
    key="multi_editor_code",
    height=300,
    help="Edit plot_spec and click 'Run Edited Python' to sync layout and axis configs."
)

ed_col1, ed_col2 = st.columns(2)
with ed_col1:
    if st.button("🧾 Show Python from Current GUI", key="multi_editor_generate"):
        st.session_state["multi_editor_code_pending"] = _generate_multi_editor_python(current_multi_plot_spec)
        st.session_state["multi_editor_status"] = "Editor refreshed from current GUI state."
        st.session_state["multi_editor_warnings"] = []
        st.rerun()

with ed_col2:
    if st.button("▶️ Run Edited Python", key="multi_editor_run", type="primary"):
        try:
            normalized_spec, warnings = _execute_multi_editor(
                editor_code,
                current_multi_plot_spec,
                dataframes,
            )
            st.session_state["multi_pending_plot_spec"] = normalized_spec
            st.session_state["multi_editor_status"] = "Script applied. GUI synced from edited plot_spec."
            st.session_state["multi_editor_warnings"] = warnings
            st.rerun()
        except Exception as exc:
            st.session_state["multi_editor_status"] = f"Script execution failed: {exc}"
            st.session_state["multi_editor_warnings"] = []

if st.session_state.get("multi_editor_status"):
    status_text = st.session_state["multi_editor_status"]
    if status_text.lower().startswith("script execution failed"):
        st.error(status_text)
    else:
        st.success(status_text)
for warning_msg in st.session_state.get("multi_editor_warnings", []):
    st.warning(warning_msg)

# ---------------------------------------------------------------------------
# Create Figure
# ---------------------------------------------------------------------------

st.divider()
st.header("📊 Multi-Panel Figure")

# Create figure and axes
if layout_type == "Grid (rows × cols)":
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if n_axes == 1:
        axes = np.array([axes])
    axes = axes.flatten()
else:
    # Custom layout with gridspec
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    axes = [fig.add_subplot(gs[i]) for i in range(n_axes)]

# Plot on each axis
for ax_idx, ax in enumerate(axes):
    assignment = st.session_state['axes_assignments'].get(ax_idx, {})

    if not assignment or not assignment.get('datasets'):
        # Empty axis
        ax.text(0.5, 0.5, f"No data\n{axes_labels[ax_idx]}",
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        continue

    # Plot datasets
    datasets = assignment['datasets']
    x_col = assignment['x_col']
    y_col = assignment['y_col']

    for idx, file_name in enumerate(datasets):
        df = dataframes[file_name]

        if x_col not in df.columns or y_col not in df.columns:
            continue

        x_data = df[x_col].values
        y_data = df[y_col].values

        # Remove NaN
        mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_data = x_data[mask]
        y_data = y_data[mask]

        if len(x_data) == 0:
            continue

        # Plot
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]

        ax.plot(x_data, y_data,
               color=color,
               marker=marker,
               markersize=4,
               linewidth=2,
               alpha=0.8,
               label=Path(file_name).stem,
               markevery=max(1, len(x_data)//20))

    # Apply settings
    ax.set_xscale(assignment['x_scale'])
    ax.set_yscale(assignment['y_scale'])
    ax.set_xlabel(assignment['xlabel'], fontsize=10)
    ax.set_ylabel(assignment['ylabel'], fontsize=10)
    ax.set_title(assignment['title'], fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    if assignment['show_legend'] and len(datasets) > 1:
        ax.legend(loc='best', fontsize=8, framealpha=0.9)

plt.tight_layout()

# Show figure
st.pyplot(fig)

# Export
st.divider()
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    buf = _save_fig_to_bytes(fig, dpi=300)
    st.download_button(
        label="💾 Download Figure (PNG, 300 DPI)",
        data=buf,
        file_name="multi_panel_figure.png",
        mime="image/png"
    )

with col2:
    buf_svg = _save_fig_to_bytes(fig, format='svg')
    st.download_button(
        label="💾 Download (SVG)",
        data=buf_svg,
        file_name="multi_panel_figure.svg",
        mime="image/svg+xml"
    )

with col3:
    st.metric("Axes", n_axes)

plt.close(fig)

# ---------------------------------------------------------------------------
# Tips
# ---------------------------------------------------------------------------

with st.expander("💡 Usage Tips"):
    st.markdown("""
    ### Creating Multi-Panel Figures

    **1. Load Data**
    - Upload or browse for multiple CSV/NPZ files
    - Each file can be plotted on one or more axes

    **2. Configure Layout**
    - Grid layout: Simple rows×columns arrangement
    - Custom positions: More flexible arrangements (future feature)

    **3. Assign Data to Axes**
    - Use tabs to configure each subplot
    - Select which datasets appear on each axis
    - Choose X/Y columns independently for each axis
    - Customize scales, labels, and legends

    **4. Export**
    - High-quality PNG (300 DPI) for presentations
    - SVG for publications and further editing

    ### Example Use Cases

    - **Compare techniques**: UV-Vis on top, SAXS on bottom
    - **Time series**: Multiple time points in grid
    - **Multi-component**: Different samples side-by-side
    - **Publication figures**: Create complex multi-panel layouts
    """)
