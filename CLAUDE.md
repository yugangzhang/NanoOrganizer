# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 📝 Latest Session Summary (2026-02-05)

### Completed Features
1. ✅ **Advanced file filters** with AND/OR/NOT logic
   - Contains ALL, Contains ANY, NOT contains
   - Comma-separated patterns in folder browser
   - File: `components/folder_browser.py`

2. ✅ **Multi-column Y selection** (UI done, plotting has bug)
   - Can select multiple Y columns per file
   - Each should create separate curve
   - Debug panel added to diagnose
   - File: `pages/1_📊_CSV_Plotter.py`

3. ✅ **Collapsible sections** in folder browser
   - "Current Path" and "Folders & Files" collapsible
   - File: `components/folder_browser.py`

4. ✅ **Plotly upgrade** to 6.1.1+ with Kaleido
   - File: `setup.py`

5. ✅ **Session state persistence**
   - Dataframes survive page reruns
   - File: `pages/1_📊_CSV_Plotter.py`

### Files Modified This Session
- `setup.py`
- `pages/1_📊_CSV_Plotter.py`
- `components/folder_browser.py`
- `ADVANCED_FILTERS_GUIDE.md` (new)

### Next Session - Priority Fixes
1. **Fix multi-column plotting** - only last column shows (not all)
   - Location: `pages/1_📊_CSV_Plotter.py`
   - Check: lines 304-353 (selection), 536+ (plotting loop)
   - Debug panel at line ~530 shows what's selected

2. **Fix toggle button** - shows but onClick doesn't work
   - Location: `pages/1_📊_CSV_Plotter.py` lines 143-185
   - JavaScript not working, need Streamlit alternative

3. **Apply to other tools** - Universal Plotter, Image Viewer, etc.

## Project Overview

NanoOrganizer is a modular framework for managing nanoparticle-synthesis experimental data. It uses a metadata-first approach with lazy loading, separating data references (links) from actual file I/O. The architecture is designed for extensibility: adding a new data type follows a standard 8-step pattern documented in `docs/adding_new_datatype.md`.

## Installation & Setup

```bash
# Editable install with all optional dependencies
pip install -e ".[web,image,dev]"

# Individual extras:
# [web]   - Streamlit, Plotly, Seaborn (multi-page web app)
# [image] - Pillow for SEM/TEM/2D detector images
# [dev]   - pytest and coverage tools

# Dependencies installed with [web]:
# - streamlit >= 1.20.0
# - plotly >= 5.0.0 (interactive 3D plots)
# - seaborn >= 0.11.0 (pretty 1D/2D plots)
# - kaleido (Plotly PNG export)
# - pandas >= 1.3.0
```

## Running the Web App

### **NEW: Single-Port Multi-Page App (Recommended)**

```bash
# ONE command for ALL tools on port 8501!
nanoorganizer

# Opens: http://localhost:8501 (or http://your.server.ip:8501)
```

**What you get:**
- Home page with welcome & documentation
- **8 tools** accessible via sidebar navigation:
  1. 📊 CSV Plotter - Enhanced with per-curve styling, NPZ support
  2. 🖼️ Image Viewer - 2D images, stacks, 3 view modes
  3. 📐 Multi-Axes - Publication multi-panel figures
  4. 📈 3D Plotter - **Interactive Plotly** (rotate with mouse!)
  5. 📊 Data Viewer - NanoOrganizer project explorer
  6. 🔧 Data Manager - Create projects, metadata forms
  7. 🧪 Test Data Generator - Generate comprehensive test data
  8. 🎯 Universal Plotter - **NEW!** Integrated 1D/2D/3D plotting with hover values

**All on port 8501!** No more managing multiple ports.

**Key Features:**
- ✅ **Interactive hover** - All Plotly plots show (x,y) or (x,y,z) values on cursor hover
- ✅ **Interactive folder browser** - Click through directories visually, no typing paths!
- ✅ **Mix plot types** - Combine 1D, 2D, and 3D in one figure (Universal Plotter)
- ✅ **Per-curve styling** - Full control over colors, markers, line styles
- ✅ **Multiple export formats** - HTML (interactive), PNG, SVG

### Legacy Individual Tools (Still Available)

```bash
nanoorganizer-viz      # Data viewer (port 8502)
nanoorganizer-csv      # CSV plotter (port 8504)
nanoorganizer-3d       # 3D plotter (port 8505)
# ... etc (see COMPLETE_WEB_SUITE.md for full list)
```

**Note**: Use the main `nanoorganizer` command for the best experience.

## Testing

### Test Data Generator (Built-In)

Generate comprehensive test data from the web GUI:
1. Launch `nanoorganizer`
2. Click "🧪 Test Data Generator" in sidebar
3. Configure settings (or use defaults)
4. Click "🚀 Generate All Test Data"
5. Creates `TestData/` folder with:
   - CSV time-series (100+ files)
   - NPZ arrays (10 files)
   - 2D detector images (10 images, 512×512)
   - Image stacks (5 stacks, 20 frames each)
   - 3D surface data (5 datasets)

### Manual Testing

- Example notebooks in `example/full_demo.ipynb`
- Demo scripts in `example/quick_reference.py` and `example/demo_nanoorganizer.py`
- Use generated TestData/ for all tools

## Architecture

### Three-Layer Design

1. **Core** (`NanoOrganizer/core/`): Metadata dataclasses, DataOrganizer (run manager), Run (single experiment), DataLink (file references), utilities
2. **Loaders** (`NanoOrganizer/loaders/`): One class per data type. Read files → standardized dict. Registered in `LOADER_REGISTRY`
3. **Plotters** (`NanoOrganizer/viz/`): One class per data type. Standardized dict → matplotlib/Plotly. Registered in `PLOTTER_REGISTRY`

### Web App Architecture (NEW)

**Multi-Page App Structure** (`NanoOrganizer/web_app/`):
```
web_app/
├── Home.py              # Main landing page (always visible)
├── app_cli.py           # Console entry point for `nanoorganizer`
├── components/          # Reusable UI components
│   ├── __init__.py
│   └── folder_browser.py  # Interactive folder navigation
└── pages/               # All tool pages (show in sidebar)
    ├── 1_📊_CSV_Plotter.py         # With folder browser!
    ├── 2_🖼️_Image_Viewer.py
    ├── 3_📐_Multi_Axes.py
    ├── 4_📈_3D_Plotter.py         # Interactive Plotly!
    ├── 5_📊_Data_Viewer.py
    ├── 6_🔧_Data_Manager.py
    ├── 7_🧪_Test_Data_Generator.py
    └── 8_🎯_Universal_Plotter.py  # With folder browser!
```

**Navigation**: Click page names in sidebar to switch between tools. All run on port 8501.

**Interactive Features**:
- **Folder browser**: Click folders to navigate, no typing paths - visual directory navigation!
- **Hover values**: All Plotly-based pages (4, 8) show (x,y) or (x,y,z) values when hovering
- **Mouse controls**: Rotate 3D plots, zoom, pan - all interactive
- **Export options**: Interactive HTML (preserves interactivity) or static PNG/SVG

**Folder Browser Component**:
- Quick shortcuts: 🏠 Home, 💼 CWD, 🧪 TestData, ⬆️ Parent
- Breadcrumb navigation - click any part of path to jump there
- Visual folder buttons - click to navigate into directories
- File checkboxes - select multiple files easily
- **Advanced Filters** (NEW!):
  - Extension filter: *.csv, *.npz, etc.
  - **Contains ALL**: filename must have ALL specified strings (AND logic)
  - **Contains ANY**: filename must have AT LEAST ONE string (OR logic)
  - **NOT contains**: filename must NOT have any of these strings (exclusion)
  - Filters are comma-separated: "sample1, sample2"
  - Real-time filtering as you type
- Used in: CSV Plotter, Universal Plotter (can be added to other pages)

**Legacy Tools** (`NanoOrganizer/web/`): Individual standalone apps (still work but deprecated).

### Key Concepts

**Metadata-first**: All run metadata (project, experiment, run_id, sample_id, reaction parameters, chemicals) is stored in dataclasses and serialized to JSON in `.metadata/` directory. Metadata is always in memory; file data is loaded lazily.

**DataLink vs Loading**: `DataLink` (in `core/data_links.py`) stores absolute file paths and metadata (like time_points, calibration parameters) but never reads files. Each loader has a `.link` attribute. Calling `loader.load()` reads files and returns a standardized dictionary.

**Registry pattern**: Both loaders and plotters use dict-based registries (`LOADER_REGISTRY`, `PLOTTER_REGISTRY`). Each run automatically gets all loaders attached via `DEFAULT_LOADERS` in `core/run.py`.

**Run keys**: Runs are identified by slash-joined strings: `"project/experiment/run_id"`. Use `org.get_run("Project_Au/2024-10-25/Au_Test_001")` to retrieve.

**Time-series convention**: All 1-D loaders return dicts with a `times` key (1D array of timestamps) plus domain-specific axes (wavelengths, q, two_theta, etc.) and measurements (2D array: n_times × n_points).

**2-D detector data**: SAXS2D and WAXS2D loaders accept `.npy` (preferred), `.png`, `.tif` files. Detector geometry calibration (pixel_size_mm, sdd_mm, wavelength_A) is stored in link metadata and used automatically during azimuthal averaging.

### File Structure

```
NanoOrganizer/
├── core/
│   ├── metadata.py      ChemicalSpec, ReactionParams, RunMetadata dataclasses
│   ├── data_links.py    DataLink – file-reference container
│   ├── organizer.py     DataOrganizer – top-level run manager, save/load
│   ├── run.py           Run – single experiment + DEFAULT_LOADERS registry
│   └── utils.py         save_time_series_to_csv helper
├── loaders/
│   ├── base.py          BaseLoader abstract class
│   ├── uvvis.py         UVVisLoader
│   ├── saxs.py, waxs.py, dls.py, xas.py
│   ├── saxs2d.py, waxs2d.py
│   ├── image.py         ImageLoader (SEM/TEM)
│   └── __init__.py      LOADER_REGISTRY
├── viz/
│   ├── base.py          BasePlotter abstract class
│   ├── uvvis.py         UVVisPlotter (spectrum, kinetics, heatmap)
│   ├── (parallel structure to loaders/)
│   └── __init__.py      PLOTTER_REGISTRY
├── simulations/
│   ├── uvvis.py         simulate_uvvis_time_series_data()
│   ├── (one per data type)
│   └── __init__.py
├── web_app/             ⭐ NEW: Multi-page app (port 8501)
│   ├── Home.py          Main landing page
│   ├── app_cli.py       Entry point
│   └── pages/           7 tool pages
├── web/                 Legacy individual tools (deprecated)
│   ├── app.py           Old data viewer
│   ├── csv_plotter*.py  Old CSV plotters
│   └── ...              Other legacy tools
└── __init__.py          Public API exports
```

## Adding a New Data Type

Follow the 8-step checklist in `docs/adding_new_datatype.md`:

1. Create `loaders/mytype.py` – subclass `BaseLoader`, implement `load()`
2. Register in `loaders/__init__.py` → `LOADER_REGISTRY`
3. Create `viz/mytype.py` – subclass `BasePlotter`, implement plot dispatch
4. Register in `viz/__init__.py` → `PLOTTER_REGISTRY`
5. Add to `DEFAULT_LOADERS` in `core/run.py` (gives every Run a `.mytype` attribute)
6. (Optional) Create `simulations/mytype.py` and register in `simulations/__init__.py`
7. Export in `NanoOrganizer/__init__.py` → `__all__`
8. Add dynamic selectors to `SELECTORS` dict in `web/app.py` for interactive controls

## Adding a New Web Tool Page

To add a new page to the multi-page app:

1. Create `NanoOrganizer/web_app/pages/N_📊_Tool_Name.py` (N = number for ordering)
2. **Do NOT** include `st.set_page_config()` (handled by main app)
3. Add your tool code with `st.title()` and components
4. Restart `nanoorganizer` - new page appears in sidebar automatically!

Example:
```python
# web_app/pages/8_🔬_My_Tool.py
import streamlit as st

st.title("🔬 My Custom Tool")
# Your code here...
```

## Code Patterns

**Creating and saving data**:
```python
from NanoOrganizer import DataOrganizer, RunMetadata, ReactionParams, ChemicalSpec

org = DataOrganizer("./MyProject")  # creates .metadata/ directory
meta = RunMetadata(
    project="Project_Au",
    experiment="2024-10-25",
    run_id="Au_Test_001",
    sample_id="Sample_001",
    reaction=ReactionParams(
        chemicals=[ChemicalSpec(name="HAuCl4", concentration=0.5)],
        temperature_C=80.0,
    ),
)
run = org.create_run(meta)
run.uvvis.link_data(csv_files, time_points=[0, 30, 60, 120])
org.save()  # writes JSON to .metadata/
```

**Loading and plotting**:
```python
org = DataOrganizer.load("./MyProject")
run = org.get_run("Project_Au/2024-10-25/Au_Test_001")
data = run.uvvis.load()  # dict: {times, wavelengths, absorbance}
run.uvvis.plot(plot_type="heatmap")
```

**1-D time-series CSV format**: Each time point = one CSV file with two columns (header names from loader spec, e.g., "wavelength,absorbance"). Use `save_time_series_to_csv()` utility to write from long-format (times, x, y) lists returned by simulators.

**2-D detector linking**:
```python
run.saxs2d.link_data(
    npy_files,
    time_points=[0, 30, 60, 120],
    pixel_size_mm=0.172,
    sdd_mm=3000.0,
    wavelength_A=1.0,
)
```

**Validation**: Call `org.validate_all()` after linking to check all referenced files exist.

## Web App Features

### Interactive 3D Plots (Plotly)

The 3D Plotter uses Plotly for fully interactive visualizations:
- **Rotate**: Left-click drag
- **Zoom**: Scroll wheel
- **Pan**: Right-click drag
- **Export**: Interactive HTML (keeps interactivity!), PNG, SVG

Plot types: Surface, Scatter 3D, Contour 3D, Wireframe, Mesh

### Enhanced CSV Plotter

Features:
- **Per-curve styling**: Individual color, marker, line style, width, opacity
- **NPZ support**: Load NumPy compressed arrays
- **Smart path display**: Long paths auto-shortened
- **15 colors × 12 markers × 4 line styles**

### Test Data Generator

Built-in GUI tool to generate:
- CSV time-series (UV-Vis-like, 100+ files)
- NPZ arrays (multi-column)
- 2D detector images (512×512, SAXS-like)
- Image stacks (3D arrays, 20 frames)
- 3D surface data (Gaussian, ripple, saddle, volcano, waves)

### Universal Plotter (NEW!)

**Integrated plotting system** for mixing 1D, 2D, and 3D plots in one figure:

**Features**:
- ✅ **Show cursor values on hover** - All plots display (x,y) or (x,y,z) coordinates when you hover
- ✅ **Flexible layouts** - Create 1×1, 2×2, 1×3, or custom grid arrangements
- ✅ **Mix plot types** - Combine 1D line, 2D heatmap, and 3D surface in same figure
- ✅ **Independent configuration** - Each subplot has its own data source and styling
- ✅ **Fully interactive** - Zoom, pan, rotate 3D plots with mouse
- ✅ **Export options** - Interactive HTML (preserves hover!), PNG, SVG

**Plot types**:
- **1D Line**: Time-series, spectra, any (x,y) data - hover shows exact values
- **2D Heatmap**: Detector images, intensity maps - hover shows (x,y,z)
- **3D Surface**: Volumetric data, surfaces - fully rotatable with hover values

**Use case**: Create publication-ready figures with different plot types side-by-side, all with interactive hover tooltips showing exact data values.

## Important Notes

- **Single port**: Use `nanoorganizer` command - all 8 tools on port 8501
- **Absolute paths**: All file paths are converted to absolute and stored that way. No rigid directory structure is enforced.
- **Lazy loading**: Metadata is always in memory. File data only loads when `.load()` is called.
- **JSON serialization**: Runs save to `.metadata/<project>_<experiment>_<run_id>.json`. The organizer maintains an index in `.metadata/index.json`.
- **Backward compatibility**: Old loader names (UVVisData, SAXSData, etc.) are aliased to new loader classes in `__init__.py`.
- **Interactive plots**: 3D Plotter and Universal Plotter use Plotly for mouse-controllable plots with hover values
- **Test data**: Use built-in Test Data Generator tool before testing other tools

## Documentation Files

- `CLAUDE.md` - This file (updated 2026-02-05, added Universal Plotter)
- `LATEST_IMPROVEMENTS_SUMMARY.md` - Recent updates summary
- `SINGLE_PORT_APP_GUIDE.md` - Complete guide to multi-page app
- `COMPLETE_WEB_SUITE.md` - Full overview of web tools
- `QUICK_REFERENCE_NEW.md` - Quick reference card
- `docs/WEB_GUI_GUIDE.md` - Detailed user manual
- `docs/adding_new_datatype.md` - How to extend the system

## Console Commands

**Main command** (recommended):
```bash
nanoorganizer          # All 8 tools on port 8501
```

**Legacy commands** (still available):
```bash
nanoorganizer-viz      # Individual tool on port 8502
nanoorganizer-csv      # Individual tool on port 8504
# ... etc (see setup.py for full list)
```

## Quick Troubleshooting

**"Command not found: nanoorganizer"**
```bash
pip install -e ".[web,image]"
```

**"Only 2 pages showing in sidebar"** or **"IndentationError in pages"**
- Check `NanoOrganizer/web_app/pages/` has all 8 .py files (1-8)
- Verify syntax: `python3 -m py_compile NanoOrganizer/web_app/pages/*.py`
- Restart the app: `pkill -f streamlit && nanoorganizer`

**"Port 8501 already in use"**
```bash
pkill -f streamlit     # Kill all Streamlit instances
nanoorganizer          # Restart
```

**"Plotly plots not rotating"**
- Make sure you're using the 3D Plotter page (📈 3D Plotter)
- Try different browser (Chrome/Firefox work best)

**"Can't export Plotly to PNG"**
```bash
pip install kaleido
```

Last updated: 2026-02-05 - Session with multi-column support, advanced filters, collapsible UI

## 🚧 Known Issues (To Fix Next Session)
1. **Multi-column plotting**: Only last selected Y column shows in plot (not all selected columns)
   - Debug panel added to diagnose
   - Issue in plotting loop or session state
2. **Bottom toggle button**: Shows but onClick doesn't work
   - Button visible but JavaScript not triggering sidebar toggle
   - Need alternative approach (maybe Streamlit components)
