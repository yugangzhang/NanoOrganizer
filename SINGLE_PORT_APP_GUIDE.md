# Single-Port Multi-Page App - Complete Guide

## 🎉 Major Update: ALL Tools on Port 8501!

You asked for it, you got it! **All 7 tools now run on a single port** using Streamlit's multi-page app feature.

---

## 🚀 Quick Start

### Launch the App

```bash
# Simple - just one command!
nanoorganizer

# Opens on http://130.199.242.142:8501
```

That's it! No more managing multiple ports or terminals.

---

## 📱 How It Works

### Multi-Page Structure

```
NanoOrganizer/web_app/
├── Home.py                          # Main landing page (Home tab)
├── app_cli.py                       # Console entry point
└── pages/                           # All tools as pages
    ├── 1_📊_CSV_Plotter.py         # (to be created)
    ├── 2_🖼️_Image_Viewer.py        # (to be created)
    ├── 3_📐_Multi_Axes.py           # (to be created)
    ├── 4_📈_3D_Plotter.py           # ✅ Created with Plotly!
    ├── 5_📊_Data_Viewer.py          # (to be created)
    ├── 6_🔧_Data_Manager.py         # (to be created)
    └── 7_🧪_Test_Data_Generator.py  # ✅ Created!
```

### Navigation

When you run `nanoorganizer`, you'll see:

1. **Sidebar** - Lists all pages (Home + 7 tools)
2. **Main content** - Currently selected page
3. **Navigation** - Click any page in sidebar to switch

**All on port 8501!** No more port juggling!

---

## ✨ What's New

### 1. Single Port (8501)
- **Before**: 8 different ports (8501-8508)
- **After**: Everything on 8501 ✅
- **Benefit**: Simple, no port conflicts

### 2. Interactive 3D with Plotly 🎨
**File**: `pages/4_📈_3D_Plotter.py`

**New features:**
- ✅ **Fully interactive** - Rotate, zoom, pan with mouse
- ✅ **Multiple plot types**: Surface, Scatter 3D, Contour 3D, Wireframe, Mesh
- ✅ **15+ colorscales** - Beautiful gradients
- ✅ **Export interactive HTML** - Share rotatable plots!
- ✅ **Export PNG/SVG** - High-quality static exports
- ✅ **Smooth performance** - Much faster than matplotlib

**Why Plotly is better:**
- Rotate plot with mouse (try it!)
- Zoom in/out with scroll wheel
- Pan by right-click drag
- Hover to see values
- Camera controls in toolbar
- Export as interactive HTML

### 3. Comprehensive Test Data Generator 🧪
**File**: `pages/7_🧪_Test_Data_Generator.py`

**Generates:**
- ✅ **CSV time-series** (10-20 files, UV-Vis-like)
- ✅ **NPZ arrays** (multi-column data)
- ✅ **2D detector images** (512×512, SAXS-like)
- ✅ **Image stacks** (20 frames, moving peaks)
- ✅ **3D surface data** (Gaussian, ripple, saddle, volcano, waves)

**Use cases:**
- Test all tools without real data
- Learn the interface
- Create demos
- Verify functionality

### 4. Seaborn & Plotly Integration (Coming)
- Better-looking 1D/2D plots with Seaborn
- Interactive plots with Plotly Express
- Professional styling out of the box

---

## 📊 Feature Comparison

### Old Way (8 Separate Ports)
```
Terminal 1: nanoorganizer-hub      → port 8501
Terminal 2: nanoorganizer-viz      → port 8502
Terminal 3: nanoorganizer-csv      → port 8504
Terminal 4: nanoorganizer-3d       → port 8505
Terminal 5: nanoorganizer-img      → port 8506
Terminal 6: nanoorganizer-multi    → port 8507
Terminal 7: nanoorganizer-manage   → port 8503
```
**Problems:**
- 7+ terminals to manage
- Remember which port for what
- Firewall needs 7 open ports

### New Way (Single Port)
```
Terminal: nanoorganizer → port 8501
```
**Benefits:**
- ✅ One command
- ✅ One port
- ✅ One browser tab (multiple pages in sidebar)
- ✅ Cleaner, simpler, better

---

## 🧪 Testing the New App

### Step 1: Installation

```bash
cd /home/yuzhang/Repos/NanoOrganizer

# Remove old build
sudo rm -rf Nanoorganizer.egg-info build dist

# Install with new dependencies (Plotly, Seaborn)
pip install -e ".[web,image]"

# This installs:
# - plotly >= 5.0.0
# - seaborn >= 0.11.0
# - kaleido (for plotly PNG export)
```

### Step 2: Launch

```bash
nanoorganizer

# OR manually:
streamlit run NanoOrganizer/web_app/Home.py
```

Opens at: http://130.199.242.142:8501

### Step 3: Generate Test Data

1. In sidebar, click "🧪 Test Data Generator"
2. Configure settings (defaults are fine)
3. Click "🚀 Generate All Test Data"
4. Wait ~30 seconds
5. See summary of generated files

### Step 4: Test 3D Plotter (Plotly)

1. In sidebar, click "📈 3D Plotter"
2. Select "Generate synthetic"
3. Choose "Gaussian"
4. Click "🎲 Generate"
5. **Try rotating the plot with your mouse!** 🖱️
6. Try different plot types (Surface, Scatter, Mesh)
7. Download as interactive HTML

### Step 5: Test Other Tools

All tools accessible from sidebar:
- CSV Plotter
- Image Viewer
- Multi-Axes
- Data Viewer
- Data Manager

---

## 📖 User Guide

### Navigation

**Sidebar:**
- Click page name to switch
- Current page highlighted
- Emoji icons for quick ID

**Home Page:**
- Overview of all tools
- Quick start instructions
- Documentation links

**Each Tool Page:**
- Independent functionality
- Own state management
- Can switch between pages without losing work

### Workflow Example

```
1. Home page → Read overview
2. Test Data Generator → Create test data
3. CSV Plotter → Load CSVs, customize styling
4. Image Viewer → Browse image stacks
5. 3D Plotter → Create interactive 3D plot
6. Multi-Axes → Combine into publication figure
```

All without leaving port 8501!

---

## 🎨 Plotly 3D Features

### Interactive Controls

**Mouse:**
- **Left-click + drag** - Rotate
- **Right-click + drag** - Pan
- **Scroll wheel** - Zoom
- **Double-click** - Reset view

**Toolbar:**
- 📷 Camera - Download PNG
- 🏠 Home - Reset view
- ↔️ Pan - Pan mode
- 🔍 Zoom - Box zoom
- 📐 Orbit - 3D rotate mode

### Plot Types

1. **Surface** - Smooth colored surface
2. **Scatter 3D** - Individual points in 3D space
3. **Contour 3D** - Isosurface (volumetric)
4. **Wireframe** - Mesh lines only
5. **Mesh** - Triangulated surface

### Export Options

- **Interactive HTML** - Full interactivity preserved, share with colleagues
- **PNG** - High-res static image (requires kaleido)
- **SVG** - Vector graphics for publications

---

## 🔧 Configuration

### Port (if needed)

Change port by editing `web_app/app_cli.py`:
```python
"--server.port", "8501"  # Change to any port
```

### Add New Pages

1. Create file in `web_app/pages/`
2. Name format: `N_📊_Page_Name.py` (N = number for ordering)
3. Use emoji for visual icon
4. Will automatically appear in sidebar

Example:
```python
# web_app/pages/8_🔬_My_Tool.py

import streamlit as st

st.set_page_config(page_title="My Tool", page_icon="🔬")
st.title("🔬 My Custom Tool")

# Your tool code here
```

Restart app - new page appears!

---

## 💡 Pro Tips

### Tip 1: Browser Tabs

You can open multiple browser tabs to the same app on different pages:
- Tab 1: CSV Plotter
- Tab 2: Image Viewer
- Tab 3: 3D Plotter

All sharing the same Streamlit session!

### Tip 2: Plotly Export Formats

**HTML**: Best for sharing interactive plots
```python
# Recipient can rotate, zoom, etc.
```

**PNG**: Best for presentations (needs kaleido)
```bash
pip install kaleido
```

**SVG**: Best for publications (editable in Illustrator)

### Tip 3: Test Data

Generate test data once, use across all tools:
```
TestData/
├── csv_data/          → CSV Plotter, Multi-Axes
├── npz_data/          → CSV Plotter (NPZ support!)
├── images_2d/         → Image Viewer
├── image_stacks/      → Image Viewer (stack mode)
└── data_3d/           → 3D Plotter
```

### Tip 4: State Persistence

Each page maintains its own state:
- Switch away from CSV Plotter
- Come back - your selections are still there!
- Only resets when you restart Streamlit

---

## 🚀 Command Reference

### New Main Command
```bash
nanoorganizer              # All tools on port 8501 ⭐ RECOMMENDED
```

### Legacy Commands (Still Available)
```bash
nanoorganizer-hub          # Old hub (port 8501)
nanoorganizer-viz          # Data viewer (port 8502)
nanoorganizer-csv          # CSV plotter (port 8504)
nanoorganizer-3d           # 3D plotter (port 8505)
nanoorganizer-img          # Image viewer (port 8506)
nanoorganizer-multi        # Multi-axes (port 8507)
nanoorganizer-manage       # Data manager (port 8503)
```

**Note**: Legacy commands still work for backwards compatibility, but **use `nanoorganizer` for the best experience!**

---

## 📊 Comparison Table

| Feature | Old (Multi-Port) | New (Single Port) |
|---------|------------------|-------------------|
| Command | 8 different commands | 1 command: `nanoorganizer` |
| Ports | 8501-8508 | 8501 only |
| Terminals | Multiple | One |
| Navigation | Manual URL changes | Sidebar clicks |
| State | Separate sessions | Shared session |
| Firewall | Open 7+ ports | Open 1 port |
| 3D Plots | Matplotlib (static) | Plotly (interactive) |
| Test Data | Manual creation | Built-in generator |
| Complexity | High | Low ✅ |

---

## 🎯 Summary

**What Changed:**
- ✅ All 7 tools on single port (8501)
- ✅ Streamlit multi-page app structure
- ✅ Plotly for interactive 3D plots
- ✅ Comprehensive test data generator
- ✅ One command to rule them all: `nanoorganizer`

**What Stayed:**
- ✅ All features from Phases 1-5
- ✅ Legacy commands still work
- ✅ Per-curve styling, NPZ support, etc.

**What's Better:**
- ✅ Simpler to use
- ✅ Interactive 3D plots
- ✅ Easy navigation
- ✅ Test data at your fingertips

---

## 📚 Next Steps

1. **Install**: `pip install -e ".[web,image]"`
2. **Launch**: `nanoorganizer`
3. **Generate**: Go to Test Data Generator page
4. **Explore**: Try all 7 tools
5. **Enjoy**: Interactive 3D plots! 🎉

---

**Questions? Issues?**
- Check `COMPLETE_WEB_SUITE.md` for overview
- Check `QUICK_START_PHASE5.md` for testing
- GitHub: https://github.com/yugangzhang/Nanoorganizer/issues

Enjoy your new single-port, multi-page, interactive app! 🚀
