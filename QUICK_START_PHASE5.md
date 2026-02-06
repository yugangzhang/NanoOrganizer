# Quick Start Guide - Phase 5 Improvements

Get started with all the new Phase 5 features in 5 minutes!

---

## 🚀 Installation (One-Time Setup)

```bash
cd /home/yuzhang/Repos/NanoOrganizer

# Remove old build files
sudo rm -rf Nanoorganizer.egg-info build dist

# Install with all features
pip install -e ".[web,image]"

# Open ports for remote access (if needed)
sudo firewall-cmd --permanent --add-port=8501-8507/tcp
sudo firewall-cmd --reload
```

---

## 🎯 Launch the Hub

The hub is your central control panel for all tools:

```bash
nanoorganizer-hub
```

Then open in your browser:
- **On server**: http://localhost:8501
- **From Windows**: http://130.199.242.142:8501

You'll see a dashboard with 6 tool cards. Each card shows:
- Tool description
- Key features
- Launch button with manual instructions

---

## 🧪 Quick Test: Enhanced CSV Plotter

Let's test the new per-curve styling feature!

### Step 1: Launch

From a new terminal:
```bash
streamlit run NanoOrganizer/web/csv_plotter_enhanced.py --server.port 8504
```

Or from the hub: Click "Launch CSV Plotter" button for instructions

### Step 2: Load Demo Data

```bash
# Navigate to Demo data
cd /home/yuzhang/Repos/NanoOrganizer
```

In the CSV plotter GUI:
1. Select "Browse server"
2. Directory: `/home/yuzhang/Repos/NanoOrganizer/Demo/Project_Cu2O/UV_Vis/2024-10-25/Cu2O_Growth_Study_001`
3. Pattern: `*.csv`
4. Click "🔍 Search"
5. Select 3-5 files

### Step 3: Customize Each Curve

You'll see an expander for each file. Open them and:

**File 1:**
- Color: Blue
- Marker: Circle
- Line Style: Solid
- Line Width: 2.0
- Opacity: 0.8

**File 2:**
- Color: Red
- Marker: Square
- Line Style: Dashed
- Line Width: 2.5
- Opacity: 0.7

**File 3:**
- Color: Green
- Marker: Triangle Up
- Line Style: Dotted
- Line Width: 3.0
- Opacity: 0.9

**File 4:**
- Color: Purple
- Marker: Diamond
- Line Style: Dash-dot
- Line Width: 2.0
- Opacity: 0.6

### Step 4: View and Export

Scroll down to see your beautiful multi-curve plot!
Click "💾 Download Plot (PNG, 300 DPI)" to save.

---

## 🖼️ Quick Test: Image Viewer

### Step 1: Launch

```bash
streamlit run NanoOrganizer/web/image_viewer.py --server.port 8506
```

### Step 2: Generate Test Images

Quick Python script to create test images:

```python
import numpy as np

# Create test image stack
stack = np.zeros((10, 512, 512))
for i in range(10):
    # Create Gaussian peak that moves
    x = np.linspace(-5, 5, 512)
    y = np.linspace(-5, 5, 512)
    X, Y = np.meshgrid(x, y)

    # Moving peak
    cx = -3 + i * 0.6
    cy = -2 + i * 0.4
    stack[i] = np.exp(-((X-cx)**2 + (Y-cy)**2) / 2)

# Save
np.save('/tmp/test_stack.npy', stack)
print("✅ Saved /tmp/test_stack.npy")
```

### Step 3: Load and View

In the Image Viewer:
1. Select "Browse server"
2. Directory: `/tmp`
3. Pattern: `test_stack.npy`
4. Click "🔍 Search"
5. Select the file

### Step 4: Explore

1. View mode: "Single image"
2. Use Frame slider (0-9) to see the peak move
3. Change colormap to "hot"
4. Toggle auto-contrast off
5. Adjust min/max percentiles
6. Export frame

---

## 📐 Quick Test: Multi-Axes Plotter

### Step 1: Launch

```bash
streamlit run NanoOrganizer/web/multi_axes_plotter.py --server.port 8507
```

### Step 2: Load Multiple Datasets

1. Select "Browse server"
2. Directory: `/home/yuzhang/Repos/NanoOrganizer/Demo/Project_Cu2O`
3. Pattern: `**/uvvis_*.csv` (finds all UV-Vis files)
4. Click "🔍 Search"
5. Select 4 files

### Step 3: Configure Layout

1. Layout type: "Grid (rows × cols)"
2. Rows: 2
3. Columns: 2
4. This creates 4 axes: (1,1), (1,2), (2,1), (2,2)

### Step 4: Assign Data to Each Axis

Click through the 4 tabs:

**Tab 1: (1,1)**
- Select: File 1
- X column: wavelength
- Y column: absorbance
- Title: "Early time point"

**Tab 2: (1,2)**
- Select: File 2
- X column: wavelength
- Y column: absorbance
- Title: "Mid time point"

**Tab 3: (2,1)**
- Select: File 3, File 4 (both!)
- X column: wavelength
- Y column: absorbance
- Y scale: log
- Title: "Comparison (log scale)"

**Tab 4: (2,2)**
- Leave empty (will show "No data")

### Step 5: Export

Scroll down to see your 2×2 figure
Download as PNG or SVG

---

## 🎨 Feature Highlights

### Per-Curve Styling (Enhanced CSV Plotter)

**Why it's awesome:**
- Each curve gets individual controls
- 15 colors × 12 markers × 4 line styles = 720 combinations!
- Perfect for creating publication figures
- No need to edit in post-processing

**When to use:**
- Comparing many datasets
- Need specific color scheme
- Publication figures
- Presentations

### Image Stack Browsing (Image Viewer)

**Why it's awesome:**
- Navigate through time-series detector images
- Frame-by-frame analysis
- Side-by-side comparison of different experiments
- Auto-contrast or manual intensity control

**When to use:**
- SAXS/WAXS 2D detector data
- Microscopy time series
- Analyzing detector images
- Finding specific frames

### Multi-Panel Figures (Multi-Axes Plotter)

**Why it's awesome:**
- Combine different data types in one figure
- Independent scaling per panel
- Perfect for publications
- Export as vector graphics (SVG)

**When to use:**
- Publication figures
- Comparing techniques (UV-Vis + SAXS + WAXS)
- Multi-component analysis
- Before/after comparisons

---

## 💡 Pro Tips

### Tip 1: Run Multiple Tools Simultaneously

Each tool runs on its own port, so you can have them all open:

```bash
# Terminal 1
streamlit run NanoOrganizer/web/csv_plotter_enhanced.py --server.port 8504 &

# Terminal 2
streamlit run NanoOrganizer/web/image_viewer.py --server.port 8506 &

# Terminal 3
streamlit run NanoOrganizer/web/multi_axes_plotter.py --server.port 8507 &
```

Access each in different browser tabs!

### Tip 2: Smart Filename Display

When selecting files from deep directory paths, the enhanced CSV plotter shows:
- Short paths: Full name
- Long paths: `.../filename.csv`
- Hover for full path

### Tip 3: NPZ File Support

Save data as NPZ for faster loading:

```python
import numpy as np

# Save multiple arrays
np.savez('data.npz',
         wavelength=wavelengths,
         absorbance=absorbance,
         time=times)
```

Load in CSV plotter - it auto-extracts all arrays!

### Tip 4: Session Persistence

Enhanced CSV plotter remembers your styling choices:
- Close and reopen - settings preserved
- Reload browser - settings preserved
- Only resets when you restart Streamlit

---

## 🐛 Troubleshooting

### "Command not found: nanoorganizer-hub"

**Solution:** Run installation again:
```bash
pip install -e ".[web,image]"
```

### "Port already in use"

**Solution:** Either:
1. Kill existing Streamlit: `pkill -f streamlit`
2. Or use different port: `--server.port 8510`

### "Can't access from Windows"

**Solution:** Check firewall:
```bash
sudo firewall-cmd --list-ports  # Should show 8501-8507/tcp
```

### "Module not found: PIL"

**Solution:** Install image support:
```bash
pip install Pillow
```

---

## 📊 Summary

**Phase 5 gave you:**
- ✅ Central hub for easy access
- ✅ Full per-curve customization
- ✅ NPZ file support
- ✅ Dedicated image viewer
- ✅ Multi-panel figure creator
- ✅ Smart path display
- ✅ 3 new tools, 1 enhanced tool

**Total tools now: 7**
**Total commands: 8 (including hub)**

---

## 🎯 Next Steps

1. **Test all features** using guides above
2. **Try with your own data**
3. **Create publication figures**
4. **Share with colleagues**

---

## 📚 Full Documentation

- `PHASE5_IMPROVEMENTS.md` - Complete Phase 5 details
- `docs/WEB_GUI_GUIDE.md` - User manual for all tools
- `INSTALLATION_NEXT_STEPS.md` - Detailed installation
- `PHASE_IMPLEMENTATION_SUMMARY.md` - Phases 1-4

---

Enjoy your enhanced NanoOrganizer! 🚀
