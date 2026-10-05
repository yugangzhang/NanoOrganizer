# Quick Reference - New Single-Port App

## 🚀 Getting Started (3 Commands)

```bash
# 1. Install
cd /home/yuzhang/Repos/NanoOrganizer && \
sudo rm -rf Nanoorganizer.egg-info build dist && \
pip install -e ".[web,image]"

# 2. Launch (ONE command for everything!)
nanoorganizer

# 3. Open browser
# http://130.199.242.142:8501
```

Done! All 7 tools running on port 8501.

---

## 🎯 What Changed

| Old Way | New Way |
|---------|---------|
| 8 commands | 1 command: `nanoorganizer` ✅ |
| 8 ports (8501-8508) | 1 port: 8501 ✅ |
| Manual URLs | Sidebar navigation ✅ |
| Matplotlib 3D | **Plotly 3D** (interactive!) ✅ |
| No test data | **Built-in generator** ✅ |

---

## 📱 Navigation

**After launching `nanoorganizer`:**

1. **Sidebar** (left) - Click to switch tools:
   - 🏠 Home
   - 📊 CSV Plotter
   - 🖼️ Image Viewer
   - 📐 Multi-Axes
   - 📈 3D Plotter (NEW - Plotly!)
   - 📊 Data Viewer
   - 🔧 Data Manager
   - 🧪 Test Data Generator (NEW!)

2. **Main area** - Currently selected tool

3. **Settings** (⋮ top right) - App settings

---

## 🧪 First Thing: Generate Test Data

```
1. Sidebar → Click "🧪 Test Data Generator"
2. Keep defaults
3. Click "🚀 Generate All Test Data"
4. Wait 30 seconds
5. ✅ 150+ test files created!
```

**Output**: `~/Repos/NanoOrganizer/TestData/`
- csv_data/ - 100+ CSV files
- npz_data/ - 10 NPZ files
- images_2d/ - 10 images
- image_stacks/ - 5 stacks (20 frames each)
- data_3d/ - 5 3D datasets

---

## 🎨 Try Interactive 3D (NEW!)

```
1. Sidebar → "📈 3D Plotter"
2. Select "Generate synthetic"
3. Function: "Gaussian"
4. Click "🎲 Generate"
5. 🖱️ DRAG MOUSE TO ROTATE!
6. Scroll to zoom
7. Right-click drag to pan
8. Download as interactive HTML
```

**Why it's awesome:**
- Fully rotatable with mouse
- Zoom in/out smoothly
- Export keeps interactivity
- Much better than matplotlib!

---

## 📊 Tool Reference

### Home (Landing Page)
- Overview of all tools
- Quick start guide
- Documentation links

### CSV Plotter
- Load multiple CSV/NPZ files
- Per-curve color, marker, line style
- Seaborn styling (coming)
- Export PNG/SVG

### Image Viewer
- Load 2D images (NPY, PNG, TIFF)
- Browse image stacks frame-by-frame
- 3 view modes (single, comparison, grid)
- 15 colormaps

### Multi-Axes
- Create multi-panel figures
- Assign data to each subplot
- Independent scales per panel
- Publication-ready exports

### 3D Plotter ⭐ NEW
- **Interactive Plotly plots**
- Rotate/zoom/pan with mouse
- 5 plot types (surface, scatter, etc.)
- Export interactive HTML

### Data Viewer
- Explore NanoOrganizer projects
- Multi-run comparison
- Log/linear scales
- Export plots

### Data Manager
- Create NanoOrganizer projects
- Fill metadata forms
- Link data files
- Browse server filesystem

### Test Data Generator ⭐ NEW
- **Generate test data with 1 click**
- 5 data types
- Configurable parameters
- Instant testing

---

## 💾 Export Formats

### Static Images
- **PNG** - High-res (300 DPI), presentations
- **SVG** - Vector, publications (edit in Illustrator)

### Interactive
- **HTML** (Plotly only) - Share rotatable 3D plots!

### How to Export
1. Create your plot
2. Scroll down to export buttons
3. Click "💾 Download..."
4. Save file

---

## 🔥 Hot Keys

- `r` - Rerun app
- `c` - Clear cache
- `Ctrl+K` - Command palette
- `Ctrl+S` - Settings

---

## 🐛 Troubleshooting

### "Command not found: nanoorganizer"
```bash
pip install -e ".[web,image]"
```

### "Port 8501 already in use"
```bash
# Kill existing Streamlit
pkill -f streamlit

# Then launch again
nanoorganizer
```

### "Can't access from Windows"
```bash
# Check firewall
sudo firewall-cmd --list-ports  # Should show 8501/tcp

# If not, add it
sudo firewall-cmd --permanent --add-port=8501/tcp
sudo firewall-cmd --reload
```

### "Plotly plots not exporting to PNG"
```bash
pip install kaleido
```

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| `LATEST_IMPROVEMENTS_SUMMARY.md` | What's new |
| `SINGLE_PORT_APP_GUIDE.md` | Complete guide |
| `COMPLETE_WEB_SUITE.md` | Full overview |
| `QUICK_START_PHASE5.md` | Testing guide |

---

## 🎯 Common Tasks

### Task: Compare multiple CSV files
```
1. Test Data Generator → Generate CSV data
2. CSV Plotter → Browse to TestData/csv_data/
3. Select 5-10 files
4. Customize colors/markers
5. Export
```

### Task: View detector images
```
1. Test Data Generator → Generate images
2. Image Viewer → Browse to TestData/images_2d/
3. Select images
4. Choose colormap
5. Export
```

### Task: Create 3D surface plot
```
1. 3D Plotter → Browse to TestData/data_3d/
2. Load gaussian_3d.csv
3. Select "Surface" type
4. Rotate with mouse!
5. Download as HTML (interactive!)
```

### Task: Multi-panel figure
```
1. Multi-Axes → Load multiple CSVs
2. Set layout (2×2 grid)
3. Assign data to each panel
4. Customize labels
5. Export SVG
```

---

## 💡 Pro Tips

### Tip 1: Browser Tabs
Open multiple tabs to same app:
- Tab 1: CSV Plotter
- Tab 2: 3D Plotter
- Tab 3: Image Viewer

All sharing same port!

### Tip 2: Interactive HTML Export
3D plots exported as HTML:
- Send to colleagues
- They can rotate/zoom too
- No NanoOrganizer install needed
- Works in any browser

### Tip 3: Test Data Once
Generate test data once, use everywhere:
- CSV Plotter
- Image Viewer
- 3D Plotter
- Multi-Axes
- All tools share same TestData/

### Tip 4: State Persistence
Switch between tools freely:
- Your settings are saved per page
- Come back, still there
- Only resets on app restart

---

## 🎉 Summary

**One Command:**
```bash
nanoorganizer  # Everything on port 8501!
```

**Seven Tools:**
1. CSV Plotter (per-curve styling)
2. Image Viewer (stacks, 3 modes)
3. Multi-Axes (publication figures)
4. 3D Plotter ⭐ (interactive Plotly!)
5. Data Viewer (NanoOrganizer projects)
6. Data Manager (create projects)
7. Test Data ⭐ (generate with 1 click!)

**Infinite Possibilities:**
- Compare datasets
- Create publication figures
- Generate test data
- Interactive 3D plots
- All from one browser tab!

---

Enjoy! 🚀
