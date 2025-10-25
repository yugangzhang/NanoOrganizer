# Time-Series Features for NanoOrganizer

## 🎯 What's New: Realistic Time-Resolved Data

Our enhanced NanoOrganizer now supports **real time-series data** - perfect for monitoring nanoparticle synthesis in droplet reactors! This matches how Our experiments actually work: taking measurements at multiple time points during the reaction.

## 📊 What Changed

### Before (Original)
```python
# Single measurement per wavelength
times = [0, 1, 2, ..., 300]         # Different time for each point
wavelengths = [200, 201, 202, ...]  # One wavelength per point
absorbance = [0.1, 0.12, 0.13, ...]

# Problem: Not realistic - We don't measure one wavelength at a time!
```

### After (Time-Series)
```python
# Multiple full spectra over time
time_points = [0, 60, 120, 180, 300, 600]  # 6 measurements

# At each time point, measure full spectrum (300 wavelengths)
# Total data: 6 times × 300 wavelengths = 1800 rows

# CSV format:
# time_s, wavelength_nm, absorbance
# 0, 200, 0.10
# 0, 201, 0.11
# 0, 202, 0.12
# ...
# 60, 200, 0.15
# 60, 201, 0.16
# ...
```

## 🔬 New Simulation Functions

### 1. `simulate_uvvis_time_series_data()`
Simulates realistic UV-Vis evolution during nanoparticle growth:
- **Peak shifts**: Plasmon resonance moves as particle size increases
- **Intensity grows**: More particles form over time
- **Peak sharpens**: Better monodispersity with controlled growth

```python
from time_series_simulations import simulate_uvvis_time_series_data

times, wavelengths, absorbance = simulate_uvvis_time_series_data(
    time_points=[0, 60, 120, 180, 300, 600],  # Measurement times
    initial_peak=480,    # nm - small particles
    final_peak=530,      # nm - grown particles
    growth_rate=1.0      # Speed of growth
)

# Use exactly as before
run.uvvis().add(times, wavelengths, absorbance)
```

**What it models:**
- Small particles → Short wavelength plasmon
- Growth → Red-shift of peak
- Saturation → Peak stabilizes

### 2. `simulate_saxs_time_series_data()`
Simulates SAXS evolution showing particle size increase:
- **Form factor shift**: Features move to lower q as particles grow
- **Intensity increase**: More scattering from larger particles
- **Realistic noise**: Poisson statistics from X-ray counting

```python
from time_series_simulations import simulate_saxs_time_series_data

times, q_values, intensities = simulate_saxs_time_series_data(
    time_points=[0, 60, 120, 180, 300, 600],
    initial_size_nm=2.0,   # Starting size
    final_size_nm=10.0,    # Final size
    growth_rate=1.0
)

run.saxs().add(q_values, intensities)
```

**What it models:**
- Spherical form factor
- Size-dependent scattering
- Guinier-to-Porod transition

### 3. `simulate_waxs_time_series_data()`
Simulates WAXS evolution during crystallization:
- **Peaks emerge**: Crystalline peaks grow from amorphous background
- **Peaks sharpen**: Better crystallinity over time
- **Background decreases**: Less amorphous material

```python
from time_series_simulations import simulate_waxs_time_series_data

times, two_theta, intensities = simulate_waxs_time_series_data(
    time_points=[0, 60, 120, 180, 300, 600],
    peaks=[(30, 100), (35, 80), (62, 60)],  # Cu2O peaks
    crystallization_rate=1.0
)

run.waxs().add(two_theta, intensities)
```

**What it models:**
- Amorphous-to-crystalline transition
- Peak sharpening with time
- Multiple crystalline phases

### 4. `create_fake_image_series()`
Creates microscopy images showing morphology evolution:
```python
from time_series_simulations import create_fake_image_series

image_paths = create_fake_image_series(
    output_dir=Path("./images"),
    n_images=5,
    time_points=[0, 60, 120, 300, 600],
    pattern="sem",  # or "tem"
    particle_growth=True
)

run.sem().add(image_paths)
```

## 📈 Enhanced Plotting Functions

All plotting functions now support **three plot types**:

### UV-Vis Plotting

```python
# 1. Spectrum at specific time
run.uvvis().plot(plot_type="spectrum", time_point=180)
# Shows: wavelength vs absorbance at t=180s

# 2. Kinetics at specific wavelength
run.uvvis().plot(plot_type="kinetics", wavelength=520)
# Shows: absorbance vs time at 520nm (growth curve!)

# 3. Full 2D evolution (heatmap)
run.uvvis().plot(plot_type="heatmap")
# Shows: time vs wavelength, color = absorbance
# Beautiful visualization of peak shift!
```

### SAXS Plotting

```python
# 1. Profile at specific time
run.saxs().plot(plot_type="profile", time_point=300, loglog=True)
# Shows: I(q) at t=300s

# 2. Intensity kinetics at specific q
run.saxs().plot(plot_type="kinetics", q_value=0.02)
# Shows: I(t) at q=0.02 (sensitive to particle size!)

# 3. Full 2D evolution
run.saxs().plot(plot_type="heatmap")
# Shows: Form factor evolution over time
```

### WAXS Plotting

```python
# 1. Pattern at specific time
run.waxs().plot(plot_type="pattern", time_point=300)
# Shows: Diffraction pattern at t=300s

# 2. Peak growth kinetics
run.waxs().plot(plot_type="kinetics", two_theta_value=30)
# Shows: Peak intensity vs time (crystallization!)

# 3. Crystallization heatmap
run.waxs().plot(plot_type="heatmap")
# Shows: Peak emergence and growth
```

## 🚀 Quick Start with Time-Series

### Complete Example

```python
from nano_organizer import DataOrganizer, RunMetadata, ReactionParams
from time_series_simulations import (
    simulate_uvvis_time_series_data,
    simulate_saxs_time_series_data,
    simulate_waxs_time_series_data
)

# 1. Setup
org = DataOrganizer("my_data")

meta = RunMetadata(
    project="Cu2O_Growth",
    experiment="2024-10-25",
    run_id="Time_Series_001",
    reaction=ReactionParams(...),
    notes="Monitoring every 60 seconds for 10 minutes"
)

run = org.create_run(meta)

# 2. Add time-series data
time_points = [0, 60, 120, 180, 240, 300, 420, 600]

# UV-Vis
times, wls, absorb = simulate_uvvis_time_series_data(time_points=time_points)
run.uvvis().add(times, wls, absorb)

# SAXS
times, qs, Is = simulate_saxs_time_series_data(time_points=time_points)
run.saxs().add(qs, Is)

# WAXS
times, tts, Is = simulate_waxs_time_series_data(time_points=time_points)
run.waxs().add(tts, Is)

# 3. Visualize evolution
run.uvvis().plot(plot_type="heatmap")         # Full evolution
run.uvvis().plot(plot_type="kinetics", wavelength=520)  # Growth curve
run.saxs().plot(plot_type="profile", time_point=300)    # Snapshot
```

## 🎨 Powerful Visualizations

### Compare Fast vs Slow Growth

```python
import matplotlib.pyplot as plt

# Load two runs
fast = org.load_run("Cu2O", "2024-10-25", "Fast_85C")
slow = org.load_run("Cu2O", "2024-10-25", "Slow_60C")

# Compare kinetics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

fast.uvvis().plot(plot_type="kinetics", wavelength=520, ax=ax1, 
                  title="Fast Growth (85°C)")
slow.uvvis().plot(plot_type="kinetics", wavelength=520, ax=ax2,
                  title="Slow Growth (60°C)")

plt.tight_layout()
plt.show()
```

### Multi-Panel Figure

```python
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 1: UV-Vis
run.uvvis().plot(plot_type="spectrum", time_point=60, ax=axes[0,0])
run.uvvis().plot(plot_type="spectrum", time_point=300, ax=axes[0,1])
run.uvvis().plot(plot_type="kinetics", wavelength=520, ax=axes[0,2])

# Row 2: SAXS
run.saxs().plot(plot_type="profile", time_point=60, ax=axes[1,0], loglog=True)
run.saxs().plot(plot_type="profile", time_point=300, ax=axes[1,1], loglog=True)
run.saxs().plot(plot_type="kinetics", q_value=0.02, ax=axes[1,2])

plt.suptitle("Nanoparticle Growth Monitoring", fontsize=16)
plt.tight_layout()
plt.show()
```

## 📊 Real Data vs Simulated

The simulations create data that looks **remarkably realistic**:

| Feature | Real Data | Simulated |
|---------|-----------|-----------|
| Peak shift | ✓ Plasmon red-shifts | ✓ Modeled |
| Noise level | ✓ Realistic S/N | ✓ Gaussian/Poisson |
| Growth kinetics | ✓ Exponential saturation | ✓ Exponential model |
| Form factor | ✓ Spherical particles | ✓ Spherical FF |
| Crystallization | ✓ Peak sharpening | ✓ Width decreases |

## 🔬 Use Cases

### 1. Method Development
Test analysis pipelines with realistic synthetic data before expensive beamtime.

### 2. Student Training
Students learn to interpret time-resolved data without wasting materials.

### 3. Presentation
Generate example data for proposals, presentations, papers.

### 4. Algorithm Testing
Benchmark peak-finding, size extraction, kinetics fitting algorithms.

### 5. Teaching
Show students what to expect in real experiments.

## 📝 Tips for Real Data

When We have real time-series data:

### CSV Format
```csv
time_s,wavelength_nm,absorbance
0,200,0.10
0,201,0.11
...
60,200,0.15
60,201,0.16
...
```

### Adding to Database
```python
import pandas as pd

# If We have separate time, wavelength, absorbance arrays
times_array = ...
wavelengths_array = ...
absorbance_array = ...

# Just add them!
run.uvvis().add(times_array, wavelengths_array, absorbance_array)

# Or from DataFrame
df = pd.read_csv("my_data.csv")
run.uvvis().add(
    df['time_s'].tolist(),
    df['wavelength_nm'].tolist(),
    df['absorbance'].tolist()
)
```

### Plotting
The plotting functions **automatically detect** if data is time-series and adjust accordingly!

```python
# If single spectrum: plots wavelength vs absorbance
# If time-series: asks what We want to see
run.uvvis().plot()  # Will make heatmap if time-series
```

## 🎯 Key Advantages

1. **Realistic**: Models actual experimental physics
2. **Flexible**: Adjust growth rates, particle sizes, etc.
3. **Compatible**: Works with existing NanoOrganizer infrastructure
4. **Automatic**: Plotting functions detect time-series data
5. **Powerful**: Multiple visualization options
6. **Educational**: Students see what real data looks like

## 📚 Summary

### New Files
- **`time_series_simulations.py`**: Simulation functions
- **`nano_organizer_with_timeseries.py`**: Updated library with enhanced plotting

### New Functions
- `simulate_uvvis_time_series_data()` - UV-Vis evolution
- `simulate_saxs_time_series_data()` - SAXS particle growth  
- `simulate_waxs_time_series_data()` - WAXS crystallization
- `create_fake_image_series()` - Microscopy evolution

### Enhanced Plotting
- All modalities support: `spectrum/profile`, `kinetics`, and `heatmap` views
- Automatic time-series detection
- Beautiful 2D visualizations
- Easy comparison of conditions

### Usage
```python
# Generate data
times, wls, abs = simulate_uvvis_time_series_data(time_points=[0, 60, 120, ...])

# Add to database (same as before!)
run.uvvis().add(times, wls, abs)

# Enhanced plotting (new!)
run.uvvis().plot(plot_type="heatmap")        # Full evolution
run.uvvis().plot(plot_type="kinetics", wavelength=520)  # Growth
run.uvvis().plot(plot_type="spectrum", time_point=180)  # Snapshot
```

## 🚀 Ready to Use!

Everything is backward compatible - Our existing code still works! The new features are opt-in through plot parameters.

**Try it out:**
```python
python
>>> from time_series_simulations import simulate_uvvis_time_series_data
>>> times, wls, abs = simulate_uvvis_time_series_data()
>>> print(f"Generated {len(set(times))} time points")
>>> print(f"Each has {len(times)//len(set(times))} wavelengths")
```

---

**Perfect for droplet reactor experiments where We're monitoring synthesis in real-time! 🔬✨**