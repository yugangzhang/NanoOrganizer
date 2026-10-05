# 🚀 Getting Started with NanoOrganizer

Welcome! This guide will get you up and running in 5 minutes.

## 📦 What's Included

```
NanoOrganizer.py              # Main module (~1000 lines, production-ready)
demo_nanoorganizer.py         # Complete working demo
integration_example.py        # How to integrate with your code
quick_reference.py            # Quick reference for students
README.md                     # Comprehensive documentation
ARCHITECTURE.md               # Technical design details
GETTING_STARTED.md           # This file
NanoOrganizer_Demo/          # Example database with plots
```

---

## ⚡ 5-Minute Quick Start

### 1. Run the Demo (2 minutes)

```bash
# Make sure you have dependencies
pip install numpy matplotlib

# Run the demo
python demo_nanoorganizer.py
```

This creates:
- ✅ Example database with 2 experimental runs
- ✅ UV-Vis, SAXS, WAXS time-series data
- ✅ 6+ visualization plots
- ✅ JSON metadata you can inspect

**Check the results:**
- Data: `NanoOrganizer_Demo/Project_Cu2O/`
- Metadata: `NanoOrganizer_Demo/.metadata/`
- Plots: `NanoOrganizer_Demo/plots/`

### 2. Explore the Code (3 minutes)

Open `quick_reference.py` and look at the examples:

```python
# Basic usage
from NanoOrganizer import DataOrganizer, RunMetadata, ReactionParams, ChemicalSpec

# Create organizer
org = DataOrganizer("./MyProject")

# Create run
run = org.create_run(metadata)

# Link data
run.uvvis.link_data(csv_files, time_points=[0, 30, 60])

# Save
org.save()

# Later: load and plot
org = DataOrganizer.load("./MyProject")
run = org.get_run("Project", "Experiment", "Run")
run.uvvis.plot(plot_type="heatmap")
```

### 3. Try It Yourself (Interactive)

Open Python and try:

```python
from NanoOrganizer import DataOrganizer

# Load the demo
org = DataOrganizer.load("./NanoOrganizer_Demo")

# List runs
print(org.list_runs())

# Get a run
run = org.get_run("Project_Cu2O", "2024-10-20", "Cu2O_V1_LowTemp")

# Load data
data = run.uvvis.load()
print(f"Loaded {len(data['times'])} data points")

# Plot
import matplotlib.pyplot as plt
run.uvvis.plot(plot_type="kinetics", wavelength=520)
plt.show()
```

---

## 🎯 Next Steps

### For Students

1. **Read:** `quick_reference.py` - all common patterns
2. **Understand:** Check the example plots in `NanoOrganizer_Demo/plots/`
3. **Modify:** Try changing parameters in `demo_nanoorganizer.py`
4. **Build:** Create your own project with real data

### For Integration

1. **Study:** `integration_example.py` - complete workflow
2. **Adapt:** Replace simulations with your measurements
3. **Test:** Start with a single run
4. **Scale:** Add automation for high-throughput

### For Deep Dive

1. **Architecture:** Read `ARCHITECTURE.md` for design details
2. **API:** See `README.md` for complete API reference
3. **Extend:** Add new data types or analysis methods

---

## 📋 Common Tasks

### Create a New Experiment

```python
from NanoOrganizer import *

org = DataOrganizer("./MyProject")

metadata = RunMetadata(
    project="Project_Au",
    experiment="2024-10-25",
    run_id="Au_NP_Test_001",
    sample_id="Sample_001",
    reaction=ReactionParams(
        chemicals=[
            ChemicalSpec(name="HAuCl4", concentration=0.5, 
                        concentration_unit="mM", volume_uL=1000)
        ],
        temperature_C=80.0,
        reaction_time_s=1800,
        conductor="Your Name"
    )
)

run = org.create_run(metadata)
```

### Link Existing CSV Files

```python
# Your CSV files
csv_files = [
    "/path/to/data/uvvis_001.csv",
    "/path/to/data/uvvis_002.csv",
    "/path/to/data/uvvis_003.csv",
]

# Link to run
run.uvvis.link_data(
    csv_files,
    time_points=[0, 30, 60],  # seconds
    metadata={"instrument": "Agilent 8453"}
)

# Save
org.save()
```

### Generate and Save Data

```python
from NanoOrganizer import save_time_series_to_csv

# Your measurement data
times = [0, 0, 0, ..., 30, 30, 30, ...]
wavelengths = [200, 201, 202, ..., 200, 201, ...]
absorbance = [0.1, 0.12, 0.11, ..., 0.3, 0.32, ...]

# Save to CSV files
csv_files = save_time_series_to_csv(
    output_dir="./data/uvvis",
    prefix="uvvis",
    times=times,
    x_values=wavelengths,
    y_values=absorbance,
    x_name="wavelength",
    y_name="absorbance"
)

# Link to run
run.uvvis.link_data(csv_files, time_points=[0, 30, 60])
```

### Load and Analyze

```python
# Load
org = DataOrganizer.load("./MyProject")
run = org.get_run("Project", "Experiment", "Run")

# Get metadata
print(run.metadata.reaction.temperature_C)
print(run.metadata.reaction.chemicals)

# Load data
data = run.uvvis.load()

# Analyze
import numpy as np
unique_times = np.unique(data['times'])
for t in unique_times:
    mask = data['times'] == t
    peak_wl = data['wavelengths'][mask][np.argmax(data['absorbance'][mask])]
    print(f"t={t}s: peak at {peak_wl:.1f} nm")
```

### Create Plots

```python
import matplotlib.pyplot as plt

# Single plot
run.uvvis.plot(plot_type="spectrum", time_point=180)
plt.savefig("spectrum.png")

# Multiple plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
run.uvvis.plot(plot_type="spectrum", time_point=60, ax=axes[0,0])
run.uvvis.plot(plot_type="kinetics", wavelength=520, ax=axes[0,1])
run.saxs.plot(plot_type="profile", time_point=180, ax=axes[1,0])
run.waxs.plot(plot_type="pattern", time_point=300, ax=axes[1,1])
plt.tight_layout()
plt.savefig("summary.png")
```

### Compare Multiple Runs

```python
run1 = org.get_run("Project", "Exp", "Run1")
run2 = org.get_run("Project", "Exp", "Run2")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
run1.uvvis.plot(plot_type="kinetics", wavelength=520, ax=axes[0],
                title=f"Run 1: {run1.metadata.reaction.temperature_C}°C")
run2.uvvis.plot(plot_type="kinetics", wavelength=520, ax=axes[1],
                title=f"Run 2: {run2.metadata.reaction.temperature_C}°C")
plt.show()
```

---

## 🐛 Troubleshooting

### Import Error
```python
# If you get: ImportError: No module named 'NanoOrganizer'
# Make sure NanoOrganizer.py is in your working directory or Python path
import sys
sys.path.insert(0, '/path/to/NanoOrganizer')
```

### File Not Found
```python
# Check paths are absolute
from pathlib import Path
csv_path = Path("/full/path/to/file.csv").absolute()
print(f"Using: {csv_path}")

# Validate data
validation = run.uvvis.validate()
if not validation:
    print("Some files are missing!")
```

### Memory Issues
```python
# Clear loaded data if needed
run.uvvis._loaded_data = None

# Or force reload
data = run.uvvis.load(force_reload=True)
```

### Plot Not Showing
```python
import matplotlib.pyplot as plt
run.uvvis.plot(plot_type="spectrum", time_point=180)
plt.show()  # Don't forget this!
```

---

## 📚 Learn More

- **Quick Examples:** `quick_reference.py`
- **Complete Workflow:** `integration_example.py`
- **Full Documentation:** `README.md`
- **Technical Details:** `ARCHITECTURE.md`

---

## ✅ Verification Checklist

After installation, verify everything works:

- [ ] Run `python demo_nanoorganizer.py` successfully
- [ ] See plots created in `NanoOrganizer_Demo/plots/`
- [ ] Can import: `from NanoOrganizer import DataOrganizer`
- [ ] Can load demo: `org = DataOrganizer.load("./NanoOrganizer_Demo")`
- [ ] Can create plot: `run.uvvis.plot(plot_type="heatmap")`

If all checked, you're ready! 🎉

---

## 🎓 Learning Path

1. **Day 1:** Run demo, explore outputs
2. **Day 2:** Read quick_reference.py, try examples
3. **Day 3:** Create simple project with fake data
4. **Day 4:** Integrate with your real measurements
5. **Day 5:** Build analysis pipeline, create reports

---

## 💬 Tips from the Author

1. **Start Simple:** Begin with one data type (UV-Vis)
2. **Save Often:** Call `org.save()` frequently
3. **Validate Always:** Check files exist before analysis
4. **Use Tags:** Makes searching/filtering easier later
5. **Document Well:** Add notes and metadata
6. **Organize Early:** Good structure saves time later

---

## 🚀 Ready to Start?

```bash
# Install dependencies
pip install numpy matplotlib

# Run demo
python demo_nanoorganizer.py

# Check results
ls NanoOrganizer_Demo/plots/

# Start coding!
python
>>> from NanoOrganizer import DataOrganizer
>>> org = DataOrganizer("./MyFirstProject")
```

**Welcome to NanoOrganizer! Happy experimenting! 🔬✨**

---

*Need help? Check README.md for comprehensive documentation.*