#!/usr/bin/env python3
"""
NanoOrganizer - Quick Reference Guide for Students

This guide shows common tasks and workflows.
"""

# ============================================================================
# BASIC SETUP
# ============================================================================

from NanoOrganizer import (
    DataOrganizer, 
    RunMetadata, 
    ReactionParams, 
    ChemicalSpec,
    save_time_series_to_csv
)
from pathlib import Path

# Initialize organizer
org = DataOrganizer("./MyProject")

# ============================================================================
# CREATING A RUN
# ============================================================================

# Define your experiment metadata
metadata = RunMetadata(
    project="Project_Cu2O",
    experiment="2024-10-20",  # Usually the date
    run_id="Cu2O_V1",
    sample_id="Sample_001",
    reaction=ReactionParams(
        chemicals=[
            ChemicalSpec(name="CuCl2", concentration=0.1, 
                        concentration_unit="mM", volume_uL=500),
            ChemicalSpec(name="NaOH", concentration=0.05, 
                        concentration_unit="mM", volume_uL=200),
        ],
        temperature_C=60.0,
        stir_time_s=300,
        reaction_time_s=1800,
        pH=7.5,
        solvent="Water",
        conductor="Your Name",
        description="Brief description of synthesis"
    ),
    notes="Any additional notes",
    tags=["Cu2O", "optimization"]
)

# Create the run
run = org.create_run(metadata)

# ============================================================================
# LINKING DATA FILES
# ============================================================================

# --- Option 1: You already have CSV files ---
existing_csv_files = [
    "/path/to/uvvis_001.csv",
    "/path/to/uvvis_002.csv",
    "/path/to/uvvis_003.csv",
]
time_points = [0, 30, 60]  # seconds

run.uvvis.link_data(
    existing_csv_files,
    time_points=time_points,
    metadata={"instrument": "Agilent 8453"}
)

# --- Option 2: Generate data and save to CSV ---
# If you have time-series arrays: times, wavelengths, absorbance
times = [0, 0, 0, ..., 30, 30, 30, ..., 60, 60, 60, ...]
wavelengths = [200, 201, 202, ..., 200, 201, 202, ...]
absorbance = [0.1, 0.12, 0.11, ..., 0.3, 0.32, 0.31, ...]

# Save to CSV files
output_dir = Path("./data/uvvis")
csv_files = save_time_series_to_csv(
    output_dir, 
    prefix="uvvis",
    times=times,
    x_values=wavelengths,
    y_values=absorbance,
    x_name="wavelength",
    y_name="absorbance"
)

# Link to run
run.uvvis.link_data(csv_files, time_points=[0, 30, 60])

# --- Similar for other data types ---

# SAXS
run.saxs.link_data(saxs_csv_files, time_points=time_points,
                   metadata={"beamline": "CHESS"})

# WAXS
run.waxs.link_data(waxs_csv_files, time_points=time_points,
                   metadata={"instrument": "Bruker D8"})

# SEM/TEM images
run.sem.link_data(["/path/to/sem1.png", "/path/to/sem2.png"],
                  metadata={"magnification": "50kX"})

# ============================================================================
# SAVING AND LOADING
# ============================================================================

# Save everything (creates JSON metadata files)
org.save()

# Later: load from disk
org = DataOrganizer.load("./MyProject")

# List all runs
print(org.list_runs())

# Get specific run
run = org.get_run("Project_Cu2O", "2024-10-20", "Cu2O_V1")

# ============================================================================
# LOADING AND ANALYZING DATA
# ============================================================================

# Load UV-Vis data (lazy loading)
data = run.uvvis.load()
# Returns: {'times': array, 'wavelengths': array, 'absorbance': array}

# Access the arrays
times = data['times']
wavelengths = data['wavelengths']
absorbance = data['absorbance']

# Do your own analysis
import numpy as np
unique_times = np.unique(times)
print(f"Measured at times: {unique_times}")

# ============================================================================
# VISUALIZATION
# ============================================================================

import matplotlib.pyplot as plt

# --- UV-Vis Plots ---

# 1. Single spectrum at specific time
run.uvvis.plot(plot_type="spectrum", time_point=180)
plt.savefig("spectrum.png")
plt.show()

# 2. Growth kinetics at specific wavelength
run.uvvis.plot(plot_type="kinetics", wavelength=520)
plt.savefig("kinetics.png")
plt.show()

# 3. Full evolution heatmap
run.uvvis.plot(plot_type="heatmap")
plt.savefig("heatmap.png")
plt.show()

# --- SAXS Plots ---

# Profile at specific time
run.saxs.plot(plot_type="profile", time_point=300, loglog=True)
plt.show()

# Intensity vs time at specific q
run.saxs.plot(plot_type="kinetics", q_value=0.02)
plt.show()

# Evolution heatmap
run.saxs.plot(plot_type="heatmap")
plt.show()

# --- WAXS Plots ---

# Diffraction pattern at specific time
run.waxs.plot(plot_type="pattern", time_point=300)
plt.show()

# Peak growth at specific 2θ
run.waxs.plot(plot_type="kinetics", two_theta_value=30)
plt.show()

# Crystallization heatmap
run.waxs.plot(plot_type="heatmap")
plt.show()

# --- Images ---

# Display single image
run.sem.plot(index=0)
plt.show()

# ============================================================================
# COMPARING MULTIPLE RUNS
# ============================================================================

run1 = org.get_run("Project_Cu2O", "2024-10-20", "Cu2O_V1")
run2 = org.get_run("Project_Cu2O", "2024-10-20", "Cu2O_V2")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

run1.uvvis.plot(plot_type="kinetics", wavelength=520, ax=axes[0],
                title=f"Run 1: {run1.metadata.reaction.temperature_C}°C")
run2.uvvis.plot(plot_type="kinetics", wavelength=520, ax=axes[1],
                title=f"Run 2: {run2.metadata.reaction.temperature_C}°C")

plt.tight_layout()
plt.savefig("comparison.png")
plt.show()

# ============================================================================
# VALIDATION
# ============================================================================

# Check if all data files exist
results = org.validate_all()

# Or check individual run
is_valid = run.uvvis.validate()
if not is_valid:
    print("Some UV-Vis files are missing!")

# ============================================================================
# ADVANCED: CUSTOM ANALYSIS
# ============================================================================

# Example: Extract peak position vs time
def extract_peak_positions(run):
    """Extract UV-Vis peak positions over time."""
    data = run.uvvis.load()
    
    unique_times = np.unique(data['times'])
    peak_positions = []
    
    for t in unique_times:
        mask = data['times'] == t
        wl = data['wavelengths'][mask]
        abs_val = data['absorbance'][mask]
        
        # Find peak
        peak_idx = np.argmax(abs_val)
        peak_positions.append(wl[peak_idx])
    
    return unique_times, np.array(peak_positions)

# Use it
times, peaks = extract_peak_positions(run)
plt.plot(times, peaks, 'o-')
plt.xlabel('Time (s)')
plt.ylabel('Peak Position (nm)')
plt.title('Plasmon Peak Shift During Growth')
plt.show()

# ============================================================================
# CSV FILE FORMAT
# ============================================================================

"""
Each CSV file should have format:

wavelength,absorbance
200.0,0.05
201.0,0.06
202.0,0.07
...

Or for SAXS:
q,intensity
0.01,1000.0
0.011,950.0
...

Or for WAXS:
two_theta,intensity
10.0,50.0
10.1,52.0
...

Time information comes from:
1. The time_points list you provide
2. Or filename (if it contains time info like "uvvis_t0060s.csv")
3. Or sequential order (file 1 = t[0], file 2 = t[1], etc.)
"""

# ============================================================================
# TIPS AND BEST PRACTICES
# ============================================================================

"""
1. ORGANIZATION:
   - Use descriptive project/experiment/run_id names
   - Include dates in experiment names (e.g., "2024-10-20")
   - Add meaningful tags for easy searching

2. DATA FILES:
   - Keep raw data files organized by technique
   - Use consistent naming (uvvis_001.csv, uvvis_002.csv, ...)
   - Include headers in CSV files
   
3. METADATA:
   - Record all experimental conditions
   - Add notes about anything unusual
   - Include operator name and instrument details

4. WORKFLOW:
   - Create run → Link data → Save → Validate
   - Always save() after adding data
   - Validate before important analysis

5. MEMORY:
   - Data is loaded lazily (only when you call .load())
   - This makes the system fast for large datasets
   - You can load/unload data as needed

6. PLOTTING:
   - Use plot_type="heatmap" for overview
   - Use plot_type="kinetics" to track specific features
   - Use plot_type="spectrum/profile/pattern" for snapshots

7. BACKUP:
   - The .metadata folder is small - easy to backup
   - Your data files stay where you put them
   - Version control friendly (JSON metadata)
"""

print("=" * 70)
print("  Quick Reference Guide")
print("=" * 70)
print("\nSee the code above for common usage patterns!")
print("For full demo, run: python demo_nanoorganizer.py")