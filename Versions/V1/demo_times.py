#!/usr/bin/env python3
"""
Complete Demo: NanoOrganizer with Time-Series Data

This demonstrates the full workflow:
1. Create organizer and runs
2. Generate simulated data → save to CSV files
3. Link CSV files to runs
4. Save metadata
5. Load organizer and validate
6. Load data and visualize
"""

import sys
from pathlib import Path
import shutil

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from NanoOrganizer import (
    DataOrganizer, RunMetadata, ReactionParams, ChemicalSpec,
    save_time_series_to_csv
)

# Import simulation functions (from your uploaded file)
try:
    from time_series_simulations import (
        simulate_uvvis_time_series_data,
        simulate_saxs_time_series_data,
        simulate_waxs_time_series_data
    )
    HAS_SIMULATIONS = True
except ImportError:
    HAS_SIMULATIONS = False
    print("⚠ time_series_simulations.py not found - will create dummy data")

import numpy as np


def create_dummy_data(time_points):
    """Create dummy time-series data if simulations not available."""
    times, x_vals, y_vals = [], [], []
    for t in time_points:
        x = np.linspace(200, 800, 300)
        y = np.random.rand(300) * (1 + t/600)
        times.extend([t] * len(x))
        x_vals.extend(x)
        y_vals.extend(y)
    return times, x_vals, y_vals


print("=" * 70)
print("  NANOORGANIZER - COMPLETE DEMO")
print("=" * 70)

# ============================================================================
# STEP 1: Setup and Create Organizer
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 1: Initialize DataOrganizer")
print("=" * 70)

demo_dir = Path("./NanoOrganizer_Demo")
if demo_dir.exists():
    shutil.rmtree(demo_dir)

org = DataOrganizer(demo_dir)
print(f"✓ Created DataOrganizer at: {demo_dir}")

# ============================================================================
# STEP 2: Create Experimental Runs
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 2: Create Experimental Runs")
print("=" * 70)

# Run 1: Low temperature Cu2O synthesis
print("\n--- Run 1: Cu2O Low Temperature ---")
run1_meta = RunMetadata(
    project="Project_Cu2O",
    experiment="2024-10-20",
    run_id="Cu2O_V1_LowTemp",
    sample_id="Sample_001",
    reaction=ReactionParams(
        chemicals=[
            ChemicalSpec(name="CuCl2", concentration=0.1, concentration_unit="mM", volume_uL=500),
            ChemicalSpec(name="NaOH", concentration=0.05, concentration_unit="mM", volume_uL=200),
            ChemicalSpec(name="AA", concentration=1.0, concentration_unit="mM", volume_uL=100),
        ],
        temperature_C=60.0,
        stir_time_s=300,
        reaction_time_s=1800,
        pH=7.5,
        solvent="Water",
        conductor="Dr. Zhang",
        description="Low temperature Cu2O synthesis"
    ),
    notes="First attempt at low temperature synthesis",
    tags=["Cu2O", "low_temp", "optimization"]
)

run1 = org.create_run(run1_meta)

# Run 2: High temperature variant
print("\n--- Run 2: Cu2O High Temperature ---")
run2_meta = RunMetadata(
    project="Project_Cu2O",
    experiment="2024-10-20",
    run_id="Cu2O_V2_HighTemp",
    sample_id="Sample_002",
    reaction=ReactionParams(
        chemicals=[
            ChemicalSpec(name="CuCl2", concentration=0.1, concentration_unit="mM", volume_uL=500),
            ChemicalSpec(name="NaOH", concentration=0.05, concentration_unit="mM", volume_uL=200),
            ChemicalSpec(name="AA", concentration=1.0, concentration_unit="mM", volume_uL=100),
        ],
        temperature_C=80.0,  # Higher temperature
        stir_time_s=300,
        reaction_time_s=1200,  # Shorter reaction time
        pH=7.5,
        solvent="Water",
        conductor="Dr. Zhang",
        description="High temperature Cu2O synthesis - faster kinetics"
    ),
    notes="Testing higher temperature for faster growth",
    tags=["Cu2O", "high_temp", "optimization"]
)

run2 = org.create_run(run2_meta)

# ============================================================================
# STEP 3: Generate and Save Time-Series Data
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 3: Generate and Save Time-Series Data")
print("=" * 70)

time_points = [0, 30, 60, 120, 180, 300, 600]  # seconds

# --- Run 1 Data ---
print("\n--- Run 1: Low Temperature (slower growth) ---")

# UV-Vis data for Run 1
print("\n  1. UV-Vis spectroscopy...")
if HAS_SIMULATIONS:
    times, wls, absorb = simulate_uvvis_time_series_data(
        time_points=time_points,
        initial_peak=480,
        final_peak=530,
        growth_rate=1.0,  # Slower growth
        n_wavelengths=300
    )
else:
    times, wls, absorb = create_dummy_data(time_points)

# Save to CSV files
uvvis_dir1 = demo_dir / "Project_Cu2O" / "UV_Vis" / "2024-10-20" / "Cu2O_V1_LowTemp"
uvvis_files1 = save_time_series_to_csv(
    uvvis_dir1, "uvvis", times, wls, absorb,
    x_name="wavelength", y_name="absorbance"
)

# Link to run
run1.uvvis.link_data(
    uvvis_files1,
    time_points=time_points,
    metadata={"instrument": "Agilent 8453", "operator": "Dr. Zhang"}
)

# SAXS data for Run 1
print("\n  2. SAXS scattering...")
if HAS_SIMULATIONS:
    times, qs, Is = simulate_saxs_time_series_data(
        time_points=time_points,
        initial_size_nm=2.0,
        final_size_nm=10.0,
        growth_rate=1.0
    )
else:
    times, qs, Is = create_dummy_data(time_points)

saxs_dir1 = demo_dir / "Project_Cu2O" / "SAXS" / "2024-10-20" / "Cu2O_V1_LowTemp"
saxs_files1 = save_time_series_to_csv(
    saxs_dir1, "saxs", times, qs, Is,
    x_name="q", y_name="intensity"
)

run1.saxs.link_data(
    saxs_files1,
    time_points=time_points,
    metadata={"beamline": "CHESS", "energy_keV": 10.0}
)

# WAXS data for Run 1
print("\n  3. WAXS diffraction...")
if HAS_SIMULATIONS:
    times, tts, Is = simulate_waxs_time_series_data(
        time_points=time_points,
        crystallization_rate=1.0
    )
else:
    times, tts, Is = create_dummy_data(time_points)

waxs_dir1 = demo_dir / "Project_Cu2O" / "WAXS" / "2024-10-20" / "Cu2O_V1_LowTemp"
waxs_files1 = save_time_series_to_csv(
    waxs_dir1, "waxs", times, tts, Is,
    x_name="two_theta", y_name="intensity"
)

run1.waxs.link_data(
    waxs_files1,
    time_points=time_points,
    metadata={"instrument": "Bruker D8", "wavelength_A": 1.54}
)

# --- Run 2 Data ---
print("\n--- Run 2: High Temperature (faster growth) ---")

# UV-Vis for Run 2
print("\n  1. UV-Vis spectroscopy...")
if HAS_SIMULATIONS:
    times, wls, absorb = simulate_uvvis_time_series_data(
        time_points=time_points,
        initial_peak=480,
        final_peak=540,  # Larger particles
        growth_rate=1.5,  # Faster growth
        n_wavelengths=300
    )
else:
    times, wls, absorb = create_dummy_data(time_points)

uvvis_dir2 = demo_dir / "Project_Cu2O" / "UV_Vis" / "2024-10-20" / "Cu2O_V2_HighTemp"
uvvis_files2 = save_time_series_to_csv(
    uvvis_dir2, "uvvis", times, wls, absorb,
    x_name="wavelength", y_name="absorbance"
)

run2.uvvis.link_data(
    uvvis_files2,
    time_points=time_points,
    metadata={"instrument": "Agilent 8453", "operator": "Dr. Zhang"}
)

# SAXS for Run 2
print("\n  2. SAXS scattering...")
if HAS_SIMULATIONS:
    times, qs, Is = simulate_saxs_time_series_data(
        time_points=time_points,
        initial_size_nm=2.0,
        final_size_nm=12.0,  # Larger particles
        growth_rate=1.5
    )
else:
    times, qs, Is = create_dummy_data(time_points)

saxs_dir2 = demo_dir / "Project_Cu2O" / "SAXS" / "2024-10-20" / "Cu2O_V2_HighTemp"
saxs_files2 = save_time_series_to_csv(
    saxs_dir2, "saxs", times, qs, Is,
    x_name="q", y_name="intensity"
)

run2.saxs.link_data(
    saxs_files2,
    time_points=time_points,
    metadata={"beamline": "CHESS", "energy_keV": 10.0}
)

# ============================================================================
# STEP 4: Save Metadata
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 4: Save Metadata to JSON")
print("=" * 70)

org.save()

print(f"\n✓ Complete database created at: {demo_dir}")
print(f"  - Metadata: {demo_dir / '.metadata'}")
print(f"  - Data files organized by project/experiment/run")

# ============================================================================
# STEP 5: Reload and Validate
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 5: Reload DataOrganizer and Validate")
print("=" * 70)

# Simulate closing and reopening
del org, run1, run2

# Load from disk
org_loaded = DataOrganizer.load(demo_dir)

print(f"\n✓ Available runs:")
for run_key in org_loaded.list_runs():
    print(f"  - {run_key}")

# Validate all data
validation_results = org_loaded.validate_all()

# ============================================================================
# STEP 6: Load and Visualize Data
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 6: Load and Visualize Data")
print("=" * 70)

# Get a run
run = org_loaded.get_run("Project_Cu2O", "2024-10-20", "Cu2O_V1_LowTemp")
print(f"\n✓ Loaded run: {run.metadata.run_id}")
print(f"  Temperature: {run.metadata.reaction.temperature_C}°C")
print(f"  Chemicals: {', '.join([c.name for c in run.metadata.reaction.chemicals])}")

# Load UV-Vis data (lazy loading)
print("\n--- Loading UV-Vis Data ---")
uvvis_data = run.uvvis.load()
print(f"  Total data points: {len(uvvis_data['times'])}")
print(f"  Time points: {np.unique(uvvis_data['times'])}")
print(f"  Wavelength range: {uvvis_data['wavelengths'].min():.0f} - {uvvis_data['wavelengths'].max():.0f} nm")

# Load SAXS data
print("\n--- Loading SAXS Data ---")
saxs_data = run.saxs.load()
print(f"  Total data points: {len(saxs_data['times'])}")
print(f"  q range: {saxs_data['q'].min():.3f} - {saxs_data['q'].max():.3f} 1/Å")

# ============================================================================
# STEP 7: Create Visualizations
# ============================================================================
print("\n" + "=" * 70)
print("  STEP 7: Create Visualizations")
print("=" * 70)

try:
    import matplotlib.pyplot as plt
    
    plot_dir = demo_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    # UV-Vis plots for Run 1
    print("\n--- UV-Vis Plots (Run 1: Low Temp) ---")
    
    # Spectrum at different times
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    run.uvvis.plot(plot_type="spectrum", time_point=60, ax=axes[0], title="t=60s")
    run.uvvis.plot(plot_type="spectrum", time_point=180, ax=axes[1], title="t=180s")
    run.uvvis.plot(plot_type="spectrum", time_point=600, ax=axes[2], title="t=600s")
    plt.tight_layout()
    plt.savefig(plot_dir / "uvvis_run1_spectra.png", dpi=150)
    plt.close()
    print("  ✓ Saved: uvvis_run1_spectra.png")
    
    # Kinetics
    fig, ax = plt.subplots(figsize=(10, 6))
    run.uvvis.plot(plot_type="kinetics", wavelength=520, ax=ax)
    plt.savefig(plot_dir / "uvvis_run1_kinetics.png", dpi=150)
    plt.close()
    print("  ✓ Saved: uvvis_run1_kinetics.png")
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    run.uvvis.plot(plot_type="heatmap", ax=ax)
    plt.savefig(plot_dir / "uvvis_run1_heatmap.png", dpi=150)
    plt.close()
    print("  ✓ Saved: uvvis_run1_heatmap.png")
    
    # SAXS plots
    print("\n--- SAXS Plots (Run 1) ---")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    run.saxs.plot(plot_type="profile", time_point=300, loglog=True, ax=ax)
    plt.savefig(plot_dir / "saxs_run1_profile.png", dpi=150)
    plt.close()
    print("  ✓ Saved: saxs_run1_profile.png")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    run.saxs.plot(plot_type="kinetics", q_value=0.02, ax=ax)
    plt.savefig(plot_dir / "saxs_run1_kinetics.png", dpi=150)
    plt.close()
    print("  ✓ Saved: saxs_run1_kinetics.png")
    
    # Compare two runs
    print("\n--- Comparing Run 1 vs Run 2 ---")
    run2 = org_loaded.get_run("Project_Cu2O", "2024-10-20", "Cu2O_V2_HighTemp")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Row 1: UV-Vis spectra at t=180s
    run.uvvis.plot(plot_type="spectrum", time_point=180, ax=axes[0, 0], 
                   title="Run 1 (60°C) - t=180s")
    run2.uvvis.plot(plot_type="spectrum", time_point=180, ax=axes[0, 1], 
                    title="Run 2 (80°C) - t=180s")
    
    # Row 2: Kinetics comparison
    run.uvvis.plot(plot_type="kinetics", wavelength=520, ax=axes[1, 0], 
                   title="Run 1 Growth Kinetics")
    run2.uvvis.plot(plot_type="kinetics", wavelength=520, ax=axes[1, 1], 
                    title="Run 2 Growth Kinetics (Faster)")
    
    plt.tight_layout()
    plt.savefig(plot_dir / "comparison_run1_vs_run2.png", dpi=150)
    plt.close()
    print("  ✓ Saved: comparison_run1_vs_run2.png")
    
    print(f"\n✓ All plots saved to: {plot_dir}")
    
except ImportError:
    print("\n⚠ matplotlib not available - skipping plots")
    print("  Install with: pip install matplotlib")

# ============================================================================
# STEP 8: Summary and Usage Examples
# ============================================================================
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)

print(f"""
✅ Created complete NanoOrganizer database
✅ 2 experimental runs with time-series data
✅ UV-Vis, SAXS, and WAXS data linked
✅ Metadata stored in JSON (easy to read/edit)
✅ Data files organized in flexible structure
✅ Lazy loading for efficient memory use
✅ Validation checks all files exist
✅ Easy visualization with built-in plotting

DATABASE STRUCTURE:
-------------------
{demo_dir}/
├── .metadata/           # JSON metadata files
│   ├── index.json       # Master index
│   ├── Project_Cu2O_2024-10-20_Cu2O_V1_LowTemp.json
│   └── Project_Cu2O_2024-10-20_Cu2O_V2_HighTemp.json
├── Project_Cu2O/        # Your data files
│   ├── UV_Vis/
│   │   └── 2024-10-20/
│   │       ├── Cu2O_V1_LowTemp/
│   │       │   ├── uvvis_001.csv
│   │       │   ├── uvvis_002.csv
│   │       │   └── ...
│   │       └── Cu2O_V2_HighTemp/
│   ├── SAXS/
│   └── WAXS/
└── plots/               # Generated visualizations

USAGE EXAMPLES:
---------------

# 1. Load organizer
from NanoOrganizer import DataOrganizer
org = DataOrganizer.load("{demo_dir}")

# 2. Get a run
run = org.get_run("Project_Cu2O", "2024-10-20", "Cu2O_V1_LowTemp")

# 3. Access metadata
print(run.metadata.reaction.temperature_C)
print(run.metadata.reaction.chemicals)

# 4. Load and plot data
data = run.uvvis.load()  # Lazy loading
run.uvvis.plot(plot_type="heatmap")
run.saxs.plot(plot_type="kinetics", q_value=0.02)

# 5. Compare runs
run1 = org.get_run("Project_Cu2O", "2024-10-20", "Cu2O_V1_LowTemp")
run2 = org.get_run("Project_Cu2O", "2024-10-20", "Cu2O_V2_HighTemp")

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
run1.uvvis.plot(plot_type="kinetics", wavelength=520, ax=axes[0])
run2.uvvis.plot(plot_type="kinetics", wavelength=520, ax=axes[1])
plt.show()

# 6. Validate data integrity
org.validate_all()

NEXT STEPS:
-----------
1. Check the generated plots in: {demo_dir}/plots/
2. Examine the JSON metadata in: {demo_dir}/.metadata/
3. Try loading and plotting different runs
4. Adapt the code for your actual experimental data!

KEY FEATURES:
-------------
✨ Flexible: Any directory structure works
✨ Fast: Lazy loading - only load data when needed
✨ Safe: Validation checks ensure files exist
✨ Easy: Simple API for common tasks
✨ Extensible: Add new data types easily
""")

print("\n" + "=" * 70)
print("  ✨ DEMO COMPLETE! ✨")
print("=" * 70)