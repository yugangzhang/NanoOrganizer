#!/usr/bin/env python3
"""
Integration Example: Using NanoOrganizer with Your Existing Simulation Code

This shows how to integrate NanoOrganizer with your existing
time_series_simulations.py code for a complete workflow.
"""

from pathlib import Path
import sys


# New imports
from NanoOrganizer import (
    DataOrganizer, RunMetadata, ReactionParams, ChemicalSpec,
    save_time_series_to_csv,
    simulate_uvvis_time_series_data,
    simulate_saxs_time_series_data,
    simulate_waxs_time_series_data,
    create_fake_image_series
)

print("=" * 70)
print("  INTEGRATION EXAMPLE")
print("  Your Simulations + NanoOrganizer")
print("=" * 70)

# ============================================================================
# STEP 1: Setup Project
# ============================================================================
print("\n1. Setting up project...")

project_dir = Path("/home/yuzhang/Repos/NanoOrganizer/Demo/")
org = DataOrganizer(project_dir)
print(f"   ✓ Initialized at: {project_dir}")

# ============================================================================
# STEP 2: Define Experiment
# ============================================================================
print("\n2. Defining experiment metadata...")

metadata = RunMetadata(
    project="Project_Cu2O",
    experiment="2024-10-25",
    run_id="Cu2O_Growth_Study_001",
    sample_id="Sample_Cu2O_001",
    reaction=ReactionParams(
        chemicals=[
            ChemicalSpec(name="CuCl2", concentration=0.1, 
                        concentration_unit="mM", volume_uL=500),
            ChemicalSpec(name="NaOH", concentration=0.05, 
                        concentration_unit="mM", volume_uL=200),
            ChemicalSpec(name="Ascorbic Acid", concentration=1.0, 
                        concentration_unit="mM", volume_uL=100),
        ],
        temperature_C=65.0,
        stir_time_s=300,
        reaction_time_s=1800,
        pH=7.5,
        solvent="Water",
        conductor="Dr. Zhang",
        description="Cu2O nanoparticle growth study with time-resolved characterization"
    ),
    notes="Using your droplet reactor setup",
    tags=["Cu2O", "time_series", "optimization", "2024"]
)

run = org.create_run(metadata)
print(f"   ✓ Created run: {run.metadata.run_id}")

# ============================================================================
# STEP 3: Run Simulations (or Real Measurements)
# ============================================================================
print("\n3. Generating time-series data...")
print("   (Replace these simulations with your actual measurements!)")

# Define time points for your experiment
time_points = [0, 30, 60, 120, 180, 300, 600]  # seconds
print(f"   Time points: {time_points} seconds")

# --- UV-Vis Simulation ---
print("\n   a) UV-Vis spectroscopy...")
times, wavelengths, absorbance = simulate_uvvis_time_series_data(
    time_points=time_points,
    initial_peak=480,    # nm
    final_peak=530,      # nm
    growth_rate=1.2,
    n_wavelengths=300
)
print(f"      Generated {len(set(times))} spectra, {len(times)} total points")

# --- SAXS Simulation ---
print("\n   b) SAXS scattering...")
times_saxs, q_values, intensities = simulate_saxs_time_series_data(
    time_points=time_points,
    initial_size_nm=2.0,
    final_size_nm=10.0,
    growth_rate=1.2,
    n_points=200
)
print(f"      Generated {len(set(times_saxs))} profiles")

# --- WAXS Simulation ---
print("\n   c) WAXS diffraction...")
times_waxs, two_theta, intensities_waxs = simulate_waxs_time_series_data(
    time_points=time_points,
    crystallization_rate=1.2,
    peaks=[(30.0, 100), (35.0, 80), (62.0, 60)]  # Cu2O peaks
)
print(f"      Generated {len(set(times_waxs))} patterns")

# --- Microscopy Images ---
print("\n   d) SEM images...")
image_dir = project_dir / "microscopy" / "sem_images"
try:
    sem_paths = create_fake_image_series(
        image_dir,
        n_images=5,
        time_points=time_points[:5],
        pattern="sem",
        particle_growth=True
    )
    has_images = True
except Exception as e:
    print(f"      ⚠ Could not create images: {e}")
    has_images = False

# ============================================================================
# STEP 4: Save Data to CSV Files
# ============================================================================
print("\n4. Saving data to CSV files...")

# Create organized directory structure
uvvis_dir = project_dir / "Project_Cu2O" / "UV_Vis" / "2024-10-25" / "Cu2O_Growth_Study_001"
saxs_dir = project_dir / "Project_Cu2O" / "SAXS" / "2024-10-25" / "Cu2O_Growth_Study_001"
waxs_dir = project_dir / "Project_Cu2O" / "WAXS" / "2024-10-25" / "Cu2O_Growth_Study_001"

# Save UV-Vis
uvvis_files = save_time_series_to_csv(
    uvvis_dir, "uvvis",
    times, wavelengths, absorbance,
    x_name="wavelength", y_name="absorbance"
)

# Save SAXS
saxs_files = save_time_series_to_csv(
    saxs_dir, "saxs",
    times_saxs, q_values, intensities,
    x_name="q", y_name="intensity"
)

# Save WAXS
waxs_files = save_time_series_to_csv(
    waxs_dir, "waxs",
    times_waxs, two_theta, intensities_waxs,
    x_name="two_theta", y_name="intensity"
)

# ============================================================================
# STEP 5: Link Data to Run
# ============================================================================
print("\n5. Linking data to run...")

run.uvvis.link_data(
    uvvis_files,
    time_points=time_points,
    metadata={
        "instrument": "Agilent 8453",
        "operator": "Dr. Zhang",
        "wavelength_range_nm": [200, 800],
        "integration_time_ms": 100
    }
)

run.saxs.link_data(
    saxs_files,
    time_points=time_points,
    metadata={
        "beamline": "CHESS",
        "energy_keV": 10.0,
        "detector": "Pilatus 300K",
        "sample_detector_distance_m": 1.5
    }
)

run.waxs.link_data(
    waxs_files,
    time_points=time_points,
    metadata={
        "instrument": "Bruker D8",
        "wavelength_A": 1.54,
        "detector": "Lynxeye XE",
        "scan_speed_deg_per_min": 2.0
    }
)

if has_images:
    run.sem.link_data(
        sem_paths,
        metadata={
            "instrument": "FEI Quanta 250",
            "magnification": "50kX",
            "voltage_kV": 15.0,
            "operator": "Dr. Zhang"
        }
    )

print("   ✓ All data linked")

# ============================================================================
# STEP 6: Save Everything
# ============================================================================
print("\n6. Saving metadata...")
org.save()
print(f"   ✓ Metadata saved to: {project_dir / '.metadata'}")

# ============================================================================
# STEP 7: Demonstrate Reloading and Analysis
# ============================================================================
print("\n" + "=" * 70)
print("  RELOAD AND ANALYZE")
print("=" * 70)

# Clear everything from memory (simulating restart)
del org, run

# Reload
print("\n7. Reloading from disk...")
org = DataOrganizer.load(project_dir)
print(f"   ✓ Loaded {len(org.list_runs())} runs")

# Get the run
run = org.get_run("Project_Cu2O", "2024-10-25", "Cu2O_Growth_Study_001")
print(f"   ✓ Retrieved: {run.metadata.run_id}")

# Access metadata
print(f"\n   Metadata:")
print(f"   - Temperature: {run.metadata.reaction.temperature_C}°C")
print(f"   - Chemicals: {', '.join([c.name for c in run.metadata.reaction.chemicals])}")
print(f"   - Conductor: {run.metadata.reaction.conductor}")
print(f"   - Tags: {', '.join(run.metadata.tags)}")

# Validate data
print("\n8. Validating data integrity...")
validation = run.validate()
all_valid = all(validation.values())
print(f"   {'✓' if all_valid else '⚠'} Validation results:")
for data_type, is_valid in validation.items():
    if run.__dict__[data_type].link.file_paths:  # Only show if data exists
        print(f"      {data_type}: {'✓' if is_valid else '⚠'}")

# ============================================================================
# STEP 8: Load and Analyze Data
# ============================================================================
print("\n9. Loading and analyzing data...")

# UV-Vis
uvvis_data = run.uvvis.load()
print(f"\n   UV-Vis:")
print(f"   - Total points: {len(uvvis_data['times'])}")
print(f"   - Time range: {uvvis_data['times'].min():.0f} - {uvvis_data['times'].max():.0f} s")
print(f"   - Wavelength range: {uvvis_data['wavelengths'].min():.0f} - {uvvis_data['wavelengths'].max():.0f} nm")
print(f"   - Absorbance range: {uvvis_data['absorbance'].min():.3f} - {uvvis_data['absorbance'].max():.3f}")

# SAXS
saxs_data = run.saxs.load()
print(f"\n   SAXS:")
print(f"   - Total points: {len(saxs_data['times'])}")
print(f"   - q range: {saxs_data['q'].min():.3f} - {saxs_data['q'].max():.3f} 1/Å")

# WAXS
waxs_data = run.waxs.load()
print(f"\n   WAXS:")
print(f"   - Total points: {len(waxs_data['times'])}")
print(f"   - 2θ range: {waxs_data['two_theta'].min():.1f} - {waxs_data['two_theta'].max():.1f}°")

# ============================================================================
# STEP 9: Visualizations
# ============================================================================
print("\n10. Creating visualizations...")

try:
    import matplotlib.pyplot as plt
    
    plot_dir = project_dir / "analysis_plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Multi-panel summary plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # Row 1: UV-Vis
    run.uvvis.plot(plot_type="spectrum", time_point=60, ax=axes[0, 0], 
                   title="UV-Vis: t=60s")
    run.uvvis.plot(plot_type="kinetics", wavelength=520, ax=axes[0, 1], 
                   title="Growth Kinetics (λ=520nm)")
    
    # Row 2: SAXS
    run.saxs.plot(plot_type="profile", time_point=180, loglog=True, ax=axes[1, 0],
                  title="SAXS: t=180s")
    run.saxs.plot(plot_type="kinetics", q_value=0.02, ax=axes[1, 1],
                  title="SAXS Kinetics (q=0.02)")
    
    # Row 3: WAXS
    run.waxs.plot(plot_type="pattern", time_point=300, ax=axes[2, 0],
                  title="WAXS: t=300s")
    run.waxs.plot(plot_type="kinetics", two_theta_value=30, ax=axes[2, 1],
                  title="Peak Growth (2θ=30°)")
    
    plt.tight_layout()
    summary_file = plot_dir / "growth_summary.png"
    plt.savefig(summary_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {summary_file}")
    
    # Individual heatmaps
    for data_type, title in [("uvvis", "UV-Vis Evolution"), 
                              ("saxs", "SAXS Evolution"), 
                              ("waxs", "WAXS Crystallization")]:
        fig, ax = plt.subplots(figsize=(12, 6))
        getattr(run, data_type).plot(plot_type="heatmap", ax=ax, title=title)
        filename = plot_dir / f"{data_type}_heatmap.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: {filename}")
    
except ImportError:
    print("   ⚠ matplotlib not available - skipping plots")

# ============================================================================
# STEP 10: Custom Analysis Example
# ============================================================================
print("\n11. Custom analysis example...")

import numpy as np

def extract_peak_shift(run):
    """Extract UV-Vis peak position over time."""
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

times, peaks = extract_peak_shift(run)
print(f"\n   Peak position evolution:")
for t, peak in zip(times, peaks):
    print(f"   t = {t:4.0f}s → λ = {peak:.1f} nm")

# Calculate growth rate
if len(peaks) > 1:
    shift = peaks[-1] - peaks[0]
    time_span = times[-1] - times[0]
    rate = shift / time_span
    print(f"\n   ✓ Peak shift: {shift:.1f} nm over {time_span:.0f}s")
    print(f"   ✓ Growth rate: {rate:.3f} nm/s")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)

print(f"""
✅ Complete workflow demonstrated!

WHAT WE DID:
1. ✓ Created DataOrganizer
2. ✓ Defined experimental metadata
3. ✓ Generated time-series data (UV-Vis, SAXS, WAXS)
4. ✓ Saved data to CSV files in organized structure
5. ✓ Linked data to run
6. ✓ Saved metadata to JSON
7. ✓ Reloaded from disk
8. ✓ Validated data integrity
9. ✓ Loaded and analyzed data
10. ✓ Created visualizations
11. ✓ Performed custom analysis

YOUR PROJECT STRUCTURE:
{project_dir}/
├── .metadata/                          # JSON metadata (version control this!)
├── Project_Cu2O/                       # Your data files
│   ├── UV_Vis/2024-10-25/...
│   ├── SAXS/2024-10-25/...
│   └── WAXS/2024-10-25/...
└── analysis_plots/                     # Generated plots

NEXT STEPS FOR YOUR REAL EXPERIMENTS:
--------------------------------------
1. Replace simulations with your actual measurement functions:
   
   # Instead of:
   times, wls, abs = simulate_uvvis_time_series_data(...)
   
   # Use:
   times, wls, abs = measure_uvvis_from_droplet_reactor(...)

2. Add your instrument control code:
   
   def measure_uvvis_from_droplet_reactor(time_points):
       times, wls, abs = [], [], []
       for t in time_points:
           # Your instrument control code here
           spectrum = your_instrument.measure()
           times.extend([t] * len(spectrum.wavelengths))
           wls.extend(spectrum.wavelengths)
           abs.extend(spectrum.absorbance)
       return times, wls, abs

3. Integrate into your workflow:
   
   # Before experiment
   org = DataOrganizer("./experiments")
   run = org.create_run(metadata)
   
   # During experiment
   data = measure_and_save()
   run.uvvis.link_data(data_files, time_points=[...])
   org.save()
   
   # After experiment
   run.uvvis.plot(plot_type="heatmap")
   analyze_results(run)

4. Build your analysis pipeline:
   
   for run_key in org.list_runs():
       run = org.get_run(...)
       peak_shift = extract_peak_shift(run)
       particle_size = estimate_size_from_saxs(run)
       crystallinity = analyze_waxs_peaks(run)
       
       results[run_key] = {{
           'peak_shift': peak_shift,
           'size': particle_size,
           'crystallinity': crystallinity
       }}

TIPS:
-----
• Keep the organizer in memory during experiments
• Save frequently with org.save()
• Validate after linking data
• Use tags for easy searching
• Add detailed metadata for instruments

Happy experimenting! 🔬✨
""")

print("=" * 70)
print("  ✨ INTEGRATION COMPLETE! ✨")
print("=" * 70)