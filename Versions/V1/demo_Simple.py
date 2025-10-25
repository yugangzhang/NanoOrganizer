#!/usr/bin/env python3
"""
Comprehensive Demo for NanoOrganizer

This demo shows:
1. Creating simulated data for all modalities
2. Setting up the database
3. Adding multiple runs
4. Loading and plotting data
5. Data validation
6. Batch operations
7. Advanced queries and comparisons
8. Exporting data
"""

import numpy as np
from pathlib import Path
import shutil
from NanoOrganizer import (
    DataOrganizer, RunMetadata, ReactionParams, ChemicalSpec,
    VisualizationHelper
)

# Create demo workspace
DEMO_ROOT = Path("/home/yuzhang/Repos/NanoOrganizer/Demo/")
EXTERNAL_DATA = Path("/home/yuzhang/Repos/NanoOrganizer/Data/")

# Clean up if exists
if DEMO_ROOT.exists():
    shutil.rmtree(DEMO_ROOT)
if EXTERNAL_DATA.exists():
    shutil.rmtree(EXTERNAL_DATA)

print("=" * 80)
print("NANOORGANIZER COMPREHENSIVE DEMO")
print("=" * 80)

# =============================================================================
# PART 1: DATA SIMULATION
# =============================================================================
print("\n" + "=" * 80)
print("PART 1: SIMULATING EXPERIMENTAL DATA")
print("=" * 80)


# =============================================================================
# PART 2: CREATE DATABASE AND ADD RUNS
# =============================================================================
print("\n" + "=" * 80)
print("PART 2: CREATING DATABASE AND ADDING RUNS")
print("=" * 80)

# Initialize organizer
org = DataOrganizer(DEMO_ROOT)
print(f"\n✓ Initialized DataOrganizer at: {DEMO_ROOT}")

# Create first run - Cu2O synthesis
print("\n--- Creating Run 1: Cu2O Synthesis (Low Temperature) ---")
run1_meta = RunMetadata(
    project="Project_Cu2O",
    experiment="2024-10-20",
    run_id="Cu2O_V1_LowTemp",
    sample_id="Sample_001",
    reaction=ReactionParams(
        chemicals=[
            ChemicalSpec(name="CuCl2", concentration=0.1, concentration_unit="M", volume_uL=500),
            ChemicalSpec(name="NaBH4", concentration=0.05, concentration_unit="M", volume_uL=200),
            ChemicalSpec(name="PVP", concentration=1.0, concentration_unit="g/L", volume_uL=100),
        ],
        temperature_C=60.0,
        stir_time_s=300,
        reaction_time_s=1800,
        pH=7.5,
        solvent="Water",
        conductor="Dr. Smith",
        description="Low temperature Cu2O synthesis with PVP stabilization"
    ),
    notes="First attempt at low temperature synthesis",
    tags=["Cu2O", "low_temp", "optimization"]
)

run1 = org.create_run(run1_meta)
print(f"✓ Created run: {run1.metadata.run_id}")

# Add UV-Vis data
print("\n  Adding UV-Vis data...")
times, wavelengths, absorbance = simulate_uvvis_data(peak_center=520)
run1.uvvis().add(times, wavelengths, absorbance, meta={"instrument": "Agilent 8453"})
print(f"    ✓ Added {len(wavelengths)} UV-Vis data points")

# Add SAXS data
print("  Adding SAXS data...")
q, intensity = simulate_saxs_data(size_nm=5)
run1.saxs().add(q, intensity, meta={"beamline": "CHESS", "energy_keV": 12.0})
print(f"    ✓ Added {len(q)} SAXS data points")

# Add WAXS data
print("  Adding WAXS data...")
two_theta, waxs_intensity = simulate_waxs_data()
run1.waxs().add(two_theta, waxs_intensity, meta={"instrument": "Bruker D8"})
print(f"    ✓ Added {len(two_theta)} WAXS data points")

# Add images
print("  Adding microscopy images...")
try:
    from PIL import Image
    
    # Create SEM images
    sem_dir = Path("/tmp/sem_images")
    sem_dir.mkdir(exist_ok=True)
    sem_images = []
    for i in range(3):
        img_path = sem_dir / f"sem_run1_{i+1}.png"
        if create_fake_image(img_path, pattern="sem"):
            sem_images.append(img_path)
    
    if sem_images:
        run1.sem().add(sem_images, meta={"magnification": "50kX", "voltage_kV": 5.0}, copy=True)
        print(f"    ✓ Added {len(sem_images)} SEM images")
    
    # Create TEM images
    tem_dir = Path("/tmp/tem_images")
    tem_dir.mkdir(exist_ok=True)
    tem_images = []
    for i in range(2):
        img_path = tem_dir / f"tem_run1_{i+1}.png"
        if create_fake_image(img_path, pattern="tem"):
            tem_images.append(img_path)
    
    if tem_images:
        run1.tem().add(tem_images, meta={"magnification": "200kX", "voltage_kV": 200.0}, copy=True)
        print(f"    ✓ Added {len(tem_images)} TEM images")
        
except ImportError:
    print("    ⚠ Skipping images (PIL not available)")

# Create second run - Cu2O synthesis at higher temperature
print("\n--- Creating Run 2: Cu2O Synthesis (High Temperature) ---")
run2_meta = RunMetadata(
    project="Project_Cu2O",
    experiment="2024-10-21",
    run_id="Cu2O_V2_HighTemp",
    sample_id="Sample_002",
    reaction=ReactionParams(
        chemicals=[
            ChemicalSpec(name="CuCl2", concentration=0.1, concentration_unit="M", volume_uL=500),
            ChemicalSpec(name="NaBH4", concentration=0.05, concentration_unit="M", volume_uL=200),
            ChemicalSpec(name="PVP", concentration=1.0, concentration_unit="g/L", volume_uL=100),
        ],
        temperature_C=85.0,
        stir_time_s=300,
        reaction_time_s=1800,
        pH=7.5,
        solvent="Water",
        conductor="Dr. Smith",
        description="High temperature Cu2O synthesis for comparison"
    ),
    notes="Higher temperature should give larger particles",
    tags=["Cu2O", "high_temp", "optimization"]
)

run2 = org.create_run(run2_meta)
print(f"✓ Created run: {run2.metadata.run_id}")

# Add data with different parameters (larger particles)
print("\n  Adding characterization data...")
times2, wavelengths2, absorbance2 = simulate_uvvis_data(peak_center=540, peak_width=60)
run2.uvvis().add(times2, wavelengths2, absorbance2, meta={"instrument": "Agilent 8453"})

q2, intensity2 = simulate_saxs_data(size_nm=8)  # Larger particles
run2.saxs().add(q2, intensity2, meta={"beamline": "CHESS", "energy_keV": 12.0})

two_theta2, waxs_intensity2 = simulate_waxs_data(peaks=[(30, 120), (35, 100), (62, 80)])
run2.waxs().add(two_theta2, waxs_intensity2, meta={"instrument": "Bruker D8"})

print("  ✓ Added UV-Vis, SAXS, and WAXS data")

# Create third run - Different material (Au nanoparticles)
print("\n--- Creating Run 3: Au Nanoparticle Synthesis ---")
run3_meta = RunMetadata(
    project="Project_Au",
    experiment="2024-10-22",
    run_id="Au_V1_Citrate",
    sample_id="Sample_101",
    reaction=ReactionParams(
        chemicals=[
            ChemicalSpec(name="HAuCl4", concentration=0.01, concentration_unit="M", volume_uL=1000),
            ChemicalSpec(name="Sodium_citrate", concentration=0.1, concentration_unit="M", volume_uL=500),
        ],
        temperature_C=95.0,
        stir_time_s=600,
        reaction_time_s=900,
        pH=6.8,
        solvent="Water",
        conductor="Dr. Johnson",
        description="Classical citrate reduction for Au NPs"
    ),
    notes="Turkevich method for gold nanoparticles",
    tags=["Au", "citrate", "spherical"]
)

run3 = org.create_run(run3_meta)
print(f"✓ Created run: {run3.metadata.run_id}")

# Add Au NP data (different UV-Vis peak)
print("\n  Adding characterization data...")
times3, wavelengths3, absorbance3 = simulate_uvvis_data(peak_center=520, peak_width=40)
run3.uvvis().add(times3, wavelengths3, absorbance3, meta={"instrument": "Agilent 8453"})

q3, intensity3 = simulate_saxs_data(size_nm=15)  # Larger Au particles
run3.saxs().add(q3, intensity3, meta={"beamline": "CHESS", "energy_keV": 12.0})

print("  ✓ Added UV-Vis and SAXS data")

print("\n" + "=" * 80)
print(f"✓ Created {len(org.list_runs())} runs total")
print("=" * 80)

# =============================================================================
# PART 3: QUERYING AND LOADING DATA
# =============================================================================
print("\n" + "=" * 80)
print("PART 3: QUERYING AND LOADING DATA")
print("=" * 80)

# List all runs
print("\n--- All Runs ---")
all_runs = org.list_runs()
for run in all_runs:
    print(f"  • {run['project']} / {run['experiment']} / {run['run_id']}")

# Filter by project
print("\n--- Runs in Project_Cu2O ---")
cu2o_runs = org.list_runs(project="Project_Cu2O")
for run in cu2o_runs:
    print(f"  • {run['experiment']} / {run['run_id']}")

# Filter by tags
print("\n--- Runs with 'optimization' tag ---")
opt_runs = org.list_runs(tags=["optimization"])
for run in opt_runs:
    print(f"  • {run['project']} / {run['run_id']}")

# Search by keyword
print("\n--- Search for 'high' ---")
search_results = org.search("high")
for run in search_results:
    print(f"  • {run['project']} / {run['run_id']}")

# Load specific run
print("\n--- Loading Run 1 ---")
loaded_run1 = org.load_run("Project_Cu2O", "2024-10-20", "Cu2O_V1_LowTemp")
print(f"✓ Loaded: {loaded_run1.metadata.run_id}")
print(f"  Temperature: {loaded_run1.metadata.reaction.temperature_C}°C")
print(f"  Conductor: {loaded_run1.metadata.reaction.conductor}")
print(f"  Chemicals: {[c.name for c in loaded_run1.metadata.reaction.chemicals]}")

# Get summary
print("\n--- Database Summary ---")
summary = org.get_summary()
print(f"  Total runs: {summary['total_runs']}")
print(f"  Projects: {summary['projects']}")
print(f"  Experiments: {summary['experiments']}")
print(f"  Data by modality:")
for modality, count in summary['by_modality'].items():
    print(f"    - {modality}: {count} runs")

# =============================================================================
# PART 4: DATA VALIDATION
# =============================================================================
print("\n" + "=" * 80)
print("PART 4: DATA VALIDATION")
print("=" * 80)

print("\n--- Validating Run 1 ---")
validation_results = loaded_run1.validate_all()
for modality, issues in validation_results.items():
    if issues:
        print(f"  {modality}: {len(issues)} issues found")
        for issue in issues[:3]:  # Show first 3
            print(f"    - {issue}")
    else:
        print(f"  ✓ {modality}: No issues")

print("\n--- Batch Validation (Project_Cu2O) ---")
batch_val = org.batch_validate(project="Project_Cu2O")
for run_key, results in batch_val.items():
    has_issues = any(len(v) > 0 for v in results.values() if isinstance(v, list))
    status = "⚠" if has_issues else "✓"
    print(f"  {status} {run_key}")

# =============================================================================
# PART 5: STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("PART 5: STATISTICAL ANALYSIS")
print("=" * 80)

print("\n--- Statistics for Run 1 ---")
stats = loaded_run1.get_all_stats()
for modality, stat_dict in stats.items():
    if stat_dict:
        print(f"\n  {modality.upper()}:")
        for key, value in stat_dict.items():
            if isinstance(value, dict):
                print(f"    {key}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        print(f"      {k}: {v:.3f}")
                    else:
                        print(f"      {k}: {v}")
            else:
                print(f"    {key}: {value}")

# =============================================================================
# PART 6: PLOTTING DATA
# =============================================================================
print("\n" + "=" * 80)
print("PART 6: VISUALIZING DATA")
print("=" * 80)

try:
    import matplotlib.pyplot as plt
    
    # Single modality plot
    print("\n--- Plotting UV-Vis for Run 1 ---")
    fig, ax = plt.subplots(figsize=(10, 6))
    loaded_run1.uvvis().plot(ax=ax, title="Cu2O Synthesis - Low Temperature", grid=True)
    plt.savefig(DEMO_ROOT / "plot_uvvis_run1.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: plot_uvvis_run1.png")
    
    # Multi-modality plot for single run
    print("\n--- Multi-modality Plot for Run 1 ---")
    VisualizationHelper.multi_modality_plot(
        loaded_run1,
        modalities=["uvvis", "saxs", "waxs"],
        figsize=(18, 5)
    )
    plt.savefig(DEMO_ROOT / "plot_multimodal_run1.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: plot_multimodal_run1.png")
    
    # Compare two runs
    print("\n--- Comparing Run 1 and Run 2 (UV-Vis) ---")
    loaded_run2 = org.load_run("Project_Cu2O", "2024-10-21", "Cu2O_V2_HighTemp")
    VisualizationHelper.compare_runs(
        [loaded_run1, loaded_run2],
        modality="uvvis",
        figsize=(14, 5)
    )
    plt.savefig(DEMO_ROOT / "plot_comparison_uvvis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: plot_comparison_uvvis.png")
    
    # Compare SAXS
    print("\n--- Comparing Run 1 and Run 2 (SAXS) ---")
    VisualizationHelper.compare_runs(
        [loaded_run1, loaded_run2],
        modality="saxs",
        figsize=(14, 5),
        loglog=True
    )
    plt.savefig(DEMO_ROOT / "plot_comparison_saxs.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: plot_comparison_saxs.png")
    
    print("\n✓ All plots saved to:", DEMO_ROOT)
    
except ImportError:
    print("\n⚠ matplotlib not available, skipping plots")

# =============================================================================
# PART 7: EXPORTING DATA
# =============================================================================
print("\n" + "=" * 80)
print("PART 7: EXPORTING DATA")
print("=" * 80)

# Export single run
export_dir = DEMO_ROOT / "exports"
print(f"\n--- Exporting Run 1 to CSV ---")
export_results = loaded_run1.export_all(export_dir / "run1")
for modality, success in export_results.items():
    status = "✓" if success else "✗"
    print(f"  {status} {modality}.csv")

# Batch export
print(f"\n--- Batch Export (Project_Cu2O) ---")
batch_export_results = org.batch_export(export_dir / "batch", project="Project_Cu2O")
for run_key, results in batch_export_results.items():
    print(f"\n  {run_key}:")
    if isinstance(results, dict) and "error" not in results:
        for modality, success in results.items():
            status = "✓" if success else "✗"
            print(f"    {status} {modality}")

print(f"\n✓ Exports saved to: {export_dir}")

# =============================================================================
# PART 8: EXTERNAL TREE IMPORT (SIMULATION)
# =============================================================================
print("\n" + "=" * 80)
print("PART 8: IMPORTING EXTERNAL PROJECT TREE")
print("=" * 80)

# Create a fake external project structure
print("\n--- Creating fake external project structure ---")
ext_project = EXTERNAL_DATA / "Project_Cu2O_External"

# Create directory structure
for date_folder in ["20241023", "20241024"]:
    for sample in ["Sample_A", "Sample_B"]:
        # UV-Vis
        uvvis_dir = ext_project / "UV_Vis" / date_folder / sample
        uvvis_dir.mkdir(parents=True, exist_ok=True)
        
        # Create fake CSV
        csv_path = uvvis_dir / "uvvis_data.csv"
        times_ext, wl_ext, abs_ext = simulate_uvvis_data()
        with open(csv_path, 'w') as f:
            f.write("time_s,wavelength_nm,absorbance\n")
            for t, w, a in zip(times_ext[:10], wl_ext[:10], abs_ext[:10]):  # Just 10 points for demo
                f.write(f"{t},{w},{a}\n")
        
        # SAXS
        saxs_dir = ext_project / "SAXS" / date_folder / sample
        saxs_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = saxs_dir / "saxs_data.csv"
        q_ext, I_ext = simulate_saxs_data()
        with open(csv_path, 'w') as f:
            f.write("q_invA,intensity\n")
            for q, I in zip(q_ext[:10], I_ext[:10]):
                f.write(f"{q},{I}\n")

print(f"✓ Created external structure at: {ext_project}")

# Import the external tree
print("\n--- Importing external project tree (dry run) ---")
import_summary = org.import_external_tree(
    project_name="External_Cu2O",
    external_project_root=ext_project,
    dry_run=True  # First do a dry run
)

print(f"\nDry run results:")
print(f"  Project: {import_summary['project']}")
print(f"  Runs to create: {import_summary['runs_created']}")
print(f"\n  Files found:")
for run_name, modalities in import_summary['by_run'].items():
    print(f"    {run_name}:")
    for mod, files in modalities.items():
        print(f"      {mod}: {len(files)} files")

# Actually import
print("\n--- Actually importing (linking files) ---")
import_summary = org.import_external_tree(
    project_name="External_Cu2O",
    external_project_root=ext_project,
    link_only=True,  # Don't copy, just link
    dry_run=False
)

print(f"\n✓ Created {import_summary['runs_created']} runs from external tree")

# Load one of the imported runs
print("\n--- Loading imported run ---")
imported_runs = org.list_runs(project="External_Cu2O")
if imported_runs:
    first_imported = org.get_run_by_path(imported_runs[0]['path'])
    print(f"✓ Loaded: {first_imported.metadata.run_id}")
    
    # Check UV-Vis data
    uvvis_data = first_imported.uvvis().load()
    print(f"  UV-Vis data points: {len(uvvis_data.get('data', []))}")
    print(f"  External files: {uvvis_data.get('external_files', [])[:1]}...")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("FINAL DATABASE SUMMARY")
print("=" * 80)

final_summary = org.get_summary()
print(f"\nTotal runs in database: {final_summary['total_runs']}")
print(f"Projects: {', '.join(final_summary['projects'])}")
print(f"\nData coverage:")
for modality, count in final_summary['by_modality'].items():
    print(f"  {modality:10s}: {count:3d} runs ({count/final_summary['total_runs']*100:.1f}%)")

print("\n" + "=" * 80)
print("DEMO COMPLETE!")
print("=" * 80)
print(f"\nData location: {DEMO_ROOT}")
print(f"External data: {EXTERNAL_DATA}")
print("\nKey files created:")
print(f"  • Database index: {DEMO_ROOT / 'index.json'}")
print(f"  • Run metadata: {DEMO_ROOT / 'Project_Cu2O' / '2024-10-20' / 'Cu2O_V1_LowTemp' / 'metadata.json'}")
print(f"  • Plots: {DEMO_ROOT / 'plot_*.png'}")
print(f"  • Exports: {export_dir}")

print("\n" + "=" * 80)
print("QUICK START FOR YOUR STUDENTS:")
print("=" * 80)
print("""
# 1. Import the package
from nano_organizer import DataOrganizer, RunMetadata, ReactionParams, ChemicalSpec

# 2. Initialize organizer
org = DataOrganizer("my_lab_data")

# 3. Create a run
meta = RunMetadata(
    project="MyProject",
    experiment="2024-10-25",
    run_id="Sample_001",
    reaction=ReactionParams(
        chemicals=[ChemicalSpec(name="CuCl2", concentration=0.1, volume_uL=500)],
        temperature_C=60.0
    )
)
run = org.create_run(meta)

# 4. Add data
run.uvvis().add(times, wavelengths, absorbance)
run.saxs().add(q_values, intensities)
run.sem().add([image_paths])

# 5. Load and plot
loaded = org.load_run("MyProject", "2024-10-25", "Sample_001")
loaded.uvvis().plot()

# 6. Validate
issues = loaded.validate_all()

# 7. Export
loaded.export_all("exports/my_run")
""")