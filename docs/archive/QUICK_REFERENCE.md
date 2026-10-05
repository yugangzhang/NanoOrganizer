# NanoOrganizer Quick Reference

## 🚀 One-Page Cheat Sheet for Students

### Setup (Once)
```python
from nano_organizer import (
    DataOrganizer, RunMetadata, ReactionParams, 
    ChemicalSpec, VisualizationHelper
)

org = DataOrganizer("my_lab_data")
```

### Create New Run
```python
meta = RunMetadata(
    project="Project_Cu2O",              # Material
    experiment="2024-10-25",              # Date
    run_id="Sample_001",                  # Unique ID
    reaction=ReactionParams(
        chemicals=[
            ChemicalSpec(name="CuCl2", concentration=0.1, 
                        concentration_unit="M", volume_uL=500)
        ],
        temperature_C=60.0,
        conductor="Your Name"
    ),
    tags=["optimization", "first_try"]
)
run = org.create_run(meta)
```

### Add Data
```python
# UV-Vis
run.uvvis().add(times, wavelengths, absorbance)

# SAXS
run.saxs().add(q_values, intensities)

# WAXS
run.waxs().add(two_theta, intensities)

# Microscopy
run.sem().add(["image1.png", "image2.png"])
run.tem().add(["tem1.png"])
```

### Load Existing Run
```python
run = org.load_run("Project_Cu2O", "2024-10-25", "Sample_001")
```

### Visualize
```python
# Single plot
run.uvvis().plot()
run.saxs().plot(loglog=True)
run.sem().plot(idx=0)

# Compare runs
VisualizationHelper.compare_runs([run1, run2], modality="uvvis")

# Multi-modal
VisualizationHelper.multi_modality_plot(run, modalities=["uvvis", "saxs", "waxs"])
```

### Quality Check
```python
# Validate data
issues = run.validate_all()
if any(issues.values()):
    print("⚠️ Data issues found!")
    
# Get statistics
stats = run.get_all_stats()
print(stats['uvvis']['absorbance_stats'])
```

### Query Database
```python
# List all runs
all_runs = org.list_runs()

# Filter by project
runs = org.list_runs(project="Project_Cu2O")

# Filter by tags
runs = org.list_runs(tags=["optimization"])

# Search
runs = org.search("high temperature")

# Summary
summary = org.get_summary()
```

### Export Data
```python
# Single run
run.export_all("exports/run1")

# Batch export
org.batch_export("exports", project="Project_Cu2O")
```

### Import Existing Data
```python
summary = org.import_external_tree(
    project_name="Project_Cu2O",
    external_project_root="/path/to/Project_Cu2O",
    link_only=True
)
```

## 📋 Daily Workflow

### Morning (Start Experiment)
```python
meta = RunMetadata(...)
run = org.create_run(meta)
```

### During Experiment (Add Data)
```python
run.uvvis().add(...)
run.saxs().add(...)
```

### Evening (Check & Export)
```python
issues = run.validate_all()
run.uvvis().plot()
run.export_all("daily_exports")
```

## 🔍 Common Tasks

### Find Last Week's Runs
```python
from datetime import datetime, timedelta
week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
recent = [r for r in org.list_runs() if r['experiment'] >= week_ago]
```

### Find Best Results
```python
optimized = org.list_runs(tags=["successful", "optimized"])
```

### Compare Temperature Series
```python
low_temp = org.load_run("Project_Cu2O", "2024-10-20", "LowTemp")
high_temp = org.load_run("Project_Cu2O", "2024-10-21", "HighTemp")
VisualizationHelper.compare_runs([low_temp, high_temp], modality="uvvis")
```

### Batch Validate Project
```python
results = org.batch_validate(project="Project_Cu2O")
for run_id, issues in results.items():
    if any(issues.values()):
        print(f"⚠️ {run_id} has issues")
```

## 🎨 Plotting Tips

### Customize Plot
```python
run.uvvis().plot(
    kind="line",              # or "scatter"
    figsize=(10, 6),
    title="My Experiment",
    grid=True,
    linewidth=2
)
```

### Save Plot
```python
import matplotlib.pyplot as plt
run.uvvis().plot()
plt.savefig("my_plot.png", dpi=300, bbox_inches='tight')
```

### Subplots
```python
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
run.uvvis().plot(ax=axes[0])
run.saxs().plot(ax=axes[1], loglog=True)
run.waxs().plot(ax=axes[2])
plt.tight_layout()
plt.show()
```

## ⚠️ Troubleshooting

### "No data found"
```python
# Check if data exists
data = run.uvvis().load()
print(data)
```

### Validation errors
```python
# See specific issues
issues = run.validate_all()
for modality, problems in issues.items():
    if problems:
        for p in problems:
            print(f"{modality}: {p}")
```

### Plot not showing
```python
# Make sure matplotlib is imported
import matplotlib.pyplot as plt
run.uvvis().plot()
plt.show()  # Sometimes needed
```

## 📊 Data Formats

### UV-Vis CSV Format
```
time_s,wavelength_nm,absorbance
0,200,0.1
1,201,0.12
```

### SAXS CSV Format
```
q_invA,intensity,sigma
0.01,1000,10
0.02,950,9.5
```

### WAXS CSV Format
```
two_theta_deg,intensity
10,100
11,120
```

## 💡 Pro Tips

1. **Use descriptive run_id**: `Cu2O_V1_60C_pH7` better than `Sample1`
2. **Tag everything**: Makes searching easier later
3. **Validate immediately**: Catch errors while you remember the details
4. **Export regularly**: Have backup in standard format
5. **Document in notes**: Future you will thank present you
6. **Use consistent naming**: Pick a scheme and stick to it

## 🎯 Keyboard Shortcuts (in Jupyter)

```python
# Quick access pattern
o = org  # Short alias
r = o.load_run("Project", "Date", "ID")
r.uvvis().plot()  # Quick visualization
```

## 📚 Where to Learn More

- **README.md**: Full documentation
- **demo.py**: Working examples
- **IMPROVEMENTS.md**: What's new
- **Your lab notebook**: Document your specific workflows!

---

**Print this and keep it next to your computer! 📄**