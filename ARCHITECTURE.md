# NanoOrganizer: Architecture & Design

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INTERFACE                         │
│  (Your Python Scripts, Jupyter Notebooks, etc.)            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ├─── Create Runs
                         ├─── Link Data
                         ├─── Load & Visualize
                         └─── Analyze
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   DataOrganizer                             │
│  - Manages all runs                                         │
│  - Saves/loads JSON metadata                                │
│  - Validation & integrity checks                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ Contains multiple
                         │
            ┌────────────▼────────────┐
            │         Run             │
            │  - Metadata             │
            │  - Data Accessors       │
            └────────────┬────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
    ┌───▼───┐      ┌────▼────┐      ┌───▼───┐
    │UVVis  │      │  SAXS   │      │ WAXS  │
    │Data   │      │  Data   │      │ Data  │
    └───┬───┘      └────┬────┘      └───┬───┘
        │               │               │
        └───────────────┼───────────────┘
                        │
            Links to actual files
                        │
        ┌───────────────▼───────────────┐
        │     File System               │
        │  - CSV files                  │
        │  - Image files                │
        │  - Organized by you           │
        └───────────────────────────────┘
```

---

## 📦 Module Structure

### Core Classes

```
NanoOrganizer.py
│
├── Metadata Classes
│   ├── ChemicalSpec      - Chemical specifications
│   ├── ReactionParams    - Reaction conditions
│   └── RunMetadata       - Experiment metadata
│
├── Data Link Classes
│   └── DataLink          - Links to data files (no actual data)
│
├── Data Accessor Classes (Lazy Loading)
│   ├── UVVisData        - UV-Vis spectroscopy
│   ├── SAXSData         - Small-angle X-ray scattering
│   ├── WAXSData         - Wide-angle X-ray diffraction
│   └── ImageData        - SEM/TEM images
│
├── Run Class
│   └── Run              - Contains metadata + all data accessors
│
└── Main Organizer
    └── DataOrganizer    - Manages everything
```

---

## 🔄 Data Flow

### Creating a Run

```
1. User creates RunMetadata
   ↓
2. DataOrganizer.create_run(metadata)
   ↓
3. Creates Run object
   ↓
4. Run contains empty data accessors
   ↓
5. User links data via: run.uvvis.link_data(csv_files)
   ↓
6. Links stored (file paths only, no data loaded)
   ↓
7. DataOrganizer.save() → Writes JSON metadata
```

### Loading Data

```
1. DataOrganizer.load(path)
   ↓
2. Reads JSON metadata files
   ↓
3. Creates Run objects with data links
   ↓
4. User calls: run.uvvis.load()
   ↓
5. Data accessor reads CSV files
   ↓
6. Returns numpy arrays
   ↓
7. Data cached for subsequent access
```

### Visualization

```
1. User calls: run.uvvis.plot()
   ↓
2. Data accessor loads data if not already loaded
   ↓
3. Generates matplotlib plot
   ↓
4. User can customize with additional matplotlib commands
```

---

## 💾 Storage Architecture

### Metadata Storage (JSON)

```
project_root/
└── .metadata/
    ├── index.json                      # Master index
    │   {
    │     "runs": ["Project/Exp/Run1", ...],
    │     "last_updated": "2024-10-20T12:00:00"
    │   }
    │
    └── Project_Experiment_Run.json     # Individual run metadata
        {
          "metadata": {
            "project": "...",
            "reaction": {...},
            "tags": [...]
          },
          "data": {
            "uvvis": {
              "file_paths": ["/path/to/file1.csv", ...],
              "time_points": [0, 30, 60, ...],
              "metadata": {"instrument": "..."}
            }
          }
        }
```

### Data Storage (Your Files)

```
project_root/
└── Project_Cu2O/                   # Your organization
    ├── UV_Vis/
    │   └── 2024-10-20/
    │       └── Cu2O_V1_LowTemp/
    │           ├── uvvis_001.csv   # wavelength,absorbance
    │           ├── uvvis_002.csv
    │           └── ...
    ├── SAXS/
    │   └── 2024-10-20/
    │       └── Cu2O_V1_LowTemp/
    │           ├── saxs_001.csv    # q,intensity
    │           └── ...
    └── SEM/
        └── images/
            ├── sem_001.png
            └── ...
```

**Key Point**: Your data files stay exactly where you put them!

---

## 🎯 Design Principles

### 1. Separation of Concerns
- **Metadata**: Lightweight JSON files (easy to backup, version control)
- **Data**: Your files stay where they are (any structure)
- **Logic**: All in Python classes (easy to extend)

### 2. Lazy Loading
```python
org = DataOrganizer.load(path)  # Fast! Only loads metadata
run = org.get_run(...)          # Fast! Just metadata
data = run.uvvis.load()         # Slow! Actually reads CSV files
```

Benefits:
- Fast startup
- Memory efficient
- Only load what you need

### 3. Flexible File Organization
```python
# We don't care about your structure, just give us paths:
run.uvvis.link_data([
    "/any/path/you/want/uvvis_001.csv",
    "/completely/different/path/uvvis_002.csv"
])
```

### 4. Validation at Every Step
```python
# Check if files exist
run.uvvis.validate()      # Returns True/False
org.validate_all()        # Checks all runs

# Automatic warnings if files missing
# No crashes, just helpful messages
```

### 5. Extensibility
Adding a new data type is easy:

```python
class NewDataType:
    def __init__(self, run_id):
        self.link = DataLink(data_type="new_type")
    
    def link_data(self, files, ...):
        # Store file paths
        pass
    
    def load(self):
        # Read files, return data
        pass
    
    def plot(self, ...):
        # Visualize
        pass

# Add to Run class:
class Run:
    def __init__(self, ...):
        self.new_data = NewDataType(...)
```

---

## 🔐 Key Design Decisions

### Why JSON for Metadata?
- ✅ Human-readable
- ✅ Easy to edit manually if needed
- ✅ Version control friendly (text files)
- ✅ Widely supported (any language can read)
- ✅ Lightweight

vs. Database (SQLite):
- ❌ Binary format
- ❌ Harder to inspect/edit
- ❌ Less portable
- ✅ Better for complex queries (overkill for our use)

### Why Lazy Loading?
```python
# Without lazy loading:
org = DataOrganizer.load()  # Loads ALL data from ALL runs!
# 😱 Could be GBs of data, takes minutes

# With lazy loading:
org = DataOrganizer.load()  # Just metadata (<1 MB)
# ✅ Fast! <1 second
run = org.get_run(...)      # Still just metadata
data = run.uvvis.load()     # NOW load this one dataset
# ✅ Only load what you need
```

### Why Not Copy Data Files?
```python
# Bad approach:
# Copy all data into database folder
# - Duplicates data (wastes space)
# - Hard to update if source files change
# - Loses original organization

# Our approach:
# Store links to data
# - No duplication
# - Files stay where you put them
# - Any organization works
```

---

## 🧩 Class Interactions

### Creating & Saving
```
User
  │
  ├─► DataOrganizer.create_run(metadata)
  │     └─► Creates Run object
  │           ├─► UVVisData (empty)
  │           ├─► SAXSData (empty)
  │           └─► WAXSData (empty)
  │
  ├─► Run.uvvis.link_data(files)
  │     └─► Stores file paths in DataLink
  │
  └─► DataOrganizer.save()
        └─► Run.to_dict()
              ├─► RunMetadata.to_dict()
              └─► For each data type:
                    └─► DataLink.to_dict()
        └─► Write JSON files
```

### Loading & Using
```
User
  │
  ├─► DataOrganizer.load(path)
  │     ├─► Read index.json
  │     └─► For each run:
  │           ├─► Read JSON file
  │           └─► Run.from_dict()
  │                 ├─► RunMetadata.from_dict()
  │                 └─► DataLink.from_dict()
  │
  ├─► DataOrganizer.get_run()
  │     └─► Returns Run object
  │
  ├─► Run.uvvis.load()
  │     ├─► For each file in DataLink:
  │     │     └─► np.loadtxt(file)
  │     └─► Returns dict of arrays
  │
  └─► Run.uvvis.plot()
        ├─► Calls .load() if needed
        └─► Creates matplotlib plot
```

---

## 📊 Memory Management

```python
# Scenario 1: Load 100 runs, plot 1
org = DataOrganizer.load()              # ~1 MB in memory (metadata)
run = org.get_run(...)                   # ~1 KB (one run's metadata)
run.uvvis.plot()                         # ~10 MB (loads just this data)
# Total memory: ~11 MB

# Scenario 2: If data was always loaded
org = DataOrganizer.load()              # Would load ALL data!
# Total memory: ~10 GB (100 runs × 100 MB each)
# 😱 Crash or very slow
```

### Memory Optimization Tips

```python
# Load data
data = run.uvvis.load()
# ... do analysis ...

# Clear from memory if needed
run.uvvis._loaded_data = None

# Or force reload
data = run.uvvis.load(force_reload=True)
```

---

## 🎨 Extensibility Examples

### Adding Custom Analysis
```python
def extract_peak_positions(run):
    """Custom analysis function."""
    data = run.uvvis.load()
    # ... analysis ...
    return peaks

# Use it
peaks = extract_peak_positions(run)
```

### Adding New Metadata Fields
```python
@dataclass
class ExtendedReactionParams(ReactionParams):
    """Add new fields to reaction params."""
    atmosphere: str = "air"
    vessel_type: str = "round_bottom"
    stirring_speed_rpm: float = 0.0

# Use it
metadata = RunMetadata(
    ...,
    reaction=ExtendedReactionParams(
        ...,
        atmosphere="nitrogen",
        stirring_speed_rpm=300
    )
)
```

### Adding New Data Type
```python
class FTIRData:
    """FTIR spectroscopy data."""
    def __init__(self, run_id):
        self.link = DataLink(data_type="ftir")
    
    def link_data(self, csv_files, time_points=None, metadata=None):
        # Same pattern as UV-Vis
        pass
    
    def load(self):
        # Read FTIR CSV files
        pass
    
    def plot(self, plot_type="spectrum", ...):
        # Visualize
        pass

# Add to Run class
class Run:
    def __init__(self, ...):
        ...
        self.ftir = FTIRData(metadata.run_id)
```

---

## 🔍 Performance Characteristics

| Operation | Time | Memory | Notes |
|-----------|------|--------|-------|
| Create organizer | <1 ms | <1 KB | Just creates object |
| Create run | <1 ms | ~10 KB | Stores metadata |
| Link 100 CSV files | <1 ms | ~100 KB | Just stores paths |
| Save metadata | ~10 ms | 0 | Writes JSON |
| Load organizer | ~50 ms | ~1 MB | Reads all metadata |
| Load 1 run data | ~100 ms | ~10 MB | Reads CSV files |
| Plot spectrum | ~200 ms | 0 | matplotlib rendering |
| Validate 100 runs | ~100 ms | 0 | Just checks file.exists() |

---

## 🎯 Use Cases

### 1. High-Throughput Screening
```python
# Create 100 runs with different conditions
for temp in temperatures:
    for conc in concentrations:
        metadata = RunMetadata(
            run_id=f"T{temp}_C{conc}",
            reaction=ReactionParams(temperature_C=temp, ...)
        )
        run = org.create_run(metadata)
        # ... measure and link data ...

org.save()

# Later: find best conditions
for run_key in org.list_runs():
    run = org.get_run(...)
    peak_shift = analyze_growth_rate(run)
    print(f"{run_key}: {peak_shift}")
```

### 2. Time-Resolved Studies
```python
# Multiple measurements over time
for t in [0, 30, 60, 120, 180, 300]:
    # Measure UV-Vis at time t
    spectrum = measure_uvvis()
    # Save and link
    ...

# Analyze full evolution
run.uvvis.plot(plot_type="heatmap")
```

### 3. Multi-Technique Characterization
```python
# Combine data from multiple instruments
run.uvvis.link_data(uvvis_files)
run.saxs.link_data(saxs_files)
run.waxs.link_data(waxs_files)
run.tem.link_data(tem_images)

# Correlate results
uvvis_data = run.uvvis.load()
saxs_data = run.saxs.load()
correlate_size_and_absorbance(uvvis_data, saxs_data)
```

---

## 📝 Summary

**Design Philosophy**: 
- Simple things should be simple
- Complex things should be possible
- Don't repeat yourself
- Fail gracefully

**Key Innovations**:
- Metadata separate from data
- Lazy loading for performance
- Flexible file organization
- Built-in validation
- Easy to extend

**Production-Ready**:
- ✅ Clean, documented code
- ✅ Error handling
- ✅ Validation checks
- ✅ Memory efficient
- ✅ Student-friendly API

Enjoy! 🚀