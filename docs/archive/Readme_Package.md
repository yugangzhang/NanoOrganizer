# NanoOrganizer - Modular Package Structure

## 📁 Package Organization

```
NanoOrganizer/                      # Main package directory
├── __init__.py                     # Package initialization & public API
├── metadata.py                     # Metadata classes
├── data_links.py                   # Data link class
├── data_accessors.py               # Data accessor classes
├── run.py                          # Run class
├── organizer.py                    # DataOrganizer class
└── utils.py                        # Utility functions
```

## 📦 Module Descriptions

### `__init__.py` (Public API)
**What it does:** Package initialization and exports
- Imports all public classes
- Defines `__all__` for clean imports
- Package version and info

### `metadata.py` (~80 lines)
**What it does:** Metadata classes for experimental runs
- `ChemicalSpec` - Chemical specifications
- `ReactionParams` - Reaction conditions
- `RunMetadata` - Complete run metadata

**Key features:**
- Dataclass-based for clean structure
- `to_dict()` and `from_dict()` for JSON serialization
- Type hints for clarity

### `data_links.py` (~40 lines)
**What it does:** References to data files (no actual data)
- `DataLink` - Stores file paths and metadata

**Key features:**
- Lightweight (just paths, no data)
- Validation (check if files exist)
- JSON serialization

### `data_accessors.py` (~450 lines)
**What it does:** Data loading and visualization
- `UVVisData` - UV-Vis spectroscopy
- `SAXSData` - SAXS scattering
- `WAXSData` - WAXS diffraction
- `ImageData` - Microscopy images

**Key features:**
- Lazy loading (load on demand)
- Built-in validation
- Plotting methods (spectrum, kinetics, heatmap)
- Error handling

### `run.py` (~120 lines)
**What it does:** Single experimental run
- `Run` - Contains metadata + data accessors

**Key features:**
- Aggregates all data types
- JSON serialization
- Validation across all data

### `organizer.py` (~170 lines)
**What it does:** Main organizer class
- `DataOrganizer` - Manages all runs

**Key features:**
- Create/get/list runs
- Save/load metadata to/from JSON
- Validate all data
- Index management

### `utils.py` (~80 lines)
**What it does:** Utility functions
- `save_time_series_to_csv()` - Save data to CSV files

**Key features:**
- Time-series data handling
- Automatic file naming
- Flexible column naming

## 🎯 Why This Structure?

### Modularity
- Each file has a single, clear purpose
- Easy to find and modify specific functionality
- Can import individual components

### Maintainability
- Small files (~40-450 lines vs 1000+ lines)
- Clear separation of concerns
- Easy to debug specific issues

### Reusability
- Can use individual components independently
- Easy to extend with new data types
- Clean API through `__init__.py`

### Testability
- Each module can be tested independently
- Clear dependencies between modules
- Easy to mock components

## 📖 Usage

### Import Everything (Recommended)
```python
from NanoOrganizer import (
    DataOrganizer, 
    RunMetadata, 
    ReactionParams, 
    ChemicalSpec,
    save_time_series_to_csv
)
```

### Import Package
```python
import NanoOrganizer as no

org = no.DataOrganizer("./MyProject")
metadata = no.RunMetadata(...)
```

### Import Specific Modules (Advanced)
```python
from NanoOrganizer.metadata import ChemicalSpec
from NanoOrganizer.organizer import DataOrganizer
from NanoOrganizer.data_accessors import UVVisData
```

## 🔄 Module Dependencies

```
organizer.py
    └── run.py
        ├── metadata.py
        └── data_accessors.py
            └── data_links.py

utils.py (standalone)
```

**No circular dependencies!** Clean, hierarchical structure.

## 🎨 Extending the Package

### Adding a New Data Type

1. **Add to `data_accessors.py`:**
```python
class FTIRData:
    def __init__(self, run_id: str):
        self.link = DataLink(data_type="ftir")
    
    def link_data(self, csv_files, ...):
        # Implementation
        pass
    
    def load(self):
        # Implementation
        pass
    
    def plot(self, ...):
        # Implementation
        pass
```

2. **Add to `run.py`:**
```python
from .data_accessors import FTIRData

class Run:
    def __init__(self, ...):
        ...
        self.ftir = FTIRData(metadata.run_id)
```

3. **Update `__init__.py`:**
```python
from .data_accessors import FTIRData

__all__ = [
    ...
    'FTIRData',
]
```

### Adding New Metadata Fields

Edit `metadata.py`:
```python
@dataclass
class ExtendedReactionParams(ReactionParams):
    atmosphere: str = "air"
    stirring_speed_rpm: float = 0.0
```

### Adding Utility Functions

Edit `utils.py`:
```python
def new_utility_function(...):
    """Your new function."""
    pass
```

## 🐛 Debugging Guide

### Issue with Metadata?
→ Check `metadata.py` (80 lines)

### Issue with Data Loading?
→ Check `data_accessors.py` (450 lines)

### Issue with File Links?
→ Check `data_links.py` (40 lines)

### Issue with Saving/Loading?
→ Check `organizer.py` (170 lines)

### Issue with Run Creation?
→ Check `run.py` (120 lines)

### Issue with CSV Saving?
→ Check `utils.py` (80 lines)

## 📊 Line Count Summary

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | ~90 | Package API |
| `metadata.py` | ~80 | Metadata classes |
| `data_links.py` | ~40 | File references |
| `data_accessors.py` | ~450 | Data loading & viz |
| `run.py` | ~120 | Run class |
| `organizer.py` | ~170 | Main organizer |
| `utils.py` | ~80 | Utilities |
| **Total** | **~1030** | **Well-organized!** |

Compare to single file: 1000 lines in one file → 7 focused files!

## ✅ Benefits

1. **Easy to Read** - Each file < 500 lines
2. **Easy to Debug** - Know exactly where to look
3. **Easy to Extend** - Clear structure for additions
4. **Easy to Test** - Test each module independently
5. **Easy to Reuse** - Import what you need
6. **Easy to Maintain** - Clear responsibilities

## 🚀 Getting Started

No changes to your usage code! The API is the same:

```python
from NanoOrganizer import DataOrganizer, RunMetadata

# Everything works exactly the same!
org = DataOrganizer("./MyProject")
run = org.create_run(metadata)
```

The modular structure is transparent to users but makes development much easier!

---

**Old:** 1 file, 1000 lines, hard to navigate  
**New:** 7 files, clear organization, easy to debug  

Same functionality, better organization! 🎉