#!/usr/bin/env python3
"""
NanoOrganizer — Enhanced metadata/data organizer for high-throughput 
droplet-reactor nanoparticle synthesis.

NEW FEATURES:
- Data validation and quality checks
- Batch operations and data export
- Advanced querying with filters
- Enhanced visualization (comparison plots, subplots)
- Statistical analysis utilities
- Data transformation helpers
- Better error handling and logging
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
import json
import csv
import shutil
import re
import datetime
import warnings
from collections import defaultdict

# -----------------------------
# Configuration
# -----------------------------
@dataclass(frozen=True)
class Filenames:
    """Centralized configuration for all default filenames."""
    index: str = "index.json"
    run_metadata: str = "metadata.json"
    modality_meta: str = "meta.json"
    image_manifest: str = "manifest.json"
    external_manifest: str = "external.json"
    image_dir: str = "images"
    uvvis_data: str = "uvvis.csv"
    saxs_data: str = "saxs.csv"
    waxs_data: str = "waxs.csv"

FN = Filenames()

@dataclass(frozen=True)
class DirectoryMap:
    """Maps modality names to folder names in external project tree."""
    uvvis: str = "UV_Vis"
    saxs: str = "SAXS"
    waxs: str = "WAXS"
    sem: str = "SEM"
    tem: str = "TEM"
    recipes: str = "CSV"

@dataclass(frozen=True)
class ImportSchema:
    """How to interpret the external folder structure."""
    date_regex: str = r"\d{8}"
    date_format: str = "%Y%m%d"

# -----------------------------
# Metadata models
# -----------------------------
@dataclass
class ChemicalSpec:
    name: str
    concentration: Optional[float] = None
    concentration_unit: Optional[str] = None
    volume_uL: Optional[float] = None
    
    def validate(self) -> List[str]:
        """Validate chemical specification."""
        errors = []
        if self.concentration is not None and self.concentration < 0:
            errors.append(f"Negative concentration for {self.name}")
        if self.volume_uL is not None and self.volume_uL < 0:
            errors.append(f"Negative volume for {self.name}")
        return errors

@dataclass
class ReactionParams:
    chemicals: List[ChemicalSpec] = field(default_factory=list)
    temperature_C: Optional[float] = None
    stir_time_s: Optional[float] = None
    reaction_time_s: Optional[float] = None
    pH: Optional[float] = None
    solvent: Optional[str] = None
    description: Optional[str] = None
    conductor: Optional[str] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """Validate reaction parameters."""
        errors = []
        for chem in self.chemicals:
            errors.extend(chem.validate())
        if self.temperature_C is not None and (self.temperature_C < -273.15 or self.temperature_C > 1000):
            errors.append(f"Temperature {self.temperature_C}°C out of reasonable range")
        if self.pH is not None and (self.pH < 0 or self.pH > 14):
            errors.append(f"pH {self.pH} out of valid range (0-14)")
        return errors

@dataclass
class RunMetadata:
    project: str
    experiment: str
    run_id: str
    created_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    reaction: ReactionParams = field(default_factory=ReactionParams)
    sample_id: Optional[str] = None
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
    
    def validate(self) -> List[str]:
        """Validate metadata."""
        errors = []
        if not self.project:
            errors.append("Project name is required")
        if not self.experiment:
            errors.append("Experiment name is required")
        if not self.run_id:
            errors.append("Run ID is required")
        errors.extend(self.reaction.validate())
        return errors
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RunMetadata":
        """Robustly deserialize dictionary into nested dataclasses."""
        reaction_data = d.get("reaction", {}).copy()
        chems_data = reaction_data.pop("chemicals", [])
        chems = [ChemicalSpec(**c) for c in chems_data]
        rxn = ReactionParams(**reaction_data, chemicals=chems)
        meta_data = d.copy()
        meta_data["reaction"] = rxn
        known_fields = {f.name for f in dataclasses.fields(RunMetadata)}
        filtered = {k: v for k, v in meta_data.items() if k in known_fields}
        return RunMetadata(**filtered)

# -----------------------------
# Helpers
# -----------------------------
def _load_json(path: Path, default: Any):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            warnings.warn(f"Could not decode JSON from {path}. Returning default.")
            return default
    return default

def _save_json(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))

def _write_csv(path: Path, header: List[str], rows: List[Tuple]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(list(r))

def _read_csv(path: Path) -> Tuple[List[str], List[List[str]]]:
    if not path.exists():
        return [], []
    with path.open("r", newline="") as f:
        r = csv.reader(f)
        try:
            header = next(r)
            rows = [row for row in r]
            return header, rows
        except StopIteration:
            return [], []

def _safe(s: str) -> str:
    """Sanitize string for directory/filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def _norm(s: str) -> str:
    """Normalize string for comparison."""
    return re.sub(r"[^a-z0-9]", "", s.lower())

# -----------------------------
# Data validation utilities
# -----------------------------
class DataValidator:
    """Utilities for validating data quality."""
    
    @staticmethod
    def check_monotonic(values: List[float], name: str = "data") -> List[str]:
        """Check if values are monotonically increasing."""
        issues = []
        for i in range(1, len(values)):
            if values[i] <= values[i-1]:
                issues.append(f"{name} not monotonic at index {i}")
                break
        return issues
    
    @staticmethod
    def check_range(values: List[float], min_val: float, max_val: float, name: str = "data") -> List[str]:
        """Check if values are within expected range."""
        issues = []
        for i, v in enumerate(values):
            if v < min_val or v > max_val:
                issues.append(f"{name}[{i}] = {v} outside range [{min_val}, {max_val}]")
        return issues
    
    @staticmethod
    def check_no_nans(values: List[Any], name: str = "data") -> List[str]:
        """Check for missing/invalid values."""
        issues = []
        for i, v in enumerate(values):
            if v is None or (isinstance(v, float) and (v != v)):  # NaN check
                issues.append(f"{name}[{i}] is None or NaN")
        return issues

# -----------------------------
# Statistics utilities
# -----------------------------
class DataStats:
    """Statistical analysis utilities."""
    
    @staticmethod
    def basic_stats(values: List[float]) -> Dict[str, float]:
        """Calculate basic statistics."""
        if not values:
            return {}
        n = len(values)
        mean = sum(values) / n
        sorted_vals = sorted(values)
        median = sorted_vals[n//2] if n % 2 == 1 else (sorted_vals[n//2-1] + sorted_vals[n//2]) / 2
        variance = sum((x - mean)**2 for x in values) / n
        std = variance ** 0.5
        return {
            "count": n,
            "mean": mean,
            "median": median,
            "std": std,
            "min": min(values),
            "max": max(values)
        }
    
    @staticmethod
    def detect_outliers(values: List[float], threshold: float = 3.0) -> List[int]:
        """Detect outliers using z-score method."""
        if len(values) < 3:
            return []
        stats = DataStats.basic_stats(values)
        mean, std = stats["mean"], stats["std"]
        if std == 0:
            return []
        outliers = []
        for i, v in enumerate(values):
            z_score = abs((v - mean) / std)
            if z_score > threshold:
                outliers.append(i)
        return outliers

# -----------------------------
# Modality plugin base
# -----------------------------
class Modality:
    name = "base"
    
    def __init__(self, run_dir: Path):
        self.root = Path(run_dir)
    
    def ensure_dir(self) -> Path:
        d = self.root / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    def add(self, *args, **kwargs):
        raise NotImplementedError
    
    def load(self) -> Dict[str, Any]:
        raise NotImplementedError
    
    def plot(self, **kwargs):
        print(f"Plotting not implemented for modality '{self.name}'")
    
    def export_csv(self, output_path: Path) -> bool:
        """Export data to CSV file."""
        raise NotImplementedError
    
    def validate(self) -> List[str]:
        """Validate data quality."""
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for this modality."""
        return {}
    
    def _copy_into(self, files: List[Path], dest: Path) -> List[str]:
        dest.mkdir(parents=True, exist_ok=True)
        recorded = []
        for p in files:
            p = Path(p)
            tgt = dest / p.name
            if p.exists():
                shutil.copy2(p, tgt)
                recorded.append(str(tgt))
            else:
                warnings.warn(f"Source file {p} not found. Recording as external reference.")
                recorded.append(str(p))
        return recorded
    
    def _record_external(self, files: List[Path], extra_meta: Optional[Dict[str, Any]] = None):
        d = self.ensure_dir()
        payload = {
            "files": [str(Path(p).resolve()) for p in files],
            "meta": extra_meta or {},
            "linked": True
        }
        _save_json(d / FN.external_manifest, payload)
        return payload

_REGISTRY: Dict[str, type[Modality]] = {}

def register_modality(cls: type[Modality]):
    _REGISTRY[cls.name] = cls
    return cls

def get_registered_modalities() -> Dict[str, type[Modality]]:
    return dict(_REGISTRY)

# -----------------------------
# Built-in modalities
# -----------------------------
@register_modality
class UVVis(Modality):
    name = "uvvis"
    
    def add(
        self,
        time_s: List[float],
        wavelength_nm: List[float],
        absorbance: List[float],
        meta: Optional[Dict[str, Any]] = None,
    ):
        assert len(time_s) == len(wavelength_nm) == len(absorbance), "Arrays must be same length"
        d = self.ensure_dir()
        _write_csv(d / FN.uvvis_data, ["time_s", "wavelength_nm", "absorbance"], 
                   list(zip(time_s, wavelength_nm, absorbance)))
        _save_json(d / FN.modality_meta, meta or {})
    
    def link_files(self, csv_paths: List[Path | str], meta: Optional[Dict[str, Any]] = None):
        files = [Path(p) for p in csv_paths]
        return self._record_external(files, extra_meta={"kind": "uvvis", **(meta or {})})
    
    def _try_load_external_csv(self, path: Path):
        hdr, rows = _read_csv(path)
        if not hdr:
            return None
        names = [_norm(h) for h in hdr]
        
        def find(cands):
            for c in cands:
                if _norm(c) in names:
                    return names.index(_norm(c))
            return -1
        
        ti = find(["time", "times", "timesec", "time_s"])
        wi = find(["wavelength", "wavelengthnm", "lambda", "nm", "wl", "wavelength_nm"])
        ai = find(["absorbance", "a", "od", "opticaldensity", "abs"])
        
        if min(ti, wi, ai) >= 0:
            try:
                data = [[float(r[ti]), float(r[wi]), float(r[ai])] 
                        for r in rows if len(r) > max(ti, wi, ai)]
                return {"columns": ["time_s", "wavelength_nm", "absorbance"], "data": data}
            except Exception:
                return None
        return None
    
    def load(self) -> Dict[str, Any]:
        d = self.ensure_dir()
        csv_path = d / FN.uvvis_data
        
        if csv_path.exists():
            hdr, rows = _read_csv(csv_path)
            data = [[float(x) for x in r] for r in rows]
            meta = _load_json(d / FN.modality_meta, {})
            return {"columns": hdr, "data": data, "meta": meta}
        
        # Fallback to external links
        ext = _load_json(d / FN.external_manifest, {})
        files = [Path(p) for p in ext.get("files", [])]
        for p in files:
            parsed = self._try_load_external_csv(p)
            if parsed is not None:
                parsed["meta"] = ext.get("meta", {})
                parsed["external_files"] = [str(pp) for pp in files]
                return parsed
        
        return {"columns": [], "data": [], "meta": ext.get("meta", {}), 
                "external_files": [str(p) for p in files]}
    
    def validate(self) -> List[str]:
        """Validate UV-Vis data quality."""
        issues = []
        try:
            loaded = self.load()
            data = loaded.get("data", [])
            if not data:
                issues.append("No UV-Vis data found")
                return issues
            
            cols = loaded.get("columns", [])
            wl_idx = cols.index("wavelength_nm")
            abs_idx = cols.index("absorbance")
            
            wavelengths = [r[wl_idx] for r in data]
            absorbances = [r[abs_idx] for r in data]
            
            # Check wavelengths are in reasonable range
            issues.extend(DataValidator.check_range(wavelengths, 200, 1000, "wavelength_nm"))
            
            # Check absorbances are non-negative
            issues.extend(DataValidator.check_range(absorbances, 0, 10, "absorbance"))
            
        except Exception as e:
            issues.append(f"Error during validation: {e}")
        
        return issues
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for UV-Vis data."""
        try:
            loaded = self.load()
            data = loaded.get("data", [])
            if not data:
                return {}
            
            cols = loaded.get("columns", [])
            abs_idx = cols.index("absorbance")
            absorbances = [r[abs_idx] for r in data]
            
            return {
                "absorbance_stats": DataStats.basic_stats(absorbances),
                "num_points": len(data)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def export_csv(self, output_path: Path) -> bool:
        """Export UV-Vis data to CSV."""
        try:
            loaded = self.load()
            data = loaded.get("data", [])
            cols = loaded.get("columns", [])
            if not data:
                return False
            _write_csv(output_path, cols, [tuple(r) for r in data])
            return True
        except Exception as e:
            warnings.warn(f"Export failed: {e}")
            return False
    
    def plot(self, kind: str = "scatter", ax=None, **kwargs):
        """Plot UV-Vis data."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not found. Please `pip install matplotlib`")
            return
        
        loaded = self.load()
        cols, data = loaded.get("columns", []), loaded.get("data", [])
        
        if not data:
            print("No UV-Vis data to plot.")
            return
        
        wl_idx = cols.index("wavelength_nm")
        abs_idx = cols.index("absorbance")
        x = [r[wl_idx] for r in data]
        y = [r[abs_idx] for r in data]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 5)))
        
        if kind == "scatter":
            ax.scatter(x, y, s=kwargs.get("s", 2), alpha=kwargs.get("alpha", 0.6))
        else:
            ax.plot(x, y, linewidth=kwargs.get("linewidth", 1))
        
        ax.set_xlabel("Wavelength (nm)", fontsize=kwargs.get("fontsize", 11))
        ax.set_ylabel("Absorbance", fontsize=kwargs.get("fontsize", 11))
        ax.set_title(kwargs.get("title", "UV-Vis Spectrum"), fontsize=kwargs.get("fontsize", 12))
        ax.grid(kwargs.get("grid", True), alpha=0.3)
        
        if ax is None:
            plt.tight_layout()
            plt.show()
        
        return ax


@register_modality
class SAXS(Modality):
    name = "saxs"
    
    def add(
        self,
        q_invA: List[float],
        intensity: List[float],
        sigma: Optional[List[float]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        assert len(q_invA) == len(intensity), "q and intensity must match"
        rows = list(zip(q_invA, intensity, sigma if sigma else [None] * len(q_invA)))
        d = self.ensure_dir()
        _write_csv(d / FN.saxs_data, ["q_invA", "intensity", "sigma"], rows)
        _save_json(d / FN.modality_meta, meta or {})
    
    def link_files(self, csv_paths: List[Path | str], meta: Optional[Dict[str, Any]] = None):
        files = [Path(p) for p in csv_paths]
        return self._record_external(files, extra_meta={"kind": "saxs", **(meta or {})})
    
    def _try_load_external_csv(self, path: Path):
        hdr, rows = _read_csv(path)
        if not hdr:
            return None
        names = [_norm(h) for h in hdr]
        
        def find(cands):
            for c in cands:
                if _norm(c) in names:
                    return names.index(_norm(c))
            return -1
        
        qi = find(["q", "q1a", "qinva", "q1/Å", "q1a_"])
        Ii = find(["intensity", "i", "iq", "i(q)"])
        si = find(["sigma", "err", "stderr", "unc", "error"])
        
        if min(qi, Ii) >= 0:
            try:
                data = []
                for r in rows:
                    if len(r) <= max(qi, Ii, si if si>=0 else 0):
                        continue
                    qv = float(r[qi])
                    Iv = float(r[Ii])
                    sv = (float(r[si]) if si >= 0 and r[si] != "" else None)
                    data.append([qv, Iv, sv])
                return {"columns": ["q_invA", "intensity", "sigma"], "data": data}
            except Exception:
                return None
        return None
    
    def load(self) -> Dict[str, Any]:
        d = self.ensure_dir()
        csv_path = d / FN.saxs_data
        
        if csv_path.exists():
            hdr, rows = _read_csv(csv_path)
            def parse(x): return None if x == "" else float(x)
            data = [[parse(x) for x in r] for r in rows]
            meta = _load_json(d / FN.modality_meta, {})
            return {"columns": hdr, "data": data, "meta": meta}
        
        ext = _load_json(d / FN.external_manifest, {})
        files = [Path(p) for p in ext.get("files", [])]
        for p in files:
            parsed = self._try_load_external_csv(p)
            if parsed is not None:
                parsed["meta"] = ext.get("meta", {})
                parsed["external_files"] = [str(pp) for pp in files]
                return parsed
        
        return {"columns": [], "data": [], "meta": ext.get("meta", {}), 
                "external_files": [str(p) for p in files]}
    
    def validate(self) -> List[str]:
        """Validate SAXS data quality."""
        issues = []
        try:
            loaded = self.load()
            data = loaded.get("data", [])
            if not data:
                issues.append("No SAXS data found")
                return issues
            
            cols = loaded.get("columns", [])
            q_idx = cols.index("q_invA")
            I_idx = cols.index("intensity")
            
            q_values = [r[q_idx] for r in data]
            intensities = [r[I_idx] for r in data]
            
            # Check q values are positive and monotonic
            issues.extend(DataValidator.check_range(q_values, 0.001, 10, "q_invA"))
            issues.extend(DataValidator.check_monotonic(q_values, "q_invA"))
            
            # Check intensities are positive
            if any(I < 0 for I in intensities):
                issues.append("Negative intensity values found")
            
        except Exception as e:
            issues.append(f"Error during validation: {e}")
        
        return issues
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for SAXS data."""
        try:
            loaded = self.load()
            data = loaded.get("data", [])
            if not data:
                return {}
            
            cols = loaded.get("columns", [])
            I_idx = cols.index("intensity")
            intensities = [r[I_idx] for r in data]
            
            return {
                "intensity_stats": DataStats.basic_stats(intensities),
                "num_points": len(data)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def export_csv(self, output_path: Path) -> bool:
        """Export SAXS data to CSV."""
        try:
            loaded = self.load()
            data = loaded.get("data", [])
            cols = loaded.get("columns", [])
            if not data:
                return False
            _write_csv(output_path, cols, [tuple(r) for r in data])
            return True
        except Exception as e:
            warnings.warn(f"Export failed: {e}")
            return False
    
    def plot(self, loglog: bool = True, ax=None, **kwargs):
        """Plot SAXS data."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not found. Please `pip install matplotlib`")
            return
        
        loaded = self.load()
        cols, data = loaded.get("columns", []), loaded.get("data", [])
        
        if not data:
            print("No SAXS data to plot.")
            return
        
        q_idx = cols.index("q_invA")
        I_idx = cols.index("intensity")
        q = [r[q_idx] for r in data]
        I = [r[I_idx] for r in data]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 5)))
        
        if loglog:
            ax.loglog(q, I, linewidth=kwargs.get("linewidth", 1.5))
        else:
            ax.plot(q, I, linewidth=kwargs.get("linewidth", 1.5))
        
        ax.set_xlabel("q (1/Å)", fontsize=kwargs.get("fontsize", 11))
        ax.set_ylabel("Intensity (a.u.)", fontsize=kwargs.get("fontsize", 11))
        ax.set_title(kwargs.get("title", "SAXS Profile"), fontsize=kwargs.get("fontsize", 12))
        ax.grid(kwargs.get("grid", True), alpha=0.3)
        
        if ax is None:
            plt.tight_layout()
            plt.show()
        
        return ax


@register_modality
class WAXS(Modality):
    name = "waxs"
    
    def add(self, two_theta_deg: List[float], intensity: List[float], 
            meta: Optional[Dict[str, Any]] = None):
        assert len(two_theta_deg) == len(intensity), "length mismatch"
        d = self.ensure_dir()
        _write_csv(d / FN.waxs_data, ["two_theta_deg", "intensity"], 
                   list(zip(two_theta_deg, intensity)))
        _save_json(d / FN.modality_meta, meta or {})
    
    def link_files(self, csv_paths: List[Path | str], meta: Optional[Dict[str, Any]] = None):
        files = [Path(p) for p in csv_paths]
        return self._record_external(files, extra_meta={"kind": "waxs", **(meta or {})})
    
    def _try_load_external_csv(self, path: Path):
        hdr, rows = _read_csv(path)
        if not hdr:
            return None
        names = [_norm(h) for h in hdr]
        
        def find(cands):
            for c in cands:
                if _norm(c) in names:
                    return names.index(_norm(c))
            return -1
        
        xi = find(["twotheta", "2theta", "two_theta", "two_thetadeg", "twothetadeg", "2θ"])
        yi = find(["intensity", "counts", "i", "y"])
        
        if min(xi, yi) >= 0:
            try:
                data = [[float(r[xi]), float(r[yi])] for r in rows if len(r) > max(xi, yi)]
                return {"columns": ["two_theta_deg", "intensity"], "data": data}
            except Exception:
                return None
        return None
    
    def load(self) -> Dict[str, Any]:
        d = self.ensure_dir()
        csv_path = d / FN.waxs_data
        
        if csv_path.exists():
            hdr, rows = _read_csv(csv_path)
            data = [[float(x) for x in r] for r in rows]
            meta = _load_json(d / FN.modality_meta, {})
            return {"columns": hdr, "data": data, "meta": meta}
        
        ext = _load_json(d / FN.external_manifest, {})
        files = [Path(p) for p in ext.get("files", [])]
        for p in files:
            parsed = self._try_load_external_csv(p)
            if parsed is not None:
                parsed["meta"] = ext.get("meta", {})
                parsed["external_files"] = [str(pp) for pp in files]
                return parsed
        
        return {"columns": [], "data": [], "meta": ext.get("meta", {}), 
                "external_files": [str(p) for p in files]}
    
    def validate(self) -> List[str]:
        """Validate WAXS data quality."""
        issues = []
        try:
            loaded = self.load()
            data = loaded.get("data", [])
            if not data:
                issues.append("No WAXS data found")
                return issues
            
            cols = loaded.get("columns", [])
            theta_idx = cols.index("two_theta_deg")
            I_idx = cols.index("intensity")
            
            theta_values = [r[theta_idx] for r in data]
            intensities = [r[I_idx] for r in data]
            
            # Check 2theta values are in reasonable range
            issues.extend(DataValidator.check_range(theta_values, 0, 180, "two_theta_deg"))
            issues.extend(DataValidator.check_monotonic(theta_values, "two_theta_deg"))
            
            # Check intensities are non-negative
            if any(I < 0 for I in intensities):
                issues.append("Negative intensity values found")
            
        except Exception as e:
            issues.append(f"Error during validation: {e}")
        
        return issues
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for WAXS data."""
        try:
            loaded = self.load()
            data = loaded.get("data", [])
            if not data:
                return {}
            
            cols = loaded.get("columns", [])
            I_idx = cols.index("intensity")
            intensities = [r[I_idx] for r in data]
            
            return {
                "intensity_stats": DataStats.basic_stats(intensities),
                "num_points": len(data)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def export_csv(self, output_path: Path) -> bool:
        """Export WAXS data to CSV."""
        try:
            loaded = self.load()
            data = loaded.get("data", [])
            cols = loaded.get("columns", [])
            if not data:
                return False
            _write_csv(output_path, cols, [tuple(r) for r in data])
            return True
        except Exception as e:
            warnings.warn(f"Export failed: {e}")
            return False
    
    def plot(self, ax=None, **kwargs):
        """Plot WAXS data."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not found. Please `pip install matplotlib`")
            return
        
        loaded = self.load()
        cols, data = loaded.get("columns", []), loaded.get("data", [])
        
        if not data:
            print("No WAXS data to plot.")
            return
        
        theta_idx = cols.index("two_theta_deg")
        I_idx = cols.index("intensity")
        x = [r[theta_idx] for r in data]
        y = [r[I_idx] for r in data]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 5)))
        
        ax.plot(x, y, linewidth=kwargs.get("linewidth", 1.5))
        ax.set_xlabel("2θ (deg)", fontsize=kwargs.get("fontsize", 11))
        ax.set_ylabel("Intensity (a.u.)", fontsize=kwargs.get("fontsize", 11))
        ax.set_title(kwargs.get("title", "WAXS Pattern"), fontsize=kwargs.get("fontsize", 12))
        ax.grid(kwargs.get("grid", True), alpha=0.3)
        
        if ax is None:
            plt.tight_layout()
            plt.show()
        
        return ax


# Base class for image-based modalities
class ImageModality(Modality):
    def add(self, image_paths: List[Path | str], meta: Optional[Dict[str, Any]] = None, 
            copy: bool = True):
        d = self.ensure_dir()
        images_dir = d / FN.image_dir
        path_objects = [Path(p) for p in image_paths]
        
        if copy:
            recorded = self._copy_into(path_objects, images_dir)
        else:
            recorded = [str(p.resolve()) for p in path_objects]
        
        _save_json(d / FN.image_manifest, {
            "images": recorded,
            "meta": meta or {},
            "linked": not copy
        })
    
    def link_files(self, image_paths: List[Path | str], meta: Optional[Dict[str, Any]] = None):
        files = [Path(p) for p in image_paths]
        return self._record_external(files, extra_meta={"kind": self.name, **(meta or {})})
    
    def load(self) -> Dict[str, Any]:
        d = self.ensure_dir()
        
        if (d / FN.image_manifest).exists():
            return _load_json(d / FN.image_manifest, {"images": [], "meta": {}})
        
        ext = _load_json(d / FN.external_manifest, {})
        return {"images": ext.get("files", []), "meta": ext.get("meta", {}), "linked": True}
    
    def validate(self) -> List[str]:
        """Validate image data."""
        issues = []
        try:
            man = self.load()
            imgs = man.get("images", []) or man.get("files", [])
            
            if not imgs:
                issues.append(f"No {self.name.upper()} images found")
                return issues
            
            for i, img_path in enumerate(imgs):
                if not Path(img_path).exists():
                    issues.append(f"Image file not found: {img_path}")
            
        except Exception as e:
            issues.append(f"Error during validation: {e}")
        
        return issues
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for image data."""
        try:
            man = self.load()
            imgs = man.get("images", []) or man.get("files", [])
            
            valid_imgs = [p for p in imgs if Path(p).exists()]
            
            return {
                "num_images": len(imgs),
                "num_valid": len(valid_imgs),
                "num_missing": len(imgs) - len(valid_imgs)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def plot(self, idx: int = 0, ax=None, **kwargs):
        """Plot image."""
        try:
            from PIL import Image
            import matplotlib.pyplot as plt
        except ImportError:
            print("PIL or matplotlib not found. Please `pip install pillow matplotlib`")
            return
        
        man = self.load()
        imgs = man.get("images", []) or man.get("files", [])
        
        if not imgs:
            print(f"No {self.name.upper()} images recorded.")
            return
        
        if idx >= len(imgs):
            print(f"Index {idx} out of range. Only {len(imgs)} images available.")
            return
        
        try:
            im_path = imgs[idx]
            im = Image.open(im_path)
            
            if ax is None:
                fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 6)))
            
            ax.imshow(im, cmap=kwargs.get("cmap", None))
            ax.axis("off")
            title = kwargs.get("title", f"{self.name.upper()} image #{idx}")
            ax.set_title(title, fontsize=kwargs.get("fontsize", 12))
            
            if ax is None:
                plt.tight_layout()
                plt.show()
            
            return ax
            
        except FileNotFoundError:
            print(f"Error: Image file not found at {im_path}")
        except Exception as e:
            print(f"Error loading image: {e}")


@register_modality
class SEM(ImageModality):
    name = "sem"


@register_modality
class TEM(ImageModality):
    name = "tem"


# -----------------------------
# Core API: Run & DataOrganizer
# -----------------------------
class Run:
    def __init__(self, run_dir: Path, metadata: RunMetadata):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
        self._modality_cache: Dict[str, Modality] = {}
    
    def _get_modality(self, name: str) -> Modality:
        if name not in _REGISTRY:
            raise KeyError(f"Modality '{name}' not registered. Available: {list(_REGISTRY.keys())}")
        if name not in self._modality_cache:
            self._modality_cache[name] = _REGISTRY[name](self.dir)
        return self._modality_cache[name]
    
    # Accessors for built-ins
    def uvvis(self) -> UVVis:
        return self._get_modality("uvvis")
    
    def saxs(self) -> SAXS:
        return self._get_modality("saxs")
    
    def waxs(self) -> WAXS:
        return self._get_modality("waxs")
    
    def sem(self) -> SEM:
        return self._get_modality("sem")
    
    def tem(self) -> TEM:
        return self._get_modality("tem")
    
    def __getattr__(self, name: str) -> Modality:
        if name in _REGISTRY:
            return self._get_modality(name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'. "
                           f"If this is a modality, make sure it is registered.")
    
    def save_metadata(self):
        _save_json(self.dir / FN.run_metadata, self.metadata.to_dict())
    
    def validate_all(self) -> Dict[str, List[str]]:
        """Validate all modalities in this run."""
        results = {}
        
        # Validate metadata
        results["metadata"] = self.metadata.validate()
        
        # Validate each modality that has data
        for modality_name in _REGISTRY.keys():
            modality = self._get_modality(modality_name)
            mod_dir = self.dir / modality_name
            if mod_dir.exists():
                results[modality_name] = modality.validate()
        
        return results
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all modalities in this run."""
        stats = {}
        
        for modality_name in _REGISTRY.keys():
            modality = self._get_modality(modality_name)
            mod_dir = self.dir / modality_name
            if mod_dir.exists():
                stats[modality_name] = modality.get_stats()
        
        return stats
    
    def export_all(self, output_dir: Path) -> Dict[str, bool]:
        """Export all modality data to CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for modality_name in ["uvvis", "saxs", "waxs"]:  # Only CSV-exportable modalities
            modality = self._get_modality(modality_name)
            mod_dir = self.dir / modality_name
            if mod_dir.exists():
                output_path = output_dir / f"{modality_name}.csv"
                results[modality_name] = modality.export_csv(output_path)
        
        return results
    
    @staticmethod
    def load(run_dir: Path) -> "Run":
        meta_path = Path(run_dir) / FN.run_metadata
        meta_dict = _load_json(meta_path, {})
        if not meta_dict:
            raise FileNotFoundError(f"No '{FN.run_metadata}' found in {run_dir}. Cannot load run.")
        return Run(run_dir, RunMetadata.from_dict(meta_dict))


class DataOrganizer:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / FN.index
        self.index = _load_json(self.index_path, {"runs": [], "projects": {}})
    
    def _run_dir(self, project: str, experiment: str, run_id: str) -> Path:
        return self.root / _safe(project) / _safe(experiment) / _safe(run_id)
    
    def save_index(self):
        _save_json(self.index_path, self.index)
    
    def create_run(self, metadata: RunMetadata) -> Run:
        """Create a new run with metadata validation."""
        # Validate metadata before creating
        validation_errors = metadata.validate()
        if validation_errors:
            warnings.warn(f"Metadata validation issues: {'; '.join(validation_errors)}")
        
        d = self._run_dir(metadata.project, metadata.experiment, metadata.run_id)
        if d.exists():
            warnings.warn(f"Run directory {d} already exists. Overwriting.")
        
        run = Run(d, metadata)
        run.save_metadata()
        
        entry = {
            "project": metadata.project,
            "experiment": metadata.experiment,
            "run_id": metadata.run_id,
            "path": str(d.resolve()),
            "created_at": metadata.created_at,
            "tags": metadata.tags
        }
        
        # Dedupe by path
        self.index["runs"] = [r for r in self.index["runs"] if r.get("path") != entry["path"]]
        self.index["runs"].append(entry)
        self.save_index()
        
        return run
    
    def load_run(self, project: str, experiment: str, run_id: str) -> Run:
        d = self._run_dir(project, experiment, run_id)
        return Run.load(d)
    
    def get_run_by_path(self, path: Path | str) -> Run:
        return Run.load(Path(path))
    
    def list_runs(self, project: Optional[str] = None, experiment: Optional[str] = None,
                  tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List runs with optional filtering."""
        out = list(self.index.get("runs", []))
        
        if project:
            out = [r for r in out if r["project"] == project]
        
        if experiment:
            out = [r for r in out if r["experiment"] == experiment]
        
        if tags:
            out = [r for r in out if any(tag in r.get("tags", []) for tag in tags)]
        
        return out
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search runs by keyword in metadata."""
        q = query.lower()
        results: List[Dict[str, Any]] = []
        
        for r in self.index.get("runs", []):
            meta_path = Path(r["path"]) / FN.run_metadata
            if not meta_path.exists():
                continue
            try:
                meta_text = meta_path.read_text().lower()
                if q in meta_text:
                    results.append(r)
            except Exception:
                pass
        
        return results
    
    def batch_validate(self, project: Optional[str] = None) -> Dict[str, Any]:
        """Validate all runs in a project."""
        runs_to_check = self.list_runs(project=project)
        results = {}
        
        for run_entry in runs_to_check:
            run_key = f"{run_entry['project']}/{run_entry['experiment']}/{run_entry['run_id']}"
            try:
                run = self.get_run_by_path(run_entry['path'])
                results[run_key] = run.validate_all()
            except Exception as e:
                results[run_key] = {"error": str(e)}
        
        return results
    
    def batch_export(self, output_root: Path, project: Optional[str] = None) -> Dict[str, Any]:
        """Export all runs in a project."""
        output_root = Path(output_root)
        runs_to_export = self.list_runs(project=project)
        results = {}
        
        for run_entry in runs_to_export:
            run_key = f"{run_entry['project']}/{run_entry['experiment']}/{run_entry['run_id']}"
            try:
                run = self.get_run_by_path(run_entry['path'])
                output_dir = output_root / run_entry['project'] / run_entry['experiment'] / run_entry['run_id']
                results[run_key] = run.export_all(output_dir)
            except Exception as e:
                results[run_key] = {"error": str(e)}
        
        return results
    
    def get_summary(self, project: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics for all runs."""
        runs = self.list_runs(project=project)
        
        summary = {
            "total_runs": len(runs),
            "projects": set(),
            "experiments": set(),
            "by_modality": defaultdict(int)
        }
        
        for run_entry in runs:
            summary["projects"].add(run_entry["project"])
            summary["experiments"].add(run_entry["experiment"])
            
            # Check which modalities have data
            run_path = Path(run_entry["path"])
            for modality in _REGISTRY.keys():
                mod_dir = run_path / modality
                if mod_dir.exists() and any(mod_dir.iterdir()):
                    summary["by_modality"][modality] += 1
        
        summary["projects"] = list(summary["projects"])
        summary["experiments"] = list(summary["experiments"])
        summary["by_modality"] = dict(summary["by_modality"])
        
        return summary
    
    # External tree import (from original code)
    def import_external_tree(
        self,
        project_name: str,
        external_project_root: Path | str,
        directory_map: DirectoryMap = DirectoryMap(),
        schema: ImportSchema = ImportSchema(),
        link_only: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Scan an existing project tree and register/link files into runs."""
        root = Path(external_project_root)
        if not root.exists():
            raise FileNotFoundError(f"External project root not found: {root}")
        
        proj_key = _safe(project_name)
        self.index.setdefault("projects", {}).setdefault(proj_key, {
            "external_root": str(root.resolve())
        })
        
        date_re = re.compile(schema.date_regex)
        
        def collect_files(modality_folder: str, patterns: List[str]) -> List[Path]:
            mdir = root / modality_folder
            if not mdir.exists():
                return []
            found: List[Path] = []
            for pat in patterns:
                found.extend(mdir.rglob(pat))
            return found
        
        buckets: Dict[Tuple[str, str], Dict[str, List[Path]]] = {}
        
        # Collect files
        uvvis_files = collect_files(directory_map.uvvis, ["*.csv", "*.CSV"])
        saxs_files = collect_files(directory_map.saxs, ["*.csv", "*.dat", "*.txt"])
        waxs_files = collect_files(directory_map.waxs, ["*.csv", "*.dat", "*.txt"])
        img_pats = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
        sem_files = collect_files(directory_map.sem, img_pats)
        tem_files = collect_files(directory_map.tem, img_pats)
        
        def assign(file_path: Path, modality_name: str):
            mod_dir = root / getattr(directory_map, modality_name)
            try:
                rel = file_path.relative_to(mod_dir)
            except Exception:
                return
            
            parts = rel.parts
            if len(parts) < 2:
                return
            
            date_part, sample = parts[0], parts[1]
            if not date_re.fullmatch(date_part):
                return
            
            try:
                dt = datetime.datetime.strptime(date_part, schema.date_format).date()
                experiment_iso = dt.isoformat()
            except Exception:
                experiment_iso = date_part
            
            key = (experiment_iso, sample)
            buckets.setdefault(key, {}).setdefault(modality_name, []).append(file_path)
        
        for f in uvvis_files:
            assign(f, "uvvis")
        for f in saxs_files:
            assign(f, "saxs")
        for f in waxs_files:
            assign(f, "waxs")
        for f in sem_files:
            assign(f, "sem")
        for f in tem_files:
            assign(f, "tem")
        
        # Handle recipes
        recipes_dir = root / directory_map.recipes
        if recipes_dir.exists():
            recs = sorted([str(p.resolve()) for p in recipes_dir.glob("*.csv")])
            proj_meta = self.root / _safe(project_name) / "_project" / "recipes.json"
            _save_json(proj_meta, {"files": recs})
        
        summary = {"project": project_name, "runs_created": 0, "by_run": {}}
        
        if dry_run:
            for (exp, sample), mods in buckets.items():
                summary["by_run"][f"{exp}/{sample}"] = {
                    k: [str(p) for p in v] for k, v in mods.items()
                }
            return summary
        
        # Create runs and link data
        for (experiment_iso, sample), mods in buckets.items():
            meta = RunMetadata(project=project_name, experiment=experiment_iso, run_id=sample)
            run = self.create_run(meta)
            
            if "uvvis" in mods:
                run.uvvis().link_files(mods["uvvis"], 
                    meta={"source": "external", "external_root": str(root.resolve())})
            if "saxs" in mods:
                run.saxs().link_files(mods["saxs"], 
                    meta={"source": "external", "external_root": str(root.resolve())})
            if "waxs" in mods:
                run.waxs().link_files(mods["waxs"], 
                    meta={"source": "external", "external_root": str(root.resolve())})
            if "sem" in mods:
                run.sem().link_files(mods["sem"], 
                    meta={"source": "external", "external_root": str(root.resolve())})
            if "tem" in mods:
                run.tem().link_files(mods["tem"], 
                    meta={"source": "external", "external_root": str(root.resolve())})
            
            summary["by_run"][f"{experiment_iso}/{sample}"] = {
                k: [str(p) for p in v] for k, v in mods.items()
            }
            summary["runs_created"] += 1
        
        self.index["projects"][proj_key] = {
            "external_root": str(root.resolve()),
            "directory_map": dataclasses.asdict(directory_map),
            "schema": dataclasses.asdict(schema),
        }
        self.save_index()
        
        return summary


# -----------------------------
# Visualization utilities
# -----------------------------
class VisualizationHelper:
    """Helper class for advanced visualization."""
    
    @staticmethod
    def compare_runs(runs: List[Run], modality: str = "uvvis", **kwargs):
        """Compare multiple runs side by side."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not found. Please `pip install matplotlib`")
            return
        
        n_runs = len(runs)
        fig, axes = plt.subplots(1, n_runs, figsize=kwargs.get("figsize", (6*n_runs, 5)))
        
        if n_runs == 1:
            axes = [axes]
        
        for i, run in enumerate(runs):
            mod = getattr(run, modality)()
            title = f"{run.metadata.run_id}"
            mod.plot(ax=axes[i], title=title, **kwargs)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def multi_modality_plot(run: Run, modalities: List[str] = ["uvvis", "saxs", "waxs"], **kwargs):
        """Plot multiple modalities for a single run."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not found. Please `pip install matplotlib`")
            return
        
        n_mods = len(modalities)
        fig, axes = plt.subplots(1, n_mods, figsize=kwargs.get("figsize", (6*n_mods, 5)))
        
        if n_mods == 1:
            axes = [axes]
        
        for i, mod_name in enumerate(modalities):
            mod = getattr(run, mod_name)()
            mod.plot(ax=axes[i], **kwargs)
        
        plt.suptitle(f"Run: {run.metadata.run_id}", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()