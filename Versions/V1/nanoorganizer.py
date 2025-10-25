#!/usr/bin/env python3
"""
NanoOrganizer: A modular metadata and data management system for nanoparticle synthesis.

Key Features:
- Flexible metadata management with JSON storage
- Lazy data loading (load metadata fast, data on demand)
- Any directory structure (you provide paths)
- Data validation and integrity checks
- Easy visualization interface

Usage:
    from NanoOrganizer import DataOrganizer, RunMetadata, ReactionParams, ChemicalSpec
    
    # Create organizer
    org = DataOrganizer("/path/to/project")
    
    # Create run with metadata
    run = org.create_run(metadata)
    
    # Link data files
    run.uvvis.link_data(csv_files, time_points=[0, 30, 60, ...])
    
    # Save everything
    org.save()
    
    # Later: load and use
    org = DataOrganizer.load("/path/to/project")
    run = org.get_run("project", "experiment", "run_id")
    data = run.uvvis.load()  # Lazy load actual data
    run.uvvis.plot()         # Visualize
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import warnings


# ============================================================================
# Metadata Classes
# ============================================================================

@dataclass
class ChemicalSpec:
    """Chemical specification for a reaction."""
    name: str
    concentration: float
    concentration_unit: str = "mM"
    volume_uL: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ChemicalSpec':
        return cls(**data)


@dataclass
class ReactionParams:
    """Reaction parameters for nanoparticle synthesis."""
    chemicals: List[ChemicalSpec]
    temperature_C: float = 25.0
    stir_time_s: float = 0.0
    reaction_time_s: float = 0.0
    pH: Optional[float] = None
    solvent: str = "Water"
    conductor: str = "Unknown"
    description: str = ""
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data['chemicals'] = [c.to_dict() for c in self.chemicals]
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ReactionParams':
        chemicals = [ChemicalSpec.from_dict(c) for c in data.pop('chemicals')]
        return cls(chemicals=chemicals, **data)


@dataclass
class RunMetadata:
    """Metadata for a single experimental run."""
    project: str
    experiment: str  # Usually a date like "2024-10-20"
    run_id: str
    sample_id: str
    reaction: ReactionParams
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data['reaction'] = self.reaction.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'RunMetadata':
        reaction = ReactionParams.from_dict(data.pop('reaction'))
        return cls(reaction=reaction, **data)


# ============================================================================
# Data Link Classes (stores references to data files)
# ============================================================================

@dataclass
class DataLink:
    """Base class for data file links."""
    file_paths: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    data_type: str = ""
    
    def to_dict(self) -> dict:
        return {
            'file_paths': self.file_paths,
            'metadata': self.metadata,
            'data_type': self.data_type
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DataLink':
        return cls(**data)
    
    def validate(self) -> List[str]:
        """Check if all files exist. Returns list of missing files."""
        missing = []
        for fpath in self.file_paths:
            if not Path(fpath).exists():
                missing.append(fpath)
        return missing


# ============================================================================
# Data Accessor Classes (lazy loading + visualization)
# ============================================================================

class UVVisData:
    """UV-Vis spectroscopy data accessor with lazy loading."""
    
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.link = DataLink(data_type="uvvis")
        self._loaded_data = None
    
    def link_data(self, csv_files: List[Union[str, Path]], 
                  time_points: Optional[List[float]] = None,
                  metadata: Optional[Dict] = None):
        """
        Link UV-Vis CSV files to this run.
        
        Parameters
        ----------
        csv_files : list
            List of CSV file paths (e.g., ['uvvis_001.csv', 'uvvis_002.csv', ...])
        time_points : list, optional
            Time points corresponding to each file (in seconds)
        metadata : dict, optional
            Additional metadata (instrument, operator, etc.)
        """
        self.link.file_paths = [str(Path(f).absolute()) for f in csv_files]
        
        if metadata:
            self.link.metadata.update(metadata)
        
        if time_points is not None:
            if len(time_points) != len(csv_files):
                raise ValueError(f"Number of time points ({len(time_points)}) must match "
                               f"number of CSV files ({len(csv_files)})")
            self.link.metadata['time_points'] = time_points
        
        print(f"  ✓ Linked {len(csv_files)} UV-Vis CSV files")
    
    def validate(self) -> bool:
        """Check if all linked files exist."""
        missing = self.link.validate()
        if missing:
            warnings.warn(f"Missing UV-Vis files:\n" + "\n".join(f"  - {f}" for f in missing))
            return False
        return True
    
    def load(self, force_reload: bool = False) -> Dict[str, np.ndarray]:
        """
        Load UV-Vis data from CSV files (lazy loading).
        
        Returns
        -------
        data : dict
            Dictionary with keys: 'times', 'wavelengths', 'absorbance'
        """
        if self._loaded_data is not None and not force_reload:
            return self._loaded_data
        
        if not self.link.file_paths:
            raise ValueError("No data files linked. Use link_data() first.")
        
        # Load data from all CSV files
        all_times = []
        all_wavelengths = []
        all_absorbance = []
        
        time_points = self.link.metadata.get('time_points')
        
        for i, fpath in enumerate(self.link.file_paths):
            fpath = Path(fpath)
            if not fpath.exists():
                warnings.warn(f"File not found: {fpath}")
                continue
            
            # Read CSV
            try:
                data = np.loadtxt(fpath, delimiter=',', skiprows=1)
                wavelengths = data[:, 0]
                absorbance = data[:, 1]
                
                # Get time point
                if time_points:
                    time = time_points[i]
                else:
                    # Try to extract from filename (e.g., uvvis_t0060s.csv -> 60)
                    time = self._extract_time_from_filename(fpath.name, i)
                
                # Store
                times = np.full(len(wavelengths), time)
                all_times.extend(times)
                all_wavelengths.extend(wavelengths)
                all_absorbance.extend(absorbance)
                
            except Exception as e:
                warnings.warn(f"Error reading {fpath}: {e}")
                continue
        
        self._loaded_data = {
            'times': np.array(all_times),
            'wavelengths': np.array(all_wavelengths),
            'absorbance': np.array(all_absorbance)
        }
        
        print(f"  ✓ Loaded {len(self.link.file_paths)} UV-Vis spectra "
              f"({len(self._loaded_data['times'])} total data points)")
        
        return self._loaded_data
    
    def _extract_time_from_filename(self, filename: str, index: int) -> float:
        """Try to extract time from filename, otherwise use index."""
        import re
        # Try patterns like: t0060s, t060, _60s, etc.
        patterns = [r't(\d+)s', r't(\d+)', r'_(\d+)s', r'_(\d+)']
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return float(match.group(1))
        # Fallback: use index
        return float(index)
    
    def plot(self, plot_type: str = "spectrum", time_point: Optional[float] = None,
             wavelength: Optional[float] = None, ax=None, **kwargs):
        """
        Plot UV-Vis data.
        
        Parameters
        ----------
        plot_type : str
            'spectrum' - single spectrum at time_point
            'kinetics' - absorbance vs time at wavelength
            'heatmap' - 2D evolution map
        time_point : float, optional
            Time point for 'spectrum' plot
        wavelength : float, optional
            Wavelength for 'kinetics' plot
        ax : matplotlib axis, optional
            Axis to plot on
        """
        data = self.load()
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting. Install: pip install matplotlib")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        if plot_type == "spectrum":
            self._plot_spectrum(data, time_point, ax, **kwargs)
        elif plot_type == "kinetics":
            self._plot_kinetics(data, wavelength, ax, **kwargs)
        elif plot_type == "heatmap":
            self._plot_heatmap(data, ax, **kwargs)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
        
        return ax
    
    def _plot_spectrum(self, data, time_point, ax, **kwargs):
        """Plot spectrum at a specific time point."""
        import matplotlib.pyplot as plt
        
        times = data['times']
        wavelengths = data['wavelengths']
        absorbance = data['absorbance']
        
        unique_times = np.unique(times)
        
        if time_point is None:
            time_point = unique_times[len(unique_times)//2]  # Middle time
        
        # Find closest time
        closest_time = unique_times[np.argmin(np.abs(unique_times - time_point))]
        mask = times == closest_time
        
        wl = wavelengths[mask]
        abs_val = absorbance[mask]
        
        ax.plot(wl, abs_val, linewidth=2, label=f't = {closest_time:.0f}s')
        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('Absorbance (a.u.)', fontsize=12)
        ax.set_title(kwargs.get('title', f'UV-Vis Spectrum at t={closest_time:.0f}s'), 
                     fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_kinetics(self, data, wavelength, ax, **kwargs):
        """Plot absorbance vs time at specific wavelength."""
        import matplotlib.pyplot as plt
        
        times = data['times']
        wavelengths = data['wavelengths']
        absorbance = data['absorbance']
        
        if wavelength is None:
            wavelength = 520  # Default
        
        # Find closest wavelength
        unique_wl = np.unique(wavelengths)
        closest_wl = unique_wl[np.argmin(np.abs(unique_wl - wavelength))]
        
        # Get time series at this wavelength
        unique_times = np.unique(times)
        abs_at_wl = []
        for t in unique_times:
            mask = (times == t) & (np.abs(wavelengths - closest_wl) < 1.0)
            if np.any(mask):
                abs_at_wl.append(np.mean(absorbance[mask]))
            else:
                abs_at_wl.append(np.nan)
        
        ax.plot(unique_times, abs_at_wl, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Absorbance (a.u.)', fontsize=12)
        ax.set_title(kwargs.get('title', f'Growth Kinetics at λ={closest_wl:.0f}nm'), 
                     fontsize=13)
        ax.grid(True, alpha=0.3)
    
    def _plot_heatmap(self, data, ax, **kwargs):
        """Plot 2D heatmap of time vs wavelength."""
        import matplotlib.pyplot as plt
        
        times = data['times']
        wavelengths = data['wavelengths']
        absorbance = data['absorbance']
        
        # Create 2D grid
        unique_times = np.unique(times)
        unique_wl = np.unique(wavelengths)
        
        # Find data for each time point
        n_wl_per_time = int(len(wavelengths) / len(unique_times))
        Z = absorbance.reshape(len(unique_times), n_wl_per_time)
        wl_grid = wavelengths[:n_wl_per_time]
        
        im = ax.pcolormesh(wl_grid, unique_times, Z, shading='auto', cmap='viridis')
        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('Time (s)', fontsize=12)
        ax.set_title(kwargs.get('title', 'UV-Vis Evolution'), fontsize=13)
        
        import matplotlib.pyplot as plt
        plt.colorbar(im, ax=ax, label='Absorbance (a.u.)')


class SAXSData:
    """SAXS data accessor with lazy loading."""
    
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.link = DataLink(data_type="saxs")
        self._loaded_data = None
    
    def link_data(self, csv_files: List[Union[str, Path]], 
                  time_points: Optional[List[float]] = None,
                  metadata: Optional[Dict] = None):
        """Link SAXS CSV files to this run."""
        self.link.file_paths = [str(Path(f).absolute()) for f in csv_files]
        
        if metadata:
            self.link.metadata.update(metadata)
        
        if time_points is not None:
            self.link.metadata['time_points'] = time_points
        
        print(f"  ✓ Linked {len(csv_files)} SAXS CSV files")
    
    def validate(self) -> bool:
        """Check if all linked files exist."""
        missing = self.link.validate()
        if missing:
            warnings.warn(f"Missing SAXS files:\n" + "\n".join(f"  - {f}" for f in missing))
            return False
        return True
    
    def load(self, force_reload: bool = False) -> Dict[str, np.ndarray]:
        """Load SAXS data from CSV files."""
        if self._loaded_data is not None and not force_reload:
            return self._loaded_data
        
        if not self.link.file_paths:
            raise ValueError("No data files linked. Use link_data() first.")
        
        all_times = []
        all_q = []
        all_intensity = []
        
        time_points = self.link.metadata.get('time_points')
        
        for i, fpath in enumerate(self.link.file_paths):
            fpath = Path(fpath)
            if not fpath.exists():
                warnings.warn(f"File not found: {fpath}")
                continue
            
            try:
                data = np.loadtxt(fpath, delimiter=',', skiprows=1)
                q = data[:, 0]
                intensity = data[:, 1]
                
                time = time_points[i] if time_points else float(i)
                times = np.full(len(q), time)
                
                all_times.extend(times)
                all_q.extend(q)
                all_intensity.extend(intensity)
                
            except Exception as e:
                warnings.warn(f"Error reading {fpath}: {e}")
                continue
        
        self._loaded_data = {
            'times': np.array(all_times),
            'q': np.array(all_q),
            'intensity': np.array(all_intensity)
        }
        
        print(f"  ✓ Loaded {len(self.link.file_paths)} SAXS profiles")
        return self._loaded_data
    
    def plot(self, plot_type: str = "profile", time_point: Optional[float] = None,
             q_value: Optional[float] = None, loglog: bool = True, ax=None, **kwargs):
        """Plot SAXS data."""
        data = self.load()
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        if plot_type == "profile":
            self._plot_profile(data, time_point, ax, loglog, **kwargs)
        elif plot_type == "kinetics":
            self._plot_kinetics(data, q_value, ax, **kwargs)
        elif plot_type == "heatmap":
            self._plot_heatmap(data, ax, **kwargs)
        
        return ax
    
    def _plot_profile(self, data, time_point, ax, loglog, **kwargs):
        """Plot I(q) at specific time."""
        times = data['times']
        q = data['q']
        intensity = data['intensity']
        
        unique_times = np.unique(times)
        if time_point is None:
            time_point = unique_times[len(unique_times)//2]
        
        closest_time = unique_times[np.argmin(np.abs(unique_times - time_point))]
        mask = times == closest_time
        
        q_vals = q[mask]
        I_vals = intensity[mask]
        
        if loglog:
            ax.loglog(q_vals, I_vals, 'o-', linewidth=2, markersize=4)
        else:
            ax.plot(q_vals, I_vals, 'o-', linewidth=2, markersize=4)
        
        ax.set_xlabel('q (1/Å)', fontsize=12)
        ax.set_ylabel('Intensity (a.u.)', fontsize=12)
        ax.set_title(kwargs.get('title', f'SAXS Profile at t={closest_time:.0f}s'), fontsize=13)
        ax.grid(True, alpha=0.3)
    
    def _plot_kinetics(self, data, q_value, ax, **kwargs):
        """Plot I(t) at specific q."""
        times = data['times']
        q = data['q']
        intensity = data['intensity']
        
        if q_value is None:
            q_value = 0.02
        
        unique_q = np.unique(q)
        closest_q = unique_q[np.argmin(np.abs(unique_q - q_value))]
        
        unique_times = np.unique(times)
        I_at_q = []
        for t in unique_times:
            mask = (times == t) & (np.abs(q - closest_q) < 0.001)
            if np.any(mask):
                I_at_q.append(np.mean(intensity[mask]))
            else:
                I_at_q.append(np.nan)
        
        ax.plot(unique_times, I_at_q, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Intensity (a.u.)', fontsize=12)
        ax.set_title(kwargs.get('title', f'Growth at q={closest_q:.3f} 1/Å'), fontsize=13)
        ax.grid(True, alpha=0.3)
    
    def _plot_heatmap(self, data, ax, **kwargs):
        """Plot 2D heatmap."""
        import matplotlib.pyplot as plt
        
        times = data['times']
        q = data['q']
        intensity = data['intensity']
        
        unique_times = np.unique(times)
        n_q_per_time = int(len(q) / len(unique_times))
        
        Z = intensity.reshape(len(unique_times), n_q_per_time)
        q_grid = q[:n_q_per_time]
        
        im = ax.pcolormesh(q_grid, unique_times, Z, shading='auto', cmap='viridis')
        ax.set_xlabel('q (1/Å)', fontsize=12)
        ax.set_ylabel('Time (s)', fontsize=12)
        ax.set_title(kwargs.get('title', 'SAXS Evolution'), fontsize=13)
        plt.colorbar(im, ax=ax, label='Intensity (a.u.)')


class WAXSData:
    """WAXS data accessor with lazy loading."""
    
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.link = DataLink(data_type="waxs")
        self._loaded_data = None
    
    def link_data(self, csv_files: List[Union[str, Path]], 
                  time_points: Optional[List[float]] = None,
                  metadata: Optional[Dict] = None):
        """Link WAXS CSV files to this run."""
        self.link.file_paths = [str(Path(f).absolute()) for f in csv_files]
        
        if metadata:
            self.link.metadata.update(metadata)
        
        if time_points is not None:
            self.link.metadata['time_points'] = time_points
        
        print(f"  ✓ Linked {len(csv_files)} WAXS CSV files")
    
    def validate(self) -> bool:
        """Check if all linked files exist."""
        missing = self.link.validate()
        if missing:
            warnings.warn(f"Missing WAXS files:\n" + "\n".join(f"  - {f}" for f in missing))
            return False
        return True
    
    def load(self, force_reload: bool = False) -> Dict[str, np.ndarray]:
        """Load WAXS data from CSV files."""
        if self._loaded_data is not None and not force_reload:
            return self._loaded_data
        
        if not self.link.file_paths:
            raise ValueError("No data files linked. Use link_data() first.")
        
        all_times = []
        all_two_theta = []
        all_intensity = []
        
        time_points = self.link.metadata.get('time_points')
        
        for i, fpath in enumerate(self.link.file_paths):
            fpath = Path(fpath)
            if not fpath.exists():
                warnings.warn(f"File not found: {fpath}")
                continue
            
            try:
                data = np.loadtxt(fpath, delimiter=',', skiprows=1)
                two_theta = data[:, 0]
                intensity = data[:, 1]
                
                time = time_points[i] if time_points else float(i)
                times = np.full(len(two_theta), time)
                
                all_times.extend(times)
                all_two_theta.extend(two_theta)
                all_intensity.extend(intensity)
                
            except Exception as e:
                warnings.warn(f"Error reading {fpath}: {e}")
                continue
        
        self._loaded_data = {
            'times': np.array(all_times),
            'two_theta': np.array(all_two_theta),
            'intensity': np.array(all_intensity)
        }
        
        print(f"  ✓ Loaded {len(self.link.file_paths)} WAXS patterns")
        return self._loaded_data
    
    def plot(self, plot_type: str = "pattern", time_point: Optional[float] = None,
             two_theta_value: Optional[float] = None, ax=None, **kwargs):
        """Plot WAXS data."""
        data = self.load()
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        if plot_type == "pattern":
            self._plot_pattern(data, time_point, ax, **kwargs)
        elif plot_type == "kinetics":
            self._plot_kinetics(data, two_theta_value, ax, **kwargs)
        elif plot_type == "heatmap":
            self._plot_heatmap(data, ax, **kwargs)
        
        return ax
    
    def _plot_pattern(self, data, time_point, ax, **kwargs):
        """Plot diffraction pattern at specific time."""
        times = data['times']
        two_theta = data['two_theta']
        intensity = data['intensity']
        
        unique_times = np.unique(times)
        if time_point is None:
            time_point = unique_times[len(unique_times)//2]
        
        closest_time = unique_times[np.argmin(np.abs(unique_times - time_point))]
        mask = times == closest_time
        
        tt_vals = two_theta[mask]
        I_vals = intensity[mask]
        
        ax.plot(tt_vals, I_vals, linewidth=2)
        ax.set_xlabel('2θ (degrees)', fontsize=12)
        ax.set_ylabel('Intensity (a.u.)', fontsize=12)
        ax.set_title(kwargs.get('title', f'WAXS Pattern at t={closest_time:.0f}s'), fontsize=13)
        ax.grid(True, alpha=0.3)
    
    def _plot_kinetics(self, data, two_theta_value, ax, **kwargs):
        """Plot peak intensity vs time."""
        times = data['times']
        two_theta = data['two_theta']
        intensity = data['intensity']
        
        if two_theta_value is None:
            two_theta_value = 30.0
        
        unique_tt = np.unique(two_theta)
        closest_tt = unique_tt[np.argmin(np.abs(unique_tt - two_theta_value))]
        
        unique_times = np.unique(times)
        I_at_tt = []
        for t in unique_times:
            mask = (times == t) & (np.abs(two_theta - closest_tt) < 0.5)
            if np.any(mask):
                I_at_tt.append(np.mean(intensity[mask]))
            else:
                I_at_tt.append(np.nan)
        
        ax.plot(unique_times, I_at_tt, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Intensity (a.u.)', fontsize=12)
        ax.set_title(kwargs.get('title', f'Peak Growth at 2θ={closest_tt:.1f}°'), fontsize=13)
        ax.grid(True, alpha=0.3)
    
    def _plot_heatmap(self, data, ax, **kwargs):
        """Plot 2D heatmap."""
        import matplotlib.pyplot as plt
        
        times = data['times']
        two_theta = data['two_theta']
        intensity = data['intensity']
        
        unique_times = np.unique(times)
        n_tt_per_time = int(len(two_theta) / len(unique_times))
        
        Z = intensity.reshape(len(unique_times), n_tt_per_time)
        tt_grid = two_theta[:n_tt_per_time]
        
        im = ax.pcolormesh(tt_grid, unique_times, Z, shading='auto', cmap='viridis')
        ax.set_xlabel('2θ (degrees)', fontsize=12)
        ax.set_ylabel('Time (s)', fontsize=12)
        ax.set_title(kwargs.get('title', 'WAXS Crystallization'), fontsize=13)
        plt.colorbar(im, ax=ax, label='Intensity (a.u.)')


class ImageData:
    """Microscopy image data accessor."""
    
    def __init__(self, run_id: str, data_type: str = "sem"):
        self.run_id = run_id
        self.data_type = data_type
        self.link = DataLink(data_type=data_type)
    
    def link_data(self, image_files: List[Union[str, Path]], 
                  metadata: Optional[Dict] = None):
        """Link image files to this run."""
        self.link.file_paths = [str(Path(f).absolute()) for f in image_files]
        
        if metadata:
            self.link.metadata.update(metadata)
        
        print(f"  ✓ Linked {len(image_files)} {self.data_type.upper()} images")
    
    def validate(self) -> bool:
        """Check if all linked files exist."""
        missing = self.link.validate()
        if missing:
            warnings.warn(f"Missing {self.data_type.upper()} files:\n" + 
                         "\n".join(f"  - {f}" for f in missing))
            return False
        return True
    
    def load(self, index: Optional[int] = None):
        """Load images."""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("PIL required. Install: pip install Pillow")
        
        if not self.link.file_paths:
            raise ValueError("No image files linked. Use link_data() first.")
        
        if index is not None:
            return Image.open(self.link.file_paths[index])
        else:
            return [Image.open(f) for f in self.link.file_paths]
    
    def plot(self, index: int = 0, ax=None, **kwargs):
        """Display an image."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 8)))
        
        img = self.load(index)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        
        fname = Path(self.link.file_paths[index]).name
        ax.set_title(kwargs.get('title', f'{self.data_type.upper()}: {fname}'), 
                     fontsize=13)
        
        return ax


# ============================================================================
# Run Class (contains all data accessors)
# ============================================================================

class Run:
    """
    Represents a single experimental run with all associated data.
    """
    
    def __init__(self, metadata: RunMetadata, base_dir: Path):
        self.metadata = metadata
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / metadata.project / metadata.experiment / metadata.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Data accessors
        self.uvvis = UVVisData(metadata.run_id)
        self.saxs = SAXSData(metadata.run_id)
        self.waxs = WAXSData(metadata.run_id)
        self.sem = ImageData(metadata.run_id, "sem")
        self.tem = ImageData(metadata.run_id, "tem")
    
    def to_dict(self) -> dict:
        """Convert run to dictionary for JSON storage."""
        return {
            'metadata': self.metadata.to_dict(),
            'data': {
                'uvvis': self.uvvis.link.to_dict(),
                'saxs': self.saxs.link.to_dict(),
                'waxs': self.waxs.link.to_dict(),
                'sem': self.sem.link.to_dict(),
                'tem': self.tem.link.to_dict(),
            }
        }
    
    @classmethod
    def from_dict(cls, data: dict, base_dir: Path) -> 'Run':
        """Load run from dictionary."""
        metadata = RunMetadata.from_dict(data['metadata'])
        run = cls(metadata, base_dir)
        
        # Restore data links
        run.uvvis.link = DataLink.from_dict(data['data']['uvvis'])
        run.saxs.link = DataLink.from_dict(data['data']['saxs'])
        run.waxs.link = DataLink.from_dict(data['data']['waxs'])
        run.sem.link = DataLink.from_dict(data['data']['sem'])
        run.tem.link = DataLink.from_dict(data['data']['tem'])
        
        return run
    
    def validate(self) -> Dict[str, bool]:
        """Validate all data links."""
        return {
            'uvvis': self.uvvis.validate() if self.uvvis.link.file_paths else True,
            'saxs': self.saxs.validate() if self.saxs.link.file_paths else True,
            'waxs': self.waxs.validate() if self.waxs.link.file_paths else True,
            'sem': self.sem.validate() if self.sem.link.file_paths else True,
            'tem': self.tem.validate() if self.tem.link.file_paths else True,
        }


# ============================================================================
# Main DataOrganizer Class
# ============================================================================

class DataOrganizer:
    """
    Main organizer for nanoparticle synthesis data.
    
    Features:
    - JSON-based metadata storage
    - Flexible directory structure
    - Lazy data loading
    - Data validation
    """
    
    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize DataOrganizer.
        
        Parameters
        ----------
        base_dir : str or Path
            Base directory for storing metadata and organizing data
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_dir = self.base_dir / ".metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        self.runs: Dict[str, Run] = {}
        self._index_file = self.metadata_dir / "index.json"
        
        # Load existing runs if any
        self._load_index()
    
    def create_run(self, metadata: RunMetadata) -> Run:
        """
        Create a new run.
        
        Parameters
        ----------
        metadata : RunMetadata
            Metadata for the run
        
        Returns
        -------
        run : Run
            The created run object
        """
        run_key = f"{metadata.project}/{metadata.experiment}/{metadata.run_id}"
        
        if run_key in self.runs:
            warnings.warn(f"Run {run_key} already exists. Overwriting.")
        
        run = Run(metadata, self.base_dir)
        self.runs[run_key] = run
        
        print(f"✓ Created run: {run_key}")
        return run
    
    def get_run(self, project: str, experiment: str, run_id: str) -> Optional[Run]:
        """Get a run by its identifiers."""
        run_key = f"{project}/{experiment}/{run_id}"
        return self.runs.get(run_key)
    
    def list_runs(self) -> List[str]:
        """List all runs."""
        return list(self.runs.keys())
    
    def save(self):
        """Save all metadata to JSON files."""
        print("\n--- Saving Metadata ---")
        
        # Save individual run files
        for run_key, run in self.runs.items():
            run_file = self.metadata_dir / f"{run_key.replace('/', '_')}.json"
            run_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(run_file, 'w') as f:
                json.dump(run.to_dict(), f, indent=2)
            
            print(f"  ✓ Saved: {run_key}")
        
        # Save index
        index = {
            'runs': list(self.runs.keys()),
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self._index_file, 'w') as f:
            json.dump(index, f, indent=2)
        
        print(f"  ✓ Saved index: {len(self.runs)} runs")
        print(f"\n✓ Metadata saved to: {self.metadata_dir}")
    
    def _load_index(self):
        """Load index of runs."""
        if not self._index_file.exists():
            return
        
        with open(self._index_file, 'r') as f:
            index = json.load(f)
        
        # Load each run
        for run_key in index['runs']:
            run_file = self.metadata_dir / f"{run_key.replace('/', '_')}.json"
            if run_file.exists():
                with open(run_file, 'r') as f:
                    run_data = json.load(f)
                self.runs[run_key] = Run.from_dict(run_data, self.base_dir)
    
    @classmethod
    def load(cls, base_dir: Union[str, Path]) -> 'DataOrganizer':
        """
        Load existing DataOrganizer from directory.
        
        Parameters
        ----------
        base_dir : str or Path
            Base directory containing metadata
        
        Returns
        -------
        organizer : DataOrganizer
            Loaded organizer with all runs
        """
        org = cls(base_dir)
        print(f"✓ Loaded DataOrganizer: {len(org.runs)} runs")
        return org
    
    def validate_all(self) -> Dict[str, Dict[str, bool]]:
        """Validate all runs."""
        print("\n--- Validating Data Files ---")
        results = {}
        
        for run_key, run in self.runs.items():
            results[run_key] = run.validate()
            
            # Print summary
            all_valid = all(results[run_key].values())
            status = "✓" if all_valid else "⚠"
            print(f"  {status} {run_key}")
        
        return results


# ============================================================================
# Convenience function
# ============================================================================

def save_time_series_to_csv(output_dir: Path, prefix: str, 
                            times: List[float], x_values: List[float], 
                            y_values: List[float],
                            x_name: str = "x", y_name: str = "y") -> List[Path]:
    """
    Save time-series data to individual CSV files.
    
    Parameters
    ----------
    output_dir : Path
        Directory to save CSV files
    prefix : str
        File prefix (e.g., 'uvvis', 'saxs')
    times : list
        Time values for each data point
    x_values : list
        X-axis values (wavelength, q, 2theta, etc.)
    y_values : list
        Y-axis values (absorbance, intensity, etc.)
    x_name : str
        Name for x column
    y_name : str
        Name for y column
    
    Returns
    -------
    csv_files : list
        List of created CSV file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    times = np.array(times)
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    
    unique_times = np.unique(times)
    csv_files = []
    
    for i, t in enumerate(unique_times):
        mask = times == t
        x_at_t = x_values[mask]
        y_at_t = y_values[mask]
        
        # Create CSV filename
        csv_file = output_dir / f"{prefix}_{i+1:03d}.csv"
        
        # Save CSV
        header = f"{x_name},{y_name}"
        data = np.column_stack([x_at_t, y_at_t])
        np.savetxt(csv_file, data, delimiter=',', header=header, comments='')
        
        csv_files.append(csv_file)
    
    print(f"  ✓ Saved {len(csv_files)} CSV files to: {output_dir}")
    return csv_files


if __name__ == "__main__":
    print("NanoOrganizer Module")
    print("=" * 70)
    print("\nImport this module to use:")
    print("  from NanoOrganizer import DataOrganizer, RunMetadata, ReactionParams, ChemicalSpec")





    