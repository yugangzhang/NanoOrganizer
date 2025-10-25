import numpy as np
from pathlib import Path
import shutil



def simulate_uvvis_data(wavelength_range=(200, 800), n_points=300, peak_center=520, peak_width=50):
    """Simulate UV-Vis absorption spectrum with Gaussian peak."""
    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_points)
    # Gaussian peak + baseline
    absorbance = 0.1 + 1.5 * np.exp(-((wavelengths - peak_center) ** 2) / (2 * peak_width ** 2))
    # Add some noise
    absorbance += np.random.normal(0, 0.02, n_points)
    times = np.linspace(0, 600, n_points)  # 10 minute scan
    return times.tolist(), wavelengths.tolist(), absorbance.tolist()


def simulate_saxs_data(q_range=(0.01, 0.5), n_points=200, size_nm=5):
    """Simulate SAXS intensity profile for spherical nanoparticles."""
    q = np.linspace(q_range[0], q_range[1], n_points)
    # Simple spherical form factor approximation
    radius_A = size_nm * 10 / 2  # convert nm to Angstroms
    intensity = 1000 * (3 * (np.sin(q * radius_A) - q * radius_A * np.cos(q * radius_A)) / (q * radius_A) ** 3) ** 2
    # Add Poisson noise
    intensity = intensity + np.random.normal(0, 0.05 * intensity, n_points)
    intensity = np.maximum(intensity, 0.1)  # Floor at small positive value
    return q.tolist(), intensity.tolist()


def simulate_waxs_data(two_theta_range=(10, 80), n_points=500, peaks=[(30, 100), (35, 80), (62, 60)]):
    """Simulate WAXS diffraction pattern with crystalline peaks."""
    two_theta = np.linspace(two_theta_range[0], two_theta_range[1], n_points)
    intensity = np.ones(n_points) * 10  # baseline
    
    # Add Gaussian peaks
    for peak_pos, peak_height in peaks:
        intensity += peak_height * np.exp(-((two_theta - peak_pos) ** 2) / (2 * 1.5 ** 2))
    
    # Add noise
    intensity += np.random.normal(0, 2, n_points)
    intensity = np.maximum(intensity, 0)
    
    return two_theta.tolist(), intensity.tolist()


def create_fake_image(output_path: Path, size=(512, 512), pattern="sem"):
    """Create a fake microscopy image."""
    try:
        from PIL import Image
        img = Image.new('L', size)
        pixels = img.load()
        
        for i in range(size[0]):
            for j in range(size[1]):
                # Create some texture
                if pattern == "sem":
                    # Simulate grain-like structure
                    val = int(128 + 50 * np.sin(i / 20) * np.cos(j / 20) + 
                             np.random.normal(0, 20))
                else:  # tem
                    # Simulate circular particles
                    center_x, center_y = size[0] // 2, size[1] // 2
                    dist = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                    val = int(200 - dist / 3 + np.random.normal(0, 15))
                
                pixels[i, j] = max(0, min(255, val))
        
        img.save(output_path)
        print(f"  ✓ Created {pattern.upper()} image: {output_path.name}")
        return True
    except ImportError:
        print(f"  ✗ PIL not available, skipping image creation")
        return False

