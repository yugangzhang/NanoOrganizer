#!/usr/bin/env python3
"""
Verification script for NanoOrganizer web GUI installation.
Run this to check if all components are properly installed.
"""

import sys
from pathlib import Path

def check_imports():
    """Check if all required modules can be imported."""
    print("=" * 60)
    print("Checking Python Imports...")
    print("=" * 60)

    required = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('pandas', 'Pandas'),
        ('streamlit', 'Streamlit'),
    ]

    optional = [
        ('PIL', 'Pillow (Image support)'),
        ('h5py', 'HDF5 support'),
    ]

    all_ok = True

    for module, name in required:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - REQUIRED")
            all_ok = False

    print("\nOptional:")
    for module, name in optional:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️  {name} - Optional")

    return all_ok


def check_nanoorganizer():
    """Check if NanoOrganizer can be imported."""
    print("\n" + "=" * 60)
    print("Checking NanoOrganizer Package...")
    print("=" * 60)

    try:
        import NanoOrganizer
        print(f"✅ NanoOrganizer imported successfully")
        print(f"   Version: {NanoOrganizer.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Cannot import NanoOrganizer: {e}")
        return False


def check_files():
    """Check if all web GUI files exist."""
    print("\n" + "=" * 60)
    print("Checking Web GUI Files...")
    print("=" * 60)

    base_dir = Path(__file__).parent / "NanoOrganizer" / "web"

    files = {
        'app.py': 'Main Data Viewer',
        'cli.py': 'Data Viewer CLI',
        'csv_plotter.py': 'CSV Plotter',
        'csv_plotter_cli.py': 'CSV Plotter CLI',
        'data_manager.py': 'Data Manager',
        'data_manager_cli.py': 'Data Manager CLI',
        'plotter_3d.py': '3D Plotter',
        'plotter_3d_cli.py': '3D Plotter CLI',
    }

    all_exist = True
    for filename, description in files.items():
        filepath = base_dir / filename
        if filepath.exists():
            print(f"✅ {filename:25s} - {description}")
        else:
            print(f"❌ {filename:25s} - {description} - MISSING")
            all_exist = False

    return all_exist


def check_console_scripts():
    """Check if console scripts are installed."""
    print("\n" + "=" * 60)
    print("Checking Console Scripts...")
    print("=" * 60)
    print("NOTE: These will only work after running:")
    print("      pip install -e \".[web,image]\"")
    print()

    import subprocess

    scripts = [
        'nanoorganizer-viz',
        'nanoorganizer-csv',
        'nanoorganizer-manage',
        'nanoorganizer-3d',
    ]

    installed = []
    missing = []

    for script in scripts:
        result = subprocess.run(['which', script], capture_output=True)
        if result.returncode == 0:
            print(f"✅ {script:25s} -> {result.stdout.decode().strip()}")
            installed.append(script)
        else:
            print(f"⚠️  {script:25s} - Not found in PATH")
            missing.append(script)

    return installed, missing


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print(" NanoOrganizer Web GUI Installation Verification")
    print("=" * 60 + "\n")

    results = []

    # Check imports
    results.append(("Python packages", check_imports()))

    # Check NanoOrganizer
    results.append(("NanoOrganizer package", check_nanoorganizer()))

    # Check files
    results.append(("Web GUI files", check_files()))

    # Check console scripts
    installed, missing = check_console_scripts()

    # Summary
    print("\n" + "=" * 60)
    print(" Summary")
    print("=" * 60)

    all_ok = all(r[1] for r in results)

    for name, ok in results:
        status = "✅ OK" if ok else "❌ FAILED"
        print(f"{status:10s} {name}")

    if installed:
        print(f"✅ OK       Console scripts ({len(installed)}/4 installed)")
    if missing:
        print(f"⚠️  PARTIAL  Console scripts ({len(missing)}/4 missing)")

    # Installation instructions
    if missing or not all_ok:
        print("\n" + "=" * 60)
        print(" Installation Instructions")
        print("=" * 60)

        if not all_ok:
            print("\n1. Install missing Python packages:")
            print("   pip install numpy scipy matplotlib pandas streamlit pillow")

        if missing:
            print("\n2. Reinstall NanoOrganizer to register console scripts:")
            print("   cd /home/yuzhang/Repos/NanoOrganizer")
            print("   sudo rm -rf Nanoorganizer.egg-info build dist")
            print("   pip install -e \".[web,image]\"")

        print("\n3. Test the installation:")
        print("   python verify_installation.py")

    else:
        print("\n🎉 All checks passed! Your installation is complete.")
        print("\nYou can now use these commands:")
        for script in installed:
            print(f"   {script}")

    return 0 if all_ok and len(installed) == 4 else 1


if __name__ == "__main__":
    sys.exit(main())
