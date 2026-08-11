from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from NanoOrganizer.viz.scattering_notebook import (
    QImageData,
    crop_qimage,
    downsample_qimage,
    load_qimage,
    load_qphi,
    plot_qimage,
    plot_qphi,
    product_index,
    qimage_linecuts,
    qimage_stats,
    robust_limits,
    valid_intensity,
)
from NanoOrganizer.web_app.components.scattering import apply_mask


def _write_qimage(path: Path, *, include_mask=True):
    qimg = np.array(
        [[0.0, 1.0, 2.0], [3.0, np.nan, 5.0], [6.0, 7.0, 8.0]]
    )
    qx = np.array([-1.0, 0.0, 1.0])
    qz = np.array([0.0, 1.0, 2.0])
    values = {"qimg": qimg, "qx": qx, "qz": qz}
    if include_mask:
        # Reduction convention: True means valid remeshed data.
        values["qimg_mask"] = np.array(
            [[False, True, True], [True, True, True], [True, True, True]]
        )
    np.savez(path, **values)


def test_load_qimage_normalizes_valid_data_mask_and_stats(tmp_path):
    path = tmp_path / "qimg_demo.tif.npz"
    _write_qimage(path)

    data = load_qimage(path)
    masked = valid_intensity(data.intensity, data.mask)

    assert data.intensity.shape == (3, 3)
    assert data.mask[0, 0]
    assert not data.mask[0, 1]
    assert masked.mask[0, 0]
    assert not masked.mask[0, 1]
    assert masked.mask[1, 1]
    assert qimage_stats(data)["valid_pixels"] == 7.0


def test_loaders_validate_geometry_and_ignore_detector_grid_qphi_mask(tmp_path):
    qimage_path = tmp_path / "bad.npz"
    np.savez(
        qimage_path,
        qimg=np.ones((2, 3)),
        qx=np.arange(3),
        qz=np.arange(4),
    )
    with pytest.raises(ValueError, match="geometry"):
        load_qimage(qimage_path)

    qphi_path = tmp_path / "qphi.npz"
    np.savez(
        qphi_path,
        qphi=np.ones((2, 3)),
        q=np.arange(3),
        phi=np.arange(2),
        qphi_mask=np.ones((8, 9), dtype=bool),
    )
    qphi = load_qphi(qphi_path)
    assert qphi.intensity.shape == (2, 3)
    assert qphi.mask is None


def test_crop_downsample_limits_and_linecuts():
    intensity = np.arange(24, dtype=float).reshape(4, 6) + 1
    data = QImageData(
        intensity=intensity,
        qx=np.linspace(-1, 4, 6),
        qz=np.linspace(0, 3, 4),
        mask=np.zeros_like(intensity, dtype=bool),
    )
    cropped = crop_qimage(data, qxlim=(0, 2), qzlim=(1, 2))
    assert cropped.intensity.shape == (2, 3)
    assert cropped.qx.tolist() == [0.0, 1.0, 2.0]
    assert cropped.qz.tolist() == [1.0, 2.0]

    reduced = downsample_qimage(data, max_pixels=6)
    assert reduced.intensity.size <= 12
    assert reduced.intensity.shape == reduced.mask.shape

    cuts = qimage_linecuts(data, qx=0.1, qz=1.1)
    assert cuts["selected_qx"] == 0.0
    assert cuts["selected_qz"] == 1.0
    assert cuts["qx_profile"].shape == (6,)
    assert cuts["qz_profile"].shape == (4,)
    assert robust_limits(intensity) == pytest.approx((1.23, 23.885))


def test_product_index_pairs_reduction_products(tmp_path):
    root = tmp_path / "giwaxs"
    stem = "sample_001_WAXS"
    for folder in ("qc", "q_image", "qphi", "cir_avg"):
        (root / folder).mkdir(parents=True)
    (root / "qc" / f"qc_{stem}.tif.png").touch()
    (root / "q_image" / f"qimg_{stem}.tif.npz").touch()
    (root / "qphi" / f"qphi_{stem}.tif.npz").touch()
    (root / "cir_avg" / f"Cir_Avg_{stem}.tif.csv").touch()

    index = product_index(root)
    assert len(index) == 1
    row = index.iloc[0]
    assert row["stem"] == stem
    assert all(bool(row[f"has_{product}"]) for product in ("qc", "q_image", "qphi", "cir_avg"))
    assert Path(row["q_image"]).name == f"qimg_{stem}.tif.npz"


def test_plot_helpers_return_matplotlib_objects(tmp_path):
    qimage_path = tmp_path / "qimg.tif.npz"
    _write_qimage(qimage_path)
    qimage = load_qimage(qimage_path)
    qphi_path = tmp_path / "qphi.npz"
    np.savez(
        qphi_path,
        qphi=np.array([[1.0, 2.0], [3.0, 4.0]]),
        q=np.array([0.1, 0.2]),
        phi=np.array([-5.0, 5.0]),
        qphi_mask=np.ones((2, 2), dtype=bool),
    )
    qphi = load_qphi(qphi_path)

    fig1, ax1, image1 = plot_qimage(qimage, max_pixels=20)
    fig2, ax2, image2 = plot_qphi(qphi)
    try:
        assert image1.get_array().ndim == 2
        assert image2.get_array().shape == (2, 2)
        assert ax1.get_xlabel() == r"$q_x$ ($\AA^{-1}$)"
        assert ax2.get_ylabel() == r"$\phi$ (deg)"
    finally:
        plt.close(fig1)
        plt.close(fig2)


def test_gui_mask_helper_keeps_current_valid_qimage_support():
    values = np.array([[0.0, 2.0], [3.0, 4.0]])
    valid_mask = np.array([[False, True], [True, True]])

    shown = apply_mask(values, valid_mask)

    assert np.isnan(shown[0, 0])
    assert np.isfinite(shown[0, 1:]).all()
    assert np.isfinite(shown[1]).all()
