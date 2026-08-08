from pathlib import Path

from NanoOrganizer.web_app.components.scattering import (
    discover_scattering_products,
    index_frames,
)


def test_discover_products_counts_root_and_focused_product(tmp_path):
    root = tmp_path / "giwaxs"
    for name, suffix in (
        ("cir_avg", ".csv"),
        ("q_image", ".npz"),
        ("qphi", ".npz"),
        ("qc", ".png"),
    ):
        folder = root / name
        folder.mkdir(parents=True)
        (folder / f"{name}_one{suffix}").touch()
        (folder / f"{name}_two{suffix}").touch()

    normalized, products, focused = discover_scattering_products(str(root))
    assert Path(normalized) == root.resolve()
    assert focused is None
    assert {item["key"]: item["count"] for item in products} == {
        "cir_avg": 2,
        "q_image": 2,
        "qphi": 2,
        "qc": 2,
    }

    normalized, products, focused = discover_scattering_products(
        str(root / "q_image")
    )
    assert Path(normalized) == root.resolve()
    assert focused == "q_image"
    assert [(item["key"], item["count"]) for item in products] == [
        ("q_image", 2)
    ]


def test_index_frames_matches_qc_with_other_reduction_products(tmp_path):
    root = tmp_path / "giwaxs"
    for folder in ("qc", "q_image", "qphi", "cir_avg"):
        (root / folder).mkdir(parents=True)

    stem = "Kim_EUV_1_0.1000deg_2026_08_08_12_00_00_10.00s_1234567_000000_WAXS"
    (root / "qc" / f"qc_{stem}.png").touch()
    (root / "q_image" / f"qimg_{stem}.tif.npz").touch()
    (root / "qphi" / f"qphi_{stem}.tif.npz").touch()
    (root / "cir_avg" / f"Cir_Avg_{stem}.tif.csv").touch()

    frame_table = index_frames(str(root))

    assert len(frame_table) == 1
    row = frame_table.iloc[0]
    assert bool(row["has_qc"])
    assert bool(row["has_qimg"])
    assert bool(row["has_qphi"])
    assert bool(row["has_cir"])

    qimg_only = index_frames(str(root), product_keys=["q_image"])
    qimg_row = qimg_only.iloc[0]
    assert bool(qimg_row["has_qimg"])
    assert not bool(qimg_row["has_qc"])
    assert not bool(qimg_row["has_qphi"])
    assert not bool(qimg_row["has_cir"])
