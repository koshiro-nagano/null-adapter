"""LIB-HSI loader (CSIRO DAP collection 58044, CC BY-NC 4.0; Habili et al., 2022).

Layout (inferred from the data, then confirmed against all 513 ENVI .hdr files by scripts/check_headers.py):
  reflectance_cubes/<id>.dat : float32 little-endian, 512 x 204 x 512 BIL (line, band, sample) = Specim IQ
                               (204 bands, 397.32-1003.58 nm; centres from the dataset's wavelengths.py).
                               The cube is rotated 90 deg relative to rgb/labels; load_cube() returns it aligned.
  rgb/<id>.png               : 512 x 512 8-bit RGB rendering shipped with the dataset (not derived from our A).
  labels/<id>.png            : 512 x 512 colour-coded labels; colours -> classes via label_info.json (44 classes,
                               class_group 0-8).
Checks behind these choices: tests/test_libhsi.py (band 60 vs rgb correlation, vegetation red edge, 940 nm H2O dip).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

H = W = 512
N_BANDS = 204
# Specim IQ band centres, copied from LIB-HSI/wavelengths.py
WAVELENGTHS = np.array([
    397.32, 400.20, 403.09, 405.97, 408.85, 411.74, 414.63, 417.52, 420.40, 423.29, 426.19, 429.08, 431.97, 434.87,
    437.76, 440.66, 443.56, 446.45, 449.35, 452.25, 455.16, 458.06, 460.96, 463.87, 466.77, 469.68, 472.59, 475.50,
    478.41, 481.32, 484.23, 487.14, 490.06, 492.97, 495.89, 498.80, 501.72, 504.64, 507.56, 510.48, 513.40, 516.33,
    519.25, 522.18, 525.10, 528.03, 530.96, 533.89, 536.82, 539.75, 542.68, 545.62, 548.55, 551.49, 554.43, 557.36,
    560.30, 563.24, 566.18, 569.12, 572.07, 575.01, 577.96, 580.90, 583.85, 586.80, 589.75, 592.70, 595.65, 598.60,
    601.55, 604.51, 607.46, 610.42, 613.38, 616.34, 619.30, 622.26, 625.22, 628.18, 631.15, 634.11, 637.08, 640.04,
    643.01, 645.98, 648.95, 651.92, 654.89, 657.87, 660.84, 663.81, 666.79, 669.77, 672.75, 675.73, 678.71, 681.69,
    684.67, 687.65, 690.64, 693.62, 696.61, 699.60, 702.58, 705.57, 708.57, 711.56, 714.55, 717.54, 720.54, 723.53,
    726.53, 729.53, 732.53, 735.53, 738.53, 741.53, 744.53, 747.54, 750.54, 753.55, 756.56, 759.56, 762.57, 765.58,
    768.60, 771.61, 774.62, 777.64, 780.65, 783.67, 786.68, 789.70, 792.72, 795.74, 798.77, 801.79, 804.81, 807.84,
    810.86, 813.89, 816.92, 819.95, 822.98, 826.01, 829.04, 832.07, 835.11, 838.14, 841.18, 844.22, 847.25, 850.29,
    853.33, 856.37, 859.42, 862.46, 865.50, 868.55, 871.60, 874.64, 877.69, 880.74, 883.79, 886.84, 889.90, 892.95,
    896.01, 899.06, 902.12, 905.18, 908.24, 911.30, 914.36, 917.42, 920.48, 923.55, 926.61, 929.68, 932.74, 935.81,
    938.88, 941.95, 945.02, 948.10, 951.17, 954.24, 957.32, 960.40, 963.47, 966.55, 969.63, 972.71, 975.79, 978.88,
    981.96, 985.05, 988.13, 991.22, 994.31, 997.40, 1000.49, 1003.58])
assert len(WAVELENGTHS) == N_BANDS


def load_classes(root) -> list[dict]:
    return json.loads((Path(root) / "label_info.json").read_text())["items"]


def list_ids(root, split: str) -> list[str]:
    return sorted(p.stem for p in (Path(root) / split / "reflectance_cubes").glob("*.dat"))


def load_cube(root, split: str, sid: str) -> np.ndarray:
    """(H, W, 204) float32 reflectance, aligned with rgb/labels."""
    a = np.fromfile(Path(root) / split / "reflectance_cubes" / f"{sid}.dat", dtype="<f4")
    assert a.size == H * W * N_BANDS, (sid, a.size)
    return np.ascontiguousarray(np.rot90(a.reshape(H, N_BANDS, W).transpose(0, 2, 1), -1))


def load_rgb(root, split: str, sid: str) -> np.ndarray:
    return np.asarray(Image.open(Path(root) / split / "rgb" / f"{sid}.png").convert("RGB"))


def load_labels(root, split: str, sid: str, classes=None) -> np.ndarray:
    """(H, W) int class index into label_info.json items; -1 where the colour is unknown."""
    classes = classes or load_classes(root)
    lab = np.asarray(Image.open(Path(root) / split / "labels" / f"{sid}.png").convert("RGB")).astype(np.int32)
    code = (lab[..., 0] << 16) | (lab[..., 1] << 8) | lab[..., 2]
    out = np.full(code.shape, -1, dtype=np.int16)
    for i, c in enumerate(classes):
        out[code == int(c["color_hex_code"][1:], 16)] = i
    return out
