# Mixar Assignment — Adaptive Quantization & Seam Tokenization

## Overview

This project implements the Mesh Normalization, Quantization, and Error Analysis pipeline assigned by Mixar for the ML Engineer hiring process. It extends the base assignment with:

* Adaptive Quantization — region-based binning driven by local vertex density
* Enhanced Error Metrics — MSE, MAE, Chamfer, Hausdorff, and normal-angle
* Seam Tokenization (Bonus Task) — robust multi-cue seam detection and token encoding for 3D mesh understanding

The code is modular and environment-agnostic — runs on any system with Python ≥ 3.9.

---

## Setup

Install Python ≥ 3.9 and pip. Then clone this repository and install dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

```
numpy
scipy
scikit-learn
matplotlib
imageio
pillow
trimesh
open3d
pyglet<2
pyopengl
imageio-ffmpeg
```

If you face OpenGL errors on Windows:

```bash
pip install "pyglet<2" PyOpenGL PyOpenGL_accelerate
```

---

## Running the Pipeline

From the project root:

```bash
python -m src.pipeline --input_dir meshes --out_dir results --clusters 4 --base_bins 1024 --k_density 16 --alpha 1.0
```

### What this does

* Loads .obj meshes from the `meshes/` folder
* Applies normalization (Min–Max and Unit Sphere)
* Performs adaptive quantization using vertex density clustering
* Reconstructs meshes and computes metrics
* Visualizes error plots and mesh comparison renders
* Detects seams and generates token sequences (Bonus Task)

### Example Output Structure

```
results/
├── person/
│   ├── minmax_recon.obj
│   ├── minmax_compare.png
│   ├── unit_sphere_recon.obj
│   ├── unit_sphere_compare.png
│   ├── seams_visualization.ply
│   ├── seams_tokens.json
│   ├── summary.csv
│   ├── bins_per_vertex.npy
│   └── stats.json
```

---

## Seam Tokenization (Bonus Task)

The seam tokenizer combines multiple geometric cues:

* Boundary edges (edges appearing in only one face)
* High-dihedral edges (sharp creases)
* Curvature edges (vertices with high normal variance)

Each seam is represented as a token sequence:

```json
{
  "t": "V",
  "v": 1542,
  "e": 4321,
  "l": 0.0059,
  "d": 28.4
}
```

These tokens can serve as a foundation for mesh-based language models like SeamGPT.

---

## Error Metrics

Each reconstruction computes:

* Mean Squared Error (MSE) per axis
* Mean Absolute Error (MAE) per axis
* Chamfer Distance (symmetric)
* Hausdorff Distance
* Normal-Angle Error (°)

A detailed CSV summary is saved for each mesh under `results/<mesh>/summary.csv`.

---

## Visualization

Each mesh outputs side-by-side comparison renders (`*_compare.png`) and seam-visualization files (`seams_visualization.ply`).

If your system supports OpenGL and pyglet < 2, the comparison renders will be shaded 3D images. Otherwise, fallback 2D projections are generated.

---

## Troubleshooting

| Issue                                                    | Cause                    | Fix                                            |
| -------------------------------------------------------- | ------------------------ | ---------------------------------------------- |
| `ModuleNotFoundError: src`                               | Ran from wrong directory | Run `python -m src.pipeline` from project root |
| `ImportError: trimesh.viewer.windowed requires pyglet<2` | Pyglet v2 installed      | `pip install "pyglet<2"`                       |
| `OpenGL context error`                                   | Missing bindings         | `pip install PyOpenGL PyOpenGL_accelerate`     |
| Flat renders (no shading)                                | Fallback renderer used   | Install pyglet + PyOpenGL for 3D rendering     |

---

## Repository Structure

```
├── src/
│   ├── pipeline.py
│   ├── normalize.py
│   ├── density.py
│   ├── quantize.py
│   ├── metrics.py
│   ├── seam_tokenizer.py
│   ├── visualize.py
│   └── utils.py
├── meshes/
├── results/
├── requirements.txt
├── README.md
└── report.pdf
```

---

## Report Guidelines

Include a concise `report.pdf` summarizing:

1. Objective – explain normalization, quantization, and seam tokenization.
2. Methods – describe adaptive quantization and multi-cue seam detection.
3. Results – error table and example visuals.
4. Conclusions – adaptive quantization improved geometry fidelity; seam tokens captured mesh discontinuities.

---

## Author

**Pranay Thakur**
Final Year B.Tech — Artificial Intelligence
Email: [thakurpranayyy@gmail.com](mailto:thakurpranayyy@gmail.com)
