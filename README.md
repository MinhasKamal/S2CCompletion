# S2C-Completion

### 3DV2026 - Official repository for "CAOA-Completion-Assisted Object-CAD Alignment"

[Paper](https://openreview.net/pdf/c438ccb8604dc76f5f355bfafa7516a62c716275.pdf) | [Project](https://github.com/kumarhiranya/S2CCompletion) | [SyntheticPCDCompletionDataset-GColab](https://colab.research.google.com/drive/14UBBk6mOH5IHOBv5ABmiyF4vus9ZtnmN?usp=sharing)

**S2C-Completion** is a dataset and annotation toolkit for establishing ground-truth correspondences between real-world 3D point cloud instances from ScanNetv2 and their matching CAD models from ShapeNet, building on top of the [Scan2CAD](https://github.com/skanti/Scan2CAD) alignment dataset.

---

## Table of Contents

- [Overview](#overview)
- [Dataset Structure](#dataset-structure)
  - [Top-level Fields](#top-level-fields)
  - [Per-Model Fields](#per-model-fields)
- [Prerequisites & Environment Setup](#prerequisites--environment-setup)
- [Annotation Tool](#annotation-tool)
  - [Semi-Automatic Annotation (`annotator.py`)](#semi-automatic-annotation-annotatorpy)
  - [Fully Manual Annotation (`object_completion_annotator.py`)](#fully-manual-annotation-object_completion_annotatorpy)
  - [Merging Per-Scene Annotations (`combine_scene_annos.py`)](#merging-per-scene-annotations-combine_scene_annospy)
- [Annotation Workflow](#annotation-workflow)
- [License](#license)

---

## Overview

The S2C-Completion dataset augments Scan2CAD annotations with a `scannet_instance_id` field that links each aligned CAD model to its corresponding segmented instance in the ScanNet point cloud. This enables tasks such as **3D object completion**, where a partial scan of an object must be completed using a CAD model as a reference, and object 9-DoF pose estimation.

The annotation toolkit supports two modes:

| Mode | Script | Strategy |
|------|--------|----------|
| Semi-automatic | `annotator.py` | Overlap-based auto-matching with manual fallback |
| Fully manual | `object_completion_annotator.py` | Interactive multi-candidate selection via colored visualization |

---

## Dataset Structure

The main dataset file is `S2CCompletion_annotations.json`. It is a JSON dictionary keyed by **ScanNet scene ID** (e.g., `"scene0000_00"`).

```
S2CCompletion_annotations.json
└── scene0000_00
    ├── id_scan              # ScanNet scene ID
    ├── id_alignment         # Scan2CAD alignment ID
    ├── trs                  # Scene-level S2C transformation
    │   ├── translation      # [x, y, z]
    │   ├── rotation         # [qw, qx, qy, qz] quaternion
    │   └── scale            # [sx, sy, sz]
    ├── n_aligned_models     # Number of CAD models in this scene
    └── aligned_models       # List of CAD model annotation objects
        └── [0..N]
            ├── id_cad            # ShapeNet model ID
            ├── catid_cad         # ShapeNet category ID
            ├── sym               # Symmetry type (e.g. "__SYM_ROTATE_UP_2")
            ├── trs               # Model-level transformation (same format as scene trs)
            ├── bbox              # Oriented bounding box dimensions [dx, dy, dz]
            ├── center            # Bounding box center [x, y, z]
            ├── keypoints_cad     # Keypoints on CAD model
            │   ├── n_keypoints
            │   └── position      # Flattened list of 3D points [x0,y0,z0, x1,y1,z1, ...]
            ├── keypoints_scan    # Corresponding keypoints on scan
            │   ├── n_keypoints
            │   └── position
            └── scannet_instance_id   # ← S2C-Completion key field
                                      #   >= 0 : matched ScanNet instance ID
                                      #   -1   : no match found
```

### Top-level Fields

| Field | Type | Description |
|-------|------|-------------|
| `id_scan` | `string` | ScanNet scene identifier (e.g. `"scene0000_00"`) |
| `id_alignment` | `string` | Scan2CAD alignment identifier |
| `trs` | `object` | Scene-to-CAD coordinate transformation |
| `n_aligned_models` | `int` | Number of aligned CAD models in the scene |
| `aligned_models` | `list` | Per-model annotation entries (see below) |

### Per-Model Fields

| Field | Type | Description |
|-------|------|-------------|
| `id_cad` | `string` | ShapeNet model ID |
| `catid_cad` | `string` | ShapeNet category ID |
| `sym` | `string` | Rotational symmetry type of the object |
| `trs` | `object` | 6-DoF transformation (translation, quaternion rotation, scale) |
| `bbox` | `[float]` | Oriented bounding box half-extents |
| `center` | `[float]` | Bounding box center in scene coordinates |
| `keypoints_cad` | `object` | Sparse 3D keypoints on the CAD model |
| `keypoints_scan` | `object` | Corresponding keypoints on the scan |
| `scannet_instance_id` | `int` | **S2C-Completion annotation**: ScanNet instance index (`-1` = unmatched) |

---

## Prerequisites & Environment Setup

The annotation scripts require the following external datasets and Python packages.

### Required Datasets

- **ScanNet v2** — RGB-D scans with instance segmentation labels
- **ShapeNet** — 3D CAD model repository
- **Scan2CAD** — CAD model alignments to ScanNet scenes (`scan2cad_v2_annotations.json`)

### Python Dependencies

Install the required packages via pip:

```bash
pip install open3d trimesh torch numpy tqdm
```

---

## Annotation Tool

### Semi-Automatic Annotation (`annotator.py`)

This script automatically matches CAD models to ScanNet instances using voxel-overlap scoring, with a manual vetting step for ambiguous cases.

**How it works:**

1. Loads Scan2CAD alignments and ShapeNet taxonomy.
2. For each unannotated scene:
   - Loads the ScanNet point cloud and instance segmentation labels.
   - Aligns the CAD models into scene coordinates using Scan2CAD transformations.
   - For each CAD model, finds the **5 nearest ScanNet instances** (by center distance) of the matching category.
   - Computes voxel overlap between the CAD model bounding box and each candidate instance.
3. Applies automatic decisions based on overlap thresholds:

   | Overlap | Action |
   |---------|--------|
   | ≥ 0.7 | **Auto-accept** — instance is matched automatically |
   | < 0.05 | **Auto-discard** — no match assigned (`-1`) |
   | 0.05 – 0.7 | **Manual review** — opens interactive Open3D window |

4. In the manual review window:
   - **W** — Accept the current candidate as a match
   - **R** — Reject and move to the next candidate

**Output:** Per-scene JSON files saved to `obj_completion_per_scene/` with the naming pattern `s2c_v2_completion_scene{XXXX}_{YY}.json`.

**Run:**
```bash
python annotator.py
```

---

### Fully Manual Annotation (`object_completion_annotator.py`)

This script provides a richer interactive annotation experience by displaying **up to 6 candidate instances simultaneously**, each rendered in a distinct color, allowing annotators to select the best match with a single keypress.

**How it works:**

1. Follows the same scene loading and alignment pipeline as `annotator.py`.
2. For each CAD model, finds the **6 nearest ScanNet instances**.
3. Opens an Open3D visualization window showing:
   - The CAD model bounding box in **black**
   - Up to 6 candidate ScanNet instances color-coded as:

     | Key | Color | Instance |
     |-----|-------|----------|
     | `1` | Red | Candidate 1 |
     | `2` | Green | Candidate 2 |
     | `3` | Blue | Candidate 3 |
     | `4` | Orange | Candidate 4 |
     | `5` | Teal | Candidate 5 |
     | `6` | Pink | Candidate 6 |
     | `R` | — | Reject (no match) |


**Output:** Same format and location as `annotator.py` — per-scene files in `obj_completion_per_scene/`.

**Run:**
```bash
python object_completion_annotator.py
```

---

### Merging Per-Scene Annotations (`combine_scene_annos.py`)

After annotating scenes with either script above, run this script to merge all per-scene JSON files into the final consolidated dataset file.

**Run:**
```bash
python combine_scene_annos.py
```

---

## Citation

If you use this dataset or code please cite:

```
@inproceedings{kumar2026caoa,
  title={CAOA-Completion-Assisted Object-CAD Alignment},
  author={Kumar, Hiranya Garbha and Kamal, Minhas and Prabhakaran, Balakrishnan},
  booktitle={Thirteenth International Conference on 3D Vision},
  year={2026},
  organization={IEEE}
}
```
