# YOLO Synthetic Data Generator

This module provides a tool for generating synthetic training datasets for YOLO object detection. Instead of manual labeling, it uses **Synthetic Data Injection** to create thousands of perfectly labeled images.

## How it works

The generator takes "clean" face images and "accessory" assets (PNGs with alpha channels) and performs the following steps:

1.  **Landmark Detection**: Uses `face_recognition` to find facial landmarks.
2.  **ROI Calculation**: Uses the project's `LandmarkProvider` to identify precise target areas (e.g., earlobes, nostrils).
3.  **Asset Transformation**: 
    *   **Scaling**: Adjusts accessory size relative to the face.
    *   **Rotation**: Randomly rotates the asset to simulate different angles.
    *   **Jittering**: Slightly offsets the placement to prevent overfitting to exact coordinates.
4.  **Alpha Blending**: Seamlessly blends the transparent PNG onto the face.
5.  **Auto-Labeling**: Automatically generates YOLO-format `.txt` files with normalized bounding box coordinates.

## Requirements

*   `opencv-python`
*   `numpy`
*   `face-recognition`
*   `dataset_filter` (the main project) must be in your `PYTHONPATH`.

## Usage

### 1. Prepare Assets
Place your clean face images in one folder and your transparent PNG accessories in another.

```bash
mkdir -p data/clean_faces data/assets data/synthetic_output
```

### 2. Run the Generator

Run the script via Python, pointing to your directories:

```bash
python src/yolo/generator.py \
    --faces data/clean_faces \
    --assets data/assets \
    --output data/synthetic_output \
    --iterations 500
```

### Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--faces` | Path to directory containing clean face images (`.png`, `.jpg`) | *Required* |
| `--assets` | Path to directory containing accessory PNGs (must have alpha channel) | *Required* |
| `--output` | Directory where generated images and labels will be saved | *Required* |
| `--iterations` | Number of synthetic images to generate | `100` |

## Output Structure

The output directory will follow the standard YOLO dataset format:

```text
synthetic_output/
├── images/
│   ├── synth_00000.png
│   └── ...
└── labels/
    ├── synth_00000.txt
    └── ...
```

## Pro-Tips for Better Data

* **Diversity is key**: Use a wide variety of accessory shapes and colors.
* **Alpha Channels**: Ensure your assets are high-quality PNGs with clean transparency.
* **Scale & Rotation**: You can tweak `min_scale`, `max_scale`, and `rotation_range` in the `SyntheticDataGenerator` constructor to increase data variance.
