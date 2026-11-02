# Usage Example: Piercing Detection Benchmark

## Prerequisites

Before running the piercing detection benchmark, ensure you have:

1. Python 3.8+ installed
2. Required dependencies (opencv-python, numpy)
3. A directory with face images to test
4. The specification file at `src/dataset_filter/specs/smlq_sfw.json`

## Running the Script

### Basic Usage

```bash
# Run against a directory of face images
python tests/piercing_benchmark.py data/ideal_faces
```

### With Custom Specification

```bash
# Use a custom specification file
python tests/piercing_benchmark.py data/ideal_faces --spec-file src/dataset_filter/specs/smlq_sfw.json
```

## Expected Output

When run, the script will produce output similar to:

```
Loading specification...
Found 5 image files
Processing images...
================================================================================
PIERCING DETECTION BENCHMARK REPORT
================================================================================
Character ID: smlq_sfw
Piercing Requirements:
  RIGHT_EYEBROW: 2 x silver_hoop
  LEFT_EYEBROW: 0 x silver_hoop
  LEFT_NOSTRIL: 1 x silver_hoop
  RIGHT_NOSTRIL: 0 x silver_hoop

File Name                      Status     Error Details
--------------------------------------------------------------------------------
match39_comfyui_00187__m39.png PASS
match34_comfyui_00032__m34.png PASS
match33_smlq_sfw_04__m33.png   FAIL       [FAIL] LEFT_NOSTRIL: expected 1, found 0
match34_comfyui_00250__m34.png FAIL       [FAIL] RIGHT_EYEBROW: expected 2, found 3 (extra)

--------------------------------------------------------------------------------
SUMMARY
--------------------------------------------------------------------------------
Total Files: 4
PASS: 2
FAIL: 2
Accuracy: 50.0%

DETAILED FAILURES:
match33_smlq_sfw_04__m33.png:
  [FAIL] LEFT_NOSTRIL: expected 1, found 0
match34_comfyui_00250__m34.png:
  [FAIL] RIGHT_EYEBROW: expected 2, found 3 (extra)
```

## How It Works

1. **Specification Loading**: Loads requirements from `src/dataset_filter/specs/smlq_sfw.json`
2. **Image Processing**: For each image in the directory:
   - Calls `get_piercing_counts(image_path)` function
   - Compares results with specification requirements
3. **Reporting**: Generates:
   - Console table with file status
   - Statistical summary (accuracy percentage)
   - Detailed error information for failures

## Current Status

The current implementation is a **placeholder**. The actual piercing detection logic (`get_piercing_counts`) is not yet implemented and returns mocked data. The real implementation would require:

- Face detection to locate faces in images
- Piercing detection using computer vision or ML models
- Landmark detection for precise piercing location identification
- Accurate counting and classification of piercings

## Next Steps for Implementation

1. Implement actual face detection using OpenCV or similar
2. Add piercing detection using YOLO models or CLIP-based approaches
3. Integrate landmark detection for accurate positioning
4. Replace mock `get_piercing_counts` function with real logic