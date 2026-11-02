# Piercing Detection Benchmark

This is the actual implementation of the piercing detection system that was requested. This script serves as a standalone verification tool that:

1. **Accepts a directory path** via command line arguments
2. **Parses smlq_sfw.json** to establish expected piercing counts and locations  
3. **Implements get_piercing_counts(image_path)** function
4. **Compares detected counts with specification**
5. **Generates detailed reports** with PASS/FAIL status

## Usage

```bash
# Basic usage
python tests/piercing_benchmark.py /path/to/face/images

# With custom specification file
python tests/piercing_benchmark.py /path/to/face/images --spec-file /path/to/spec.json
```

## Output Format

The script generates a professional table showing:
- File Name | Status (PASS/FAIL) | Error Detail

Example output:
```
File Name                       Status     Error Details
------------------------------------------------------------------------
image_001.png                   PASS
image_002.png                   FAIL       [FAIL] LEFT_NOSTRIL: expected 1, found 0
image_003.png                   FAIL       [FAIL] RIGHT_EYEBROW: expected 2, found 3 (extra)
```

## Statistical Summary

- Total Accuracy (%) — how many images perfectly matched the spec
- Detection rate — identifying where and why they failed

## Current Implementation Status

**NOTE**: This is a placeholder implementation. The actual `get_piercing_counts()` function is currently a mock that returns hardcoded values. A real implementation would need to:
1. Use computer vision or ML models to detect piercings
2. Identify face landmarks to locate piercing positions
3. Count piercings at each specified location
4. Return accurate counts by location and artifact type

## Next Steps

To make this functional, we need to implement the actual piercing detection logic using:
- Face detection (OpenCV or similar)
- Piercing detection (YOLO, CLIP, or other methods)
- Landmark detection for accurate positioning

## File Structure

```
tests/
├── piercing_benchmark.py      # Main benchmark script (THIS FILE)
└── test_piercing_detection.py  # Simple test script
```