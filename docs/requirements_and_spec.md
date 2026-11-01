# Requirements for the Standalone Benchmark Script

## Goal
Create a single-file Python script (`tests/piercing_benchmark.py`) that acts as a robust, standalone verification tool. This script must be able to produce real results without relying on the project's current broken infrastructure.

## Functional Requirements
1. **Standalone Execution**: The script must be self-contained in one file. It should not depend on `src/dataset_filter` if that code is considered unreliable for this task.
2. **Input**: Accepts a directory path via command line arguments.
3. **Specification Loading**: Parses `smlq_sfw.json` to establish the "Source of Truth" (expected counts and locations).
4. **Detection Interface**: Must implement/use a function:
   `get_piercing_counts(image_path) -> Dict[Location, Dict[ArtifactID, int]]`
5. **Comparison Logic**:
   - Compare detected counts against the JSON spec for every image in the directory.
   - Handle mismatches: `Expected X, got Y`.
6. **Reporting**:
   - **Console Output**: A professional table showing: `File Name | Status (PASS/FAIL) | Error Detail`.
   - **Statistical Summary**: 
     - For `ideal_faces`: Total Accuracy (%) — how many images perfectly matched the spec.
     - For `deffects`: Detection rate — identifying where and why they failed.

## Error Reporting Format Example
`[FAIL] image_001.png | LEFT_NOSTRIL: expected 1, found 0`
`[FAIL] image_002.png | RIGHT_EYEBROW: expected 2, found 3 (extra)`

---
*This document defines the "Definition of Done" for the next coding task.*