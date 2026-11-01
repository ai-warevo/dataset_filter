# Project Context: Dataset Face Filter

## Objective
Automated cleaning and filtering of image datasets used for training neural networks (specifically Stable Diffusion/ComfyUI outputs). The tool verifies if generated images conform to a specific character profile regarding facial piercings.

## Current Implementation (Status Quo)
The system currently operates by performing scoring through `face_recognition`. Based on a coefficient defined via an environment variable, images are either approved or rejected (bracked). 

**Note**: Facial piercing recognition is a new feature that is currently under development and not yet functional.

## Data Generation Source
Images are generated using **ComfyUI** workflows. This is a critical part of the pipeline, as the generation process itself can be leveraged to automate data labeling/annotation for model training.

## Target Workflow (Future Goal)
1.  Load master requirements (e.g., "2 silver hoops on right eyebrow").
2.  Scan a target directory of images.
3.  Detect piercings on each image.
4.  Compare counts and locations with the master profile.
5.  Generate a report:
    - `PASS`: Image perfectly matches the spec.
    - `FAIL`: Mismatch found (e.g., "Expected 2, found 1" or "Extra piercing detected").

---

*This document is for session resumption.*