# Technical Status & Problem Statement

## Current State
- The project is in a transition phase from "experimental scripts" to a structured DDD/SOLID production-grade pipeline.
- **Working Component**: Image filtering currently proceeds via `face_recognition` scoring, where an environment variable coefficient determines if an image is approved or rejected.
- **Non-functional Component**: Piercing detection (micro-artifact detection) is currently non-operational and requires a complete implementation.

## Problems Identified
1. **Failed Heuristics**: Any previous attempts to use color/edge math for piercing detection have been discarded as unreliable.
2. **Placeholder YOLO**: The current YOLO implementation is a stub.
3. **Missing Core Logic**: `get_piercing_counts(image_path) -> dict` does not exist in a working form.

## Decisions Made
- **Discard** all existing heuristic-based logic.
- **Build Standalone**: Create a robust, standalone piercing detector (using Zero-Shot/CLIP) as part of a benchmark script to provide immediate value.

---

*This document captures the history of failed approaches and current system limitations.*