# Implementation Options for `get_piercing_counts`

Since heuristics (OpenCV/color-based) have failed, we analyzed three viable technical paths for the detection logic.

| Method | Core Technology | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **1. Heuristics** (Geometry/Color) | ROI + Thresholding | Extremely fast; no extra libraries. | **FAILED**. Highly sensitive to lighting and shadows. Unreliable. |
| **2. YOLO** (Supervised Detection) | Custom trained CNN | Highest accuracy in production; extremely fast inference once trained. | Requires massive labeled dataset (to be solved via ComfyUI automation). |
| **3. Zero-Shot** (Semantic/Prompting) | Grounding DINO / OWL-ViT | No training or labeling required. Uses natural language ("silver piercing") to find objects. | Slower inference; requires downloading large pre-trained weights. |
| **4. CLIP-based** (Semantic Similarity)| Vision-Language (CLIP) | Conceptually robust; very easy to implement via similarity scores between image crops and text prompts. | Requires landmark detection first to crop the correct areas properly. |

## Strategic Roadmap & Preferred Solution

The goal is to build a high-performance, automated production pipeline using synthetic data.

### Phase 1: The "Automated Production" Backend (ComfyUI $\rightarrow$ YOLO)
Our primary priority is building an end-to-end automation loop to generate training data without manual intervention:
1.  **Synthetic Data Generation**: Use **ComfyUI** workflows to generate images with specific, controlled piercing placements.
2.  **Automated Labeling**: Extract metadata/coordinates directly from the ComfyUI generation process and automatically convert them into YOLO `.txt` format labels.
3.  **Model Training**: Train a dedicated, lightweight YOLO model on this synthetic dataset.
4.  **Integration**: Deploy the trained YOLO model as the primary detector in the pipeline.

### Phase 2: The "Fallback/Benchmark" Backend (Zero-Shot / CLIP)
If the automated YOLO pipeline proves too complex to implement immediately or requires more data than available, we will transition to semantic methods:
1.  Use **Zero-Shot (Grounding DINO)** or **CLIP** to provide a functional detector.
2.  This serves as a "smart" fallback that works immediately without training but at the cost of higher computational requirements and slower inference.

---

*This document serves as a decision matrix and long-term roadmap.*