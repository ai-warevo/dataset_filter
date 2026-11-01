from __future__ import annotations
"""
Parameter Optimizer for Heuristic Metal Validator (Mocked Landmarks).

This script performs a grid search to find the most robust 
parameters for micro-feature validation using mocked landmarks.
"""
import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any

from dataset_filter.domain.entities import Location, CharacterSpecification
from dataset_filter.infrastructure.detectors.landmark_provider import LandmarkProvider
from dataset_filter.infrastructure.validators.heuristic_metal_validator import HeuristicMetalValidator
from dataset_filter.domain.models.filter_config import HeuristicConfig

class MockLandmarkProvider(LandmarkProvider):
    """Returns the whole image as the ROI to allow optimization on crops or full images."""
    def get_roi_for_location(self, landmarks: Any, location: Location, image: np.ndarray) -> tuple[int, int, int, int, float]:
        h, w = image.shape[:2]
        return (0, 0, w, h, 0.0)

    def calculate_interocular_distance(self, landmarks: Any) -> float:
        return 50.0

def load_spec(spec_path: str) -> Dict[str, Any]:
    """Loads a JSON character specification file."""
    with open(spec_path, 'r') as f:
        return json.load(f)

def optimize(spec_path: str, pierced_folder: str, clean_folder: str):
    """
    Performs a grid search over parameter combinations to maximize 
    detection of piercings (Recall) and minimize false positives (Precision).
    """
    spec_dict = load_spec(spec_path)
    # Reconstruct spec object to use its built-in validation/parsing logic
    spec = CharacterSpecification.from_json(json.dumps(spec_dict))
    
    landmark_provider = MockLandmarkProvider()
    
    # 1. Collect all valid test cases from the folders
    test_cases = []
    image_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    
    def add_to_test_cases(folder_path: str, is_pierced: bool, is_clean: bool):
        folder = Path(folder_path)
        if not folder.exists():
            print(f"⚠️  Warning: Folder {folder_path} does not exist.")
            return
        
        all_images = sorted([p for p in folder.iterdir() if p.suffix.lower() in image_extensions])
        for img_path in all_images:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Use empty dict as landmarks; MockLandmarkProvider will return full ROI
            test_cases.append({
                "path": str(img_path),
                "image": image,
                "landmarks": {}, 
                "is_pierced": is_pierced,
                "is_clean": is_clean
            })

    print(f"🔍 Scanning pierced folder: {pierced_folder}")
    add_to_test_cases(pierced_folder, is_pierced=True, is_clean=False)
    
    print(f"🔍 Scanning clean folder: {clean_folder}")
    add_to_test_cases(clean_folder, is_pierced=False, is_clean=True)

    if not test_cases:
        print("❌ No valid test cases found.")
        return

    # 2. Grid Search for Adaptive Parameters
    std_multipliers = [1.5, 2.0, 2.5, 3.0, 3.5]
    brightness_floors = [80, 100, 120, 140, 160]
    kernels = [3, 5, 7]
    min_areas = [1, 5, 10]

    combinations = []
    for sm in std_multipliers:
        for bf in brightness_floors:
            for k in kernels:
                for a in min_areas:
                    combinations.append((sm, bf, k, a))

    tp_counts = [0] * len(combinations)
    fp_counts = [0] * len(combinations)
    tn_counts = [0] * len(combinations)
    fn_counts = [0] * len(combinations)
    
    labeled_total = len(test_cases)

    print(f"✅ Found {len(test_cases)} valid test cases ({len([t for t in test_cases if t['is_pierced']])} pierced, {len([t for t in test_cases if t['is_clean']])} clean). Starting optimization ({len(combinations)} combinations)...\n")

    for i, case in enumerate(test_cases):
        img_idx = i + 1
        img_name = Path(case["path"]).name
        print(f"{img_idx}/{len(test_cases)} | {img_name} | testing...", end='\r')
        
        img = case["image"]
        landmarks = case["landmarks"]
        is_pierced = case["is_pierced"]
        is_clean = case["is_clean"]

        for idx, (sm, bf, k, a) in enumerate(combinations):
            params = HeuristicConfig(
                brightness_std_multiplier=sm,
                min_brightness_floor=bf,
                kernel_size=k,
                min_contour_area=a
            )
            validator = HeuristicMetalValidator(landmark_provider, params=params)
            result = validator.validate(img, landmarks, spec)
            
            if result.is_valid: # Passed filter (no defect detected)
                if is_pierced:
                    fn_counts[idx] += 1 # Missed the piercing
                elif is_clean:
                    tn_counts[idx] += 1
            else: # Failed filter (defect detected)
                if is_pierced:
                    tp_counts[idx] += 1 # Correctly found the piercing
                elif is_clean:
                    fp_counts[idx] += 1

    print("-" * 60)
    
    results_summary = []
    for idx, (sm, bf, k, a) in enumerate(combinations):
        tp = tp_counts[idx]
        fp = fp_counts[idx]
        tn = tn_counts[idx]
        fn = fn_counts[idx]
        
        accuracy = (tp + tn) / labeled_total if labeled_total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        results_summary.append({
            "params": (sm, bf, k, a),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn
        })

    if not results_summary:
        print("❌ No combinations to evaluate.")
        return

    # Sort by Recall desc (most important), then Precision desc
    results_summary.sort(key=lambda x: (x["recall"], x["precision"]), reverse=True)
    best_res = results_summary[0]

    if best_res["recall"] == 1.0:
        print(f"🏆 PERFECT RECALL FOUND ({best_res['recall']*100:.1f}%):")
    else:
        print(f"🥈 BEST ATTEMPT FOUND ({best_res['recall']*100:.1f}% recall - {best_res['tp']}/{len([t for t in test_cases if t['is_pierced']])} detected):")

    for i in range(min(3, len(results_summary))):
        res = results_summary[i]
        sm, bf, k, a = res["params"]
        prefix = "⭐ " if i == 0 else "  "
        print(f"{prefix}std_multiplier={sm}, min_floor={bf}, kernel={k}x{k}, min_area={a} | Rec: {res['recall']:.2%} | Prec: {res['precision']:.2%} | Acc: {res['accuracy']:.2%}")

    if best_res["recall"] == 0 and labeled_total > 0:
        print("❌ No parameters found that detect any piercings.")

if __name__ == "__main__":
    # --- USER CONFIGURATION AREA ---
    SPEC_PATH = "src/dataset_filter/specs/smlq_sfw.json"
    PIERCED_FOLDER = "data/assets"
    CLEAN_FOLDER = "data/clean_faces"
    # -------------------------------

    optimize(SPEC_PATH, PIERCED_FOLDER, CLEAN_FOLDER)