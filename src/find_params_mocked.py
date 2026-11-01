import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any

from dataset_filter.domain.entities import Location, CharacterSpecification
from dataset_filter.infrastructure.detectors.landmark_provider import LandmarkProvider
from dataset_filter.infrastructure.validators.heuristic_metal_validator import HeuristicMetalValidator

class MockLandmarkProvider(LandmarkProvider):
    def get_roi_for_location(self, landmarks: Any, location: Location, image: np.ndarray) -> tuple[int, int, int, int, float]:
        h, w = image.shape[:2]
        # Return a fixed ROI in the upper half of the image for eyebrows
        if location in [Location.RIGHT_EYEBROW, Location.LEFT_EYEBROW]:
            return (w // 4, h // 4, 3 * w // 4, h // 2, 0.0)
        # Return a fixed ROI in the middle for nostrils
        elif location in [Location.LEFT_NOSTRIL, Location.RIGHT_NOSTRIL]:
            return (w // 3, h // 2, 2 * w // 3, 3 * h // 4, 0.0)
        return (0, 0, 0, 0, 0.0)

    def calculate_interocular_distance(self, landmarks: Any) -> float:
        return 50.0

def load_spec(spec_path: str) -> Dict[str, Any]:
    with open(spec_path, 'r') as f:
        return json.load(f)

def optimize(spec_path: str, folders: List[str]):
    spec_dict = load_spec(spec_path)
    spec = CharacterSpecification.from_json(json.dumps(spec_dict))
    landmark_provider = MockLandmarkProvider()

    test_cases = []
    image_extensions = ('.png', '.jpg', '.jpeg', '.webp')

    for folder_str in folders:
        folder = Path(folder_str)
        if not folder.exists():
            print(f"⚠️  Folder not found: {folder_str}")
            continue
            
        print(f"🔍 Scanning folder: {folder}")
        all_images = sorted([p for p in folder.iterdir() if p.suffix.lower() in image_extensions])

        for img_path in all_images:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
                
            landmarks = {} 
            path_str = str(img_path).lower()
            is_pierced = any(x in path_str for x in ["piercing", "synth"])
            is_clean = not is_pierced

            test_cases.append({
                "path": str(img_path),
                "image": image,
                "landmarks": landmarks,
                "is_pierced": is_pierced,
                "is_clean": is_clean
            })

    if not test_cases:
        print("❌ No valid test cases found.")
        return

    # Grid Search
    contrast_multipliers = [1.0, 1.1, 1.2, 1.3, 1.5]
    std_multipliers = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    brightness_floors = [0, 50, 100, 150]
    kernels = [3, 5, 7]
    min_areas = [1, 5, 10]

    combinations = []
    for cm in contrast_multipliers:
        for sm in std_multipliers:
            for bf in brightness_floors:
                for k in kernels:
                    for a in min_areas:
                        combinations.append((cm, sm, bf, k, a))

    tp_counts = [0] * len(combinations)
    fp_counts = [0] * len(combinations)
    tn_counts = [0] * len(combinations)
    fn_counts = [0] * len(combinations)
    labeled_total = len(test_cases)

    print(f"✅ Found {len(test_cases)} valid test cases ({labeled_total} labeled). Starting optimization ({len(combinations)} combinations)...\n")

    for i, case in enumerate(test_cases):
        img_idx = i + 1
        img_name = Path(case["path"]).name
        print(f"{img_idx}/{len(test_cases)} | {img_name} | testing...", end='\r')
        
        img = case["image"]
        landmarks = case["landmarks"]
        is_pierced = case["is_pierced"]
        is_clean = case["is_clean"]

        for idx, (cm, sm, bf, k, a) in enumerate(combinations):
            params = {
                "contrast_multiplier": cm,
                "brightness_std_multiplier": sm,
                "min_brightness_floor": bf,
                "kernel_size": k,
                "min_contour_area": a
            }
            validator = HeuristicMetalValidator(landmark_provider, params=params)
            result = validator.validate(img, landmarks, spec)
            
            if result.is_valid: 
                if is_pierced: fn_counts[idx] += 1
                elif is_clean: tn_counts[idx] += 1
            else: 
                if is_pierced: tp_counts[idx] += 1
                elif is_clean: fp_counts[idx] += 1

    print("-" * 60)
    results_summary = []
    for idx, (cm, sm, bf, k, a) in enumerate(combinations):
        tp, fp, tn, fn = tp_counts[idx], fp_counts[idx], tn_counts[idx], fn_counts[idx]
        accuracy = (tp + tn) / labeled_total if labeled_total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        results_summary.append({
            "params": (cm, sm, bf, k, a),
            "accuracy": accuracy, "precision": precision, "recall": recall,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn
        })

    results_summary.sort(key=lambda x: (x["accuracy"], x["precision"]), reverse=True)
    best_res = results_summary[0]

    if best_res["accuracy"] == 1.0:
        print(f"🏆 PERFECT PARAMETERS FOUND ({best_res['accuracy']*100:.1f}%):")
    else:
        print(f"🥈 BEST ATTEMPT FOUND ({best_res['accuracy']*100:.1f}% accuracy - {best_res['tp'] + best_res['tn']}/{labeled_total}):")

    for i in range(min(3, len(results_summary))):
        res = results_summary[i]
        cm, sm, bf, k, a = res["params"]
        prefix = "⭐ " if i == 0 else "  "
        print(f"{prefix}cm={cm}, std_mult={sm}, floor={bf}, kernel={k}x{k}, area={a} | Acc: {res['accuracy']:.2%} | Prec: {res['precision']:.2%} | Rec: {res['recall']:.2%}")

if __name__ == "__main__":
    SPEC_PATH = "src/dataset_filter/specs/smlq_sfw.json"
    FOLDERS = ["data/assets", "data/clean_faces"]
    optimize(SPEC_PATH, FOLDERS)
