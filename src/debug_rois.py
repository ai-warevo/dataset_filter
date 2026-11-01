"""
ROI Debugger for Heuristic Metal Validator.

This script visualizes the Regions of Interest (ROIs) calculated by 
the LandmarkProvider to ensure they are correctly aligned with facial features.
"""
import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Any

from dataset_filter.domain.entities import Location
from dataset_filter.infrastructure.detectors.landmark_provider import LandmarkProvider
import face_recognition

def load_spec(spec_path: str) -> Dict[str, Any]:
    with open(spec_path, 'r') as f:
        return json.load(f)

def debug_rois(spec_path: str, folder_path: str, output_dir: str, limit: int = 10):
    spec = load_spec(spec_path)
    requirements = spec["piercing_requirements"]
    req_map = {r["expected_location"]: r["expected_count"] for r in requirements}

    landmark_provider = LandmarkProvider()
    
    # Use os.makedirs to avoid pathlib's FileExistsError on some systems
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        print(f"⚠️  Warning: {output_dir} exists but is a file. Deleting it to create a directory.")
        os.remove(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir)

    folder = Path(folder_path)
    image_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    all_images = sorted([p for p in folder.iterdir() if p.suffix.lower() in image_extensions])
    
    print(f"🔍 Scanning {len(all_images)} images in {folder}")
    print(f"📂 Saving debug images to {output_dir}\n")

    count = 0
    for img_path in all_images:
        if count >= limit:
            break

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # Get landmarks using face_recognition to match LandmarkProvider's dict expectation
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_landmarks_list = face_recognition.face_landmarks(rgb_image)
        
        if not face_landmarks_list:
            print(f"  [!] No face found in {img_path.name}. Skipping.")
            continue

        # We assume one face per image for the gold set
        landmarks = face_landmarks_list[0]
        
        # Create a copy for annotation
        annotated_img = image.copy()
        found_any_roi = False
        
        print(f"  [+] Processing {img_path.name}...")

        for loc_name in req_map.keys():
            try:
                loc_enum = Location[loc_name]
                roi_data = landmark_provider.get_roi_for_location(landmarks, loc_enum, image)
                # roi_data is (x1, y1, x2, y2, angle)
                x1, y1, x2, y2 = map(int, roi_data[:4])
                
                # Draw rectangle on annotated image
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_img, loc_name, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Save individual crop
                crop = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
                if crop.size > 0:
                    crop_name = f"{img_path.stem}_{loc_name}.jpg"
                    cv2.imwrite(str(out_path / crop_name), crop)
                    found_any_roi = True

            except Exception as e:
                print(f"    [!] Error calculating ROI for {loc_name}: {e}")

        if found_any_roi:
            # Save the annotated full image
            annotated_name = f"{img_path.stem}_annotated.jpg"
            cv2.imwrite(str(out_path / annotated_name), annotated_img)
            count += 1

    print(f"\n✅ Done! Processed {count} images. Check the '{output_dir}' folder.")

if __name__ == "__main__":
    # --- USER CONFIGURATION AREA ---
    SPEC_PATH = "src/dataset_filter/specs/smlq_sfw.json"
    GOLD_FOLDER = "/app/data/smlq_sfw" 
    OUTPUT_DIR = "data/debug"
    LIMIT = 20  # Number of images to debug
    # -------------------------------

    debug_rois(SPEC_PATH, GOLD_FOLDER, OUTPUT_DIR, LIMIT)
