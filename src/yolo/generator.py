import os
import cv2
import numpy as np
import random
from pathlib import Path
import argparse
from typing import Dict, List, Tuple

# Add src to sys.path to allow importing from dataset_filter
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataset_filter.infrastructure.detectors.landmark_provider import LandmarkProvider
from dataset_filter.domain.entities import Location

class SyntheticDataGenerator:
    """Generates synthetic training data by injecting assets onto clean faces."""

    def __init__(
        self,
        faces_dir: Path,
        assets_dir: Path,
        output_dir: Path,
        location_to_class: Dict[Location, int],
        min_scale: float = 0.5,
        max_scale: float = 1.5,
        rotation_range: float = 15.0,
    ) -> None:
        self.faces_dir = faces_dir
        self.assets_dir = assets_dir
        self.output_dir = output_dir
        self.location_to_class = location_to_class
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.rotation_range = rotation_range
        
        self.landmark_provider = LandmarkProvider()
        
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels").mkdir(parents=True, exist_ok=True)

    def _apply_transformations(
        self, 
        asset: np.ndarray, 
        target_size: Tuple[int, int], 
        rotation: float, 
        scale: float
    ) -> np.ndarray:
        h, w = asset.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        asset = cv2.resize(asset, (new_w, new_h), interpolation=cv2.INTER_AREA)
        center = (new_w // 2, new_h // 2)
        matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
        asset = cv2.warpAffine(asset, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        return cv2.resize(asset, target_size, interpolation=cv2.INTER_AREA)

    def _blend_asset(self, face: np.ndarray, asset: np.ndarray, anchor: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        h_a, w_a = asset.shape[:2]
        ax, ay = anchor
        x1, y1 = ax - w_a // 2, ay - h_a // 2
        x2, y2 = x1 + w_a, y1 + h_a

        fh, fw = face.shape[:2]
        real_x1, real_y1 = max(0, x1), max(0, y1)
        real_x2, real_y2 = min(fw, x2), min(fh, y2)

        if real_x1 >= real_x2 or real_y1 >= real_y2: return face, (0, 0, 0, 0)

        asset_x1, asset_y1 = real_x1 - x1, real_y1 - y1
        asset_x2, asset_y2 = asset_x1 + (real_x2 - real_x1), asset_y1 + (real_y2 - real_y1)
        asset_slice = asset[asset_y1:asset_y2, asset_x1:asset_x2]
        
        if asset_slice.shape[2] == 4:
            alpha = asset_slice[:, :, 3] / 255.0
            color = asset_slice[:, :, :3]
        else:
            alpha = np.ones(asset_slice.shape[:2], dtype=float)
            color = asset_slice
        
        for c in range(3):
            face[real_y1:real_y2, real_x1:real_x2, c] = (1.0 - alpha) * face[real_y1:real_y2, real_x1:real_x2, c] + alpha * color[:, :, c]

        xc = ((real_x1 + real_x2) / 2.0) / fw
        yc = ((real_y1 + real_y2) / 2.0) / fh
        bw = (real_x2 - real_x1) / fw
        bh = (real_y2 - real_y1) / fh
        return face, (xc, yc, bw, bh)

    def generate(self, iterations: int) -> None:
        face_files = list(self.faces_dir.glob("*.png")) + list(self.faces_dir.glob("*.jpg"))
        asset_files = list(self.assets_dir.glob("*.png"))
        if not face_files or not asset_files: raise ValueError("Missing faces or assets!")

        print(f"🚀 Generating {iterations} images...")
        for i in range(iterations):
            try:
                face = cv2.imread(str(random.choice(face_files)))
                asset = cv2.imread(str(random.choice(asset_files)), cv2.IMREAD_UNCHANGED)
                if face is None or asset is None or asset.shape[2] != 4: continue

                loc = random.choice(list(self.location_to_class.keys()))
                import face_recognition
                landmarks_list = face_recognition.face_landmarks(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                if not landmarks_list: continue
                
                lms = landmarks_list[0]
                arr = np.zeros((69, 2))
                mapping = {
                    'chin': list(range(0, 17)),
                    'left_eyebrow': list(range(17, 22)),
                    'right_eyebrow': list(range(22, 27)),
                    'nose_bridge': list(range(27, 36)),
                    'nose_tip': list(range(35, 42)),
                    'left_eye': list(range(42, 48)),
                    'right_eye': list(range(48, 54)),
                    'top_lip': list(range(54, 60)),
                    'bottom_lip': list(range(60, 68))
                }
                for k, idxs in mapping.items():
                    if k in lms:
                        for j, p in enumerate(lms[k]):
                            if j < len(idxs):
                                idx = idxs[j]
                                if idx < len(arr):
                                    arr[idx] = p

                roi_data = self.landmark_provider.get_roi_for_location(arr, loc)
                if len(roi_data) < 5 or roi_data[2] <= roi_data[0] or roi_data[3] <= roi_data[1]: continue
                roi, angle = roi_data[0:4], roi_data[4]
                anchor = (int((roi[0]+roi[2])/2), int((roi[1]+roi[3])/2))
                target_w, target_h = (roi[2]-roi[0]), (roi[3]-roi[1])

                trans_asset = self._apply_transformations(asset, (target_w, target_h), angle + random.uniform(-self.rotation_range, self.rotation_range), random.uniform(self.min_scale, self.max_scale))
                new_face, bbox = self._blend_asset(face, trans_asset, anchor)

                base = f"synth_{i:05d}"
                cv2.imwrite(str(self.output_dir / "images" / f"{base}.png"), new_face)
                with open(self.output_dir / "labels" / f"{base}.txt", "w") as f:
                    f.write(f"{self.location_to_class[loc]} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
            except Exception as e: print(f"Error {i}: {e}")
        print("✅ Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--faces", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    from dataset_filter.domain.entities import Location
    mapping = {Location.RIGHT_EYEBROW: 0, Location.LEFT_EYEBROW: 1, Location.RIGHT_NOSTRIL: 2, Location.LEFT_NOSTRIL: 3, Location.RIGHT_EAR: 4, Location.LEFT_EAR: 5}
    SyntheticDataGenerator(args.faces, args.assets, args.output, mapping).generate(args.iterations)
