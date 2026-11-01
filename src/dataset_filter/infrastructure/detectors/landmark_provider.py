from __future__ import annotations
"""Wrapper for dlib landmark extraction and ROI calculation."""

import numpy as np
from dataset_filter.domain.entities import Location
from typing import Any

class LandmarkProvider:
    def __init__(self, k_padding: float = 0.5):
        self.k_padding = k_padding

    def calculate_interocular_distance(self, landmarks: Any) -> float:
        """Calculate ID using points 45 and 48 (inner corners of eyes)."""
        if isinstance(landmarks, dict):
            # Extract from face_recognition dict if possible, or fallback to a heuristic
            # For simplicity in this implementation, we'll use the nose bridge/eye distance logic
            # but ideally we'd map face_recognition keys to indices.
            # Let's try to find eye corners in the dict.
            left_eye = landmarks.get('left_eye', [])
            right_eye = landmarks.get('right_eye', [])
            if left_eye and right_eye:
                # Use the inner-most points of the eyes as a proxy for 45/48
                p_left = np.array(left_eye)[0] # face_recognition eye lists are usually ordered
                p_right = np.array(right_eye)[-1]
                return float(np.linalg.norm(p_left - p_right))
            return 50.0 # Fallback
        
        p45 = landmarks[45]
        p48 = landmarks[48]
        return float(np.linalg.norm(p45 - p48))

    def get_roi_for_location(self, landmarks: Any, location: Location, image: np.ndarray) -> tuple[int, int, int, int, float]:
        """
        Calculates ROI and orientation based on landmark indices and interocular distance.
        Returns (x1, y1, x2, y2, angle_degrees).
        """
        h, w = image.shape[:2]
        id_dist = self.calculate_interocular_distance(landmarks)
        
        # Define zones based on dlib 68-point standard OR face_recognition keys
        if isinstance(landmarks, dict):
            # Mapping Location to face_recognition keys or special logic
            mapping = {
                Location.RIGHT_EYEBROW: 'left_eyebrow',  # Left on screen
                Location.LEFT_EYEBROW: 'right_eyebrow',   # Right on screen
            }
            
            if location in mapping:
                key = mapping[location]
                if key not in landmarks:
                    return (0, 0, 0, 0, 0.0)
                points = np.array(landmarks[key])
                # Use fixed small padding for eyebrows to prevent massive ROIs
                padding = 12
            elif location in [Location.LEFT_NOSTRIL, Location.RIGHT_NOSTRIL]:
                # Precise nostril logic from reference implementation
                if 'nose_bridge' not in landmarks or 'nose_tip' not in landmarks:
                    return (0, 0, 0, 0, 0.0)
                
                pts_bridge = np.array(landmarks['nose_bridge'])
                target_wing_y = int(pts_bridge[-1][1])  # Y of bottom bridge point
                
                pts_nose_tip = np.array(landmarks['nose_tip'])
                pts_nose_sorted = pts_nose_tip[pts_nose_tip[:, 0].argsort()]
                
                nx_left_wing = int(pts_nose_sorted[0][0])   # Extreme left X
                nx_right_wing = int(pts_nose_sorted[-1][0]) # Extreme right X
                
                pad_nos = 15
                if location == Location.LEFT_NOSTRIL:
                    # Anatomically right wing, on screen right
                    x1, y1 = max(0, nx_right_wing - pad_nos), max(0, target_wing_y - pad_nos)
                    x2, y2 = min(w, nx_right_wing + pad_nos), min(h, target_wing_y + pad_nos)
                else:
                    # Anatomically left wing, on screen left
                    x1, y1 = max(0, nx_left_wing - pad_nos), max(0, target_wing_y - pad_nos)
                    x2, y2 = min(w, nx_left_wing + pad_nos), min(h, target_wing_y + pad_nos)
                
                return (x1, y1, x2, y2, 0.0)
            else:
                return (0, 0, 0, 0, 0.0)
            
            valid_points = points[np.any(points != 0, axis=1)]
        else:
            # Standard dlib index-based logic
            if location == Location.RIGHT_EYEBROW:
                indices = list(range(22, 27))
            elif location == Location.LEFT_EYEBROW:
                indices = list(range(17, 22))
            elif location == Location.RIGHT_NOSTRIL:
                indices = list(range(36, 41))
            elif location == Location.LEFT_NOSTRIL:
                indices = list(range(31, 36))
            elif location == Location.RIGHT_EAR:
                indices = list(range(12, 17))
            elif location == Location.LEFT_EAR:
                indices = list(range(0, 5))
            else:
                raise ValueError(f"Unsupported location: {location}")
            
            points = landmarks[indices]
            valid_points = points[np.any(points != 0, axis=1)]
        if len(valid_points) == 0:
            return (0, 0, 0, 0, 0.0)

        min_pts = np.min(valid_points, axis=0)
        max_pts = np.max(valid_points, axis=0)
        
        # Determine padding: fixed for eyebrows, scaled for others
        if isinstance(landmarks, dict) and location in [Location.RIGHT_EYEBROW, Location.LEFT_EYEBROW]:
            padding = 12
        else:
            padding = id_dist * self.k_padding

        x1 = int(min_pts[0] - padding)
        y1 = int(min_pts[1] - padding)
        x2 = int(max_pts[0] + padding)
        y2 = int(max_pts[1] + padding)
        
        # Calculate orientation angle
        angle = 0.0
        if len(valid_points) >= 2:
            p_start = valid_points[0]
            p_end = valid_points[-1]
            dy = p_end[1] - p_start[1]
            dx = p_end[0] - p_start[0]
            angle = np.degrees(np.arctan2(dy, dx))
        
        return (max(0, x1), max(0, y1), x2, y2, angle)
