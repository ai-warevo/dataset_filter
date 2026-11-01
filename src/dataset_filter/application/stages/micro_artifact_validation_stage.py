"""Implementation of the Micro-Artifact Validation stage."""

from __future__ import annotations
from typing import Collection
import numpy as np
import cv2
import face_recognition
from pathlib import Path

from dataset_filter.domain.interfaces.pipeline_stage import PipelineStage
from dataset_filter.domain.entities import CharacterSpecification
from dataset_filter.domain.models import MediaFile
from dataset_filter.domain.interfaces import DefectStrategy
from dataset_filter.application.services import ValidationOrchestrator

class MicroArtifactValidationStage(PipelineStage):
    """Stage that validates micro-artifacts (piercings) using a validation orchestrator."""

    def __init__(
        self,
        orchestrator: ValidationOrchestrator,
        defect_strategy: DefectStrategy,
        spec: CharacterSpecification,
    ) -> None:
        self._orchestrator = orchestrator
        self._defect_strategy = defect_strategy
        self._spec = spec

    def execute(self, input_assets: Collection[MediaFile]) -> Collection[MediaFile]:
        """Runs micro-artifact validation on the provided assets."""
        survivors: list[MediaFile] = []

        for item in input_assets:
            try:
                image = cv2.imread(str(item.path))
                if image is None:
                    self._defect_strategy.handle(item, reason="Could not read image file")
                    continue

                # 1. Get landmarks
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                landmarks_list = face_recognition.face_landmarks(rgb_image)

                if not landmarks_list:
                    self._defect_strategy.handle(item, reason="No face detected for micro-artifact validation")
                    continue

                landmarks = landmarks_list[0]

                # 2. Run orchestrator
                results = self._orchestrator.run_all(image, landmarks, self._spec)

                # Debug: save visual info as requested
                self._save_debug_image(image, results, item.path.name)

                # 3. Check if any validator failed
                failed_results = [r for r in results if not r.is_valid]

                if failed_results:
                    # Pick the first error to report
                    first_err = failed_results[0]
                    error_type = first_err.error_type.value
                    metadata = first_err.metadata
                    reason = f"Micro-artifact defect ({error_type}): {metadata}"
                    self._defect_strategy.handle(item, reason=reason)
                else:
                    survivors.append(item)

            except Exception as err:
                self._defect_strategy.handle(item, reason=f"Micro-artifact stage error: {err}")

        return survivors

    def _save_debug_image(self, image: np.ndarray, results: Collection[any], filename: str) -> None:
        """Saves an annotated image to data/debug/{filename}."""
        canvas = image.copy()
        for res in results:
            findings = res.metadata.get("findings", {})
            for loc_name, data in findings.items():
                bbox = data.get("bbox")
                count = data.get("count", 0)
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"{loc_name}: {count}"
                    cv2.putText(canvas, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        debug_path = Path("data/debug")
        debug_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path / filename), canvas)
