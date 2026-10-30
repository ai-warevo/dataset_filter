"""Implementation of the Quality control stage in the pipeline."""

from __future__ import annotations
from typing import Collection

from tqdm import tqdm

from dataset_filter.domain.interfaces.pipeline_stage import PipelineStage
from dataset_filter.domain.interfaces import Detector, DefectStrategy
from dataset_filter.domain.models import FilterConfig, EvaluationMode, MediaFile, DetectionResult


class QualityStage(PipelineStage):
    """Stage that performs quality checks (blur, face count, scale) on assets."""

    def __init__(
        self,
        detectors: list[Detector],
        config: FilterConfig,
        defect_strategy: DefectStrategy,
    ) -> None:
        self._detectors = detectors
        self._config = config
        self._defect_strategy = defect_strategy

    def execute(self, input_assets: Collection[MediaFile]) -> Collection[MediaFile]:
        """Runs all detectors on the provided assets and returns survivors."""
        survivors: list[MediaFile] = []

        with tqdm(input_assets, desc="Quality check", unit="img", leave=False) as pbar:
            for item in pbar:
                try:
                    results = [detector.check(item.path) for detector in self._detectors]
                    is_defect, reason, score = self._evaluate_policy(results)

                    if is_defect:
                        self._defect_strategy.handle(item, reason=reason)
                    else:
                        item.score = score
                        survivors.append(item)

                except Exception as err:
                    # We treat unexpected errors during detection as defects to be handled by the strategy
                    self._defect_strategy.handle(item, reason=f"Unexpected error: {err}")

        return survivors

    def _evaluate_policy(self, results: list[DetectionResult]) -> tuple[bool, str, float]:
        """Central Policy Engine logic (extracted from DatasetPurgeService)."""
        if self._config.evaluation_mode == EvaluationMode.STRICT:
            # STRICT mode: Any single defect = rejection
            defects = [r for r in results if r.is_defect]
            if defects:
                reasons = "; ".join(r.reason for r in defects if r.reason)
                return True, f"Strict violation: {reasons or 'Unknown'}", 0.0

            # For strict mode, we still return a score (average confidence) for the match strategy
            avg_confidence = sum(r.confidence for r in results) / len(results) if results else 1.0
            return False, "", avg_confidence

        else:
            # SCORING mode: Weighted average of confidences
            total_weighted_confidence = 0.0
            total_weight = 0.0
            
            for res in results:
                # Mapping detectors to weights (this is a bit brittle but works for this stage)
                # In a more robust implementation, the weight would be part of the Detector or Config mapping
                weight = 1.0  # Default weight
                
                total_weighted_confidence += res.confidence * weight
                total_weight += weight

            avg_score = total_weighted_confidence / total_weight if total_weight > 0 else 0.0

            if avg_score < self._config.global_threshold:
                return True, f"Score too low: {avg_score:.2f} (Threshold: {self._config.global_threshold})", avg_score

            return False, "", avg_score
