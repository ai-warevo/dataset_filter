"""Application layer services coordinating domain workflows."""

from __future__ import annotations
from pathlib import Path
from tqdm import tqdm

from dataset_filter.domain.exceptions import ImageProcessingError
from dataset_filter.domain.models import FilterConfig, MediaFile, EvaluationMode
from dataset_filter.domain.interfaces import Detector, DefectStrategy, MatchStrategy


class DatasetPurgeService:
    """Application Service coordinating the core data filtering lifecycle workflows."""

    def __init__(
        self, 
        detectors: list[Detector], 
        config: FilterConfig, 
        defect_strategy: DefectStrategy,
        match_strategy: MatchStrategy | None = None
    ) -> None:
        self._detectors: list[Detector] = detectors
        self._config: FilterConfig = config
        self._defect_strategy: DefectStrategy = defect_strategy
        self._match_strategy: MatchStrategy | None = match_strategy

    def process_directory(self, target_directory: Path) -> None:
        """Scan directory boundaries, sorting real assets from structural anomalies."""
        all_targets: list[MediaFile] = [
            MediaFile(f)
            for f in target_directory.iterdir()
            if f.is_file() and f.suffix.lower() in self._config.valid_extensions
        ]

        tqdm.write(f"🔍 Found {len(all_targets)} images to parse. Starting service...")
        stats: dict[str, int] = {"passed": 0, "defects": 0}

        with tqdm(all_targets, desc="Filtering dataset", unit="img") as pbar:
            for item in pbar:
                self._inspect_item(item, stats)

        tqdm.write("\n📊 Purge workflow finalized!")
        tqdm.write(f"   Defects handled: {stats['defects']} files.")
        tqdm.write(f"   Retained items: {stats['passed']} files within '{target_directory}'.")

    def _inspect_item(self, item: MediaFile, stats: dict[str, int]) -> None:
        try:
            results = []
            for detector in self._detectors:
                results.append(detector.check(item.path))

            is_defect, reason, final_score = self._evaluate_policy(results)

            if is_defect:
                self._defect_strategy.handle(item, reason=reason)
                stats["defects"] += 1
                return

            # If we reached here, it passed the policy
            stats["passed"] += 1
            if self._match_strategy:
                self._match_strategy.handle(item, final_score)
            return

        except ImageProcessingError as err:
            tqdm.write(f"💥 Defect detected (System exception: {err}) #{item.path}")
            self._defect_strategy.handle(item, reason=str(err))
            stats["defects"] += 1
        except Exception as err:
            tqdm.write(f"❌ Unexpected error processing {item.path}: {err}")
            self._defect_strategy.handle(item, reason=f"Unexpected error: {err}")
            stats["defects"] += 1

    def _evaluate_policy(self, results: list[DetectionResult]) -> tuple[bool, str, float]:
        """Central Policy Engine logic."""
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
            # We need to know which detector produced which result to apply weights.
            # Since detectors are ordered, we assume the order matches our config structure or 
            # we just use a simple unweighted average if weights aren't easily mapped here.
            # For now, let's implement weighted scoring based on the known detector types.
            
            total_weighted_confidence = 0.0
            total_weight = 0.0
            defect_reasons = []

            for i, res in enumerate(results):
                # Mapping detectors to weights (this is a bit brittle but works for this stage)
                # In a production system, we'd use a registry or metadata on the detector itself.
                weight = 1.0 # Default weight
                
                # Heuristic mapping based on class name/type if possible, 
                # but let's stick to simple unweighted for the first pass of SCORING
                # unless we want to make it more robust.
                
                total_weighted_confidence += res.confidence * weight
                total_weight += weight
                
                if res.is_defect:
                    defect_reasons.append(f"{res.reason or 'Defect'} ({res.confidence:.2f})")

            avg_score = total_weighted_confidence / total_weight if total_weight > 0 else 0.0
            
            if avg_score < self._config.global_threshold:
                return True, f"Score too low: {avg_score:.2f} (Threshold: {self._config.global_threshold})", avg_score
            
            return False, "", avg_score

