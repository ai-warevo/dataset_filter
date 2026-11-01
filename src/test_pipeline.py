import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Ensure src is in path for testing
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from dataset_filter.domain.models.media_file import MediaFile
from dataset_filter.domain.models.filter_config import FilterConfig, MatchConfig, QualityConfig, EvaluationMode
from dataset_filter.domain.models.detection_result import DetectionResult
from dataset_filter.application.pipeline_orchestrator import PipelineOrchestrator
from dataset_filter.application.stages.quality_stage import QualityStage
from dataset_filter.application.stages.deduplication_stage import DeduplicationStage
from dataset_filter.application.stages.match_stage import MatchStage

class TestPipelineRefactoring(unittest.TestCase):
    def setUp(self):
        # Common config
        self.config = FilterConfig(
            evaluation_mode=EvaluationMode.STRICT,
            global_threshold=0.5,
            valid_extensions=(".png",),
            match=MatchConfig(tolerance=0.5, weight=1.0, deduplication_threshold=0.4),
            quality=QualityConfig(blur_threshold=100, blur_weight=1.0, face_count_max=1, face_count_weight=1.0, scale_min=0.1, scale_weight=1.0)
        )
        # Mock MediaFile
        self.mock_file = MagicMock(spec=MediaFile)
        self.mock_file.path = Path("/tmp/test.png")
        self.mock_file.score = None

    def test_scenario_purge_noop(self):
        """
        Scenario: APP_DEFECT_MODE=purge, APP_MATCH_MODE=noop
        Expectation: 
        - Defects (Quality or Duplicate) call PurgeDefectStrategy.handle
        - Valid items DO call MatchStrategy.handle (but it's a noop strategy)
        """
        # Mocks
        defect_strategy = MagicMock() # Represents PurgeDefectStrategy
        match_strategy = MagicMock()  # Represents NoOpMatchStrategy
        detector = MagicMock()
        face_validator = MagicMock()
        embedding_repo = MagicMock()

        # 1. Setup QualityStage (Simulate a defect)
        detector.check.return_value = DetectionResult(is_defect=True, reason="Blurry")
        quality_stage = QualityStage(detectors=[detector], config=self.config, defect_strategy=defect_strategy)

        # 2. Setup DeduplicationStage (Simulate a duplicate)
        face_validator.get_embedding.return_value = MagicMock()
        embedding_repo.is_duplicate.return_value = True
        dedup_stage = DeduplicationStage(face_validator, embedding_repo, self.config, defect_strategy)

        # 3. Setup MatchStage (noop)
        match_stage = MatchStage(match_strategy=match_strategy)

        orchestrator = PipelineOrchestrator(stages=[quality_stage, dedup_stage, match_stage])

        # --- EXECUTION ---
        
        # Case A: Quality Defect
        assets = [self.mock_file]
        results = orchestrator.run(assets)
        
        self.assertEqual(len(results), 0)
        # Note: In STRICT mode, the reason is prefixed with "Strict violation: "
        defect_strategy.handle.assert_called_with(self.mock_file, reason="Strict violation: Blurry")
        match_strategy.handle.assert_not_called()

        # Reset mocks for next sub-case
        defect_strategy.reset_mock()
        match_strategy.reset_mock()
        detector.reset_mock()
        detector.check.return_value = DetectionResult(is_defect=False) # Pass quality

        # Case B: Duplicate Defect
        assets = [self.mock_file]
        results = orchestrator.run(assets)
        
        self.assertEqual(len(results), 0)
        defect_strategy.handle.assert_called_with(self.mock_file, reason="Duplicate face detected")
        match_strategy.handle.assert_not_called()

        # Reset mocks for next sub-case
        defect_strategy.reset_mock()
        match_strategy.reset_mock()
        detector.reset_mock()
        detector.check.return_value = DetectionResult(is_defect=False)
        embedding_repo.is_duplicate.return_value = False # Not a duplicate

        # Case C: Valid Item (Match is noop)
        assets = [self.mock_file]
        results = orchestrator.run(assets)
        
        self.assertEqual(len(results), 1)
        defect_strategy.handle.assert_not_called()
        # In the new pipeline, MatchStage is always present, so handle IS called even if it's noop
        match_strategy.handle.assert_called_once()

    def test_scenario_rename_rename(self):
        """
        Scenario: APP_DEFECT_MODE=rename, APP_MATCH_MODE=rename
        Expectation: 
        - Defects call RenameStrategy.handle
        - Valid items call RenameStrategy.handle (for match)
        """
        # Mocks
        defect_strategy = MagicMock() # Represents RenameStrategy
        match_strategy = MagicMock()  # Represents RenameStrategy
        detector = MagicMock()
        face_validator = MagicMock()
        embedding_repo = MagicMock()

        # 1. Setup QualityStage (Pass)
        detector.check.return_value = DetectionResult(is_defect=False)
        quality_stage = QualityStage(detectors=[detector], config=self.config, defect_strategy=defect_strategy)

        # 2. Setup DeduplicationStage (Pass)
        face_validator.get_embedding.return_value = MagicMock()
        embedding_repo.is_duplicate.return_value = False
        dedup_stage = DeduplicationStage(face_validator, embedding_repo, self.config, defect_strategy)

        # 3. Setup MatchStage (rename)
        match_stage = MatchStage(match_strategy=match_strategy)

        orchestrator = PipelineOrchestrator(stages=[quality_stage, dedup_stage, match_stage])

        # --- EXECUTION ---

        # Case A: Valid Item (Match is rename)
        assets = [self.mock_file]
        results = orchestrator.run(assets)
        
        self.assertEqual(len(results), 1)
        defect_strategy.handle.assert_not_called()
        match_strategy.handle.assert_called_once()

        # Case B: Quality Defect (Defect is rename)
        defect_strategy.reset_mock()
        match_strategy.reset_mock()
        detector.check.return_value = DetectionResult(is_defect=True, reason="Bad")
        
        assets = [self.mock_file]
        results = orchestrator.run(assets)
        
        self.assertEqual(len(results), 0)
        # Note: In STRICT mode, the reason is prefixed with "Strict violation: "
        defect_strategy.handle.assert_called_with(self.mock_file, reason="Strict violation: Bad")
        match_strategy.handle.assert_not_called()

if __name__ == "__main__":
    unittest.main()
