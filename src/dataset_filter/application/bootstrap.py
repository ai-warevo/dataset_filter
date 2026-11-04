from __future__ import annotations
"""Application bootstrap service to handle configuration, CLI, and DI."""

import argparse
import os
from pathlib import Path
from tqdm import tqdm

from dataset_filter.application.pipeline_orchestrator import PipelineOrchestrator
from dataset_filter.application.stages.quality_stage import QualityStage
from dataset_filter.application.stages.deduplication_stage import DeduplicationStage
from dataset_filter.application.stages.match_stage import MatchStage
from dataset_filter.application.stages.micro_artifact_validation_stage import MicroArtifactValidationStage
from dataset_filter.domain.interfaces.pipeline_stage import PipelineStage
from dataset_filter.application.strategies import (
    LogStrategy, 
    PurgeDefectStrategy, 
    NoOpMatchStrategy, 
    RenameStrategy
)  # noqa: E402
from dataset_filter.domain.models import FilterConfig, EvaluationMode
from dataset_filter.domain.models.filter_config import MatchConfig, QualityConfig, HeuristicConfig # noqa: E402
from dataset_filter.domain.interfaces import FaceValidator
from dataset_filter.domain.entities import CharacterSpecification, PiercingRequirement, Location
from dataset_filter.domain.value_objects import ValidationResult, ValidationErrorType
from dataset_filter.infrastructure.recognition_engine import RecognitionEngine  # noqa: E402
from dataset_filter.infrastructure.detectors.face_count_detector import FaceCountDetector
from dataset_filter.infrastructure.detectors.blur_detector import BlurDetector
from dataset_filter.infrastructure.detectors.scale_detector import ScaleDetector
from dataset_filter.infrastructure.embedding_repository import InMemoryEmbeddingRepository # noqa: E402
from dataset_filter.infrastructure.models.yolo_client import RealYOLOClient
from dataset_filter.infrastructure.detectors.landmark_provider import LandmarkProvider
from dataset_filter.application.services import ValidationOrchestrator
from dataset_filter.infrastructure.validators.micro_artifact_validator import MicroArtifactValidator

def load_dotenv() -> None:
    """Try to load .env if it exists."""
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

def build_config_from_env() -> FilterConfig:
    """Parses environment variables into a structured FilterConfig."""
    mode_str = os.getenv("FILTER_MODE", "strict").upper()
    eval_mode = EvaluationMode.SCORING if mode_str == "SCORING" else EvaluationMode.STRICT
    threshold = float(os.getenv("FILTER_THRESHOLD", "0.65"))
    extensions = tuple(os.getenv("FILTER_EXTENSIONS", ".png,.jpg,.jpeg").split(","))

    match_tolerance = float(os.getenv("DETECTOR_MATCH_TOLERANCE", "0.55"))
    match_weight = float(os.getenv("DETECTOR_MATCH_WEIGHT", "1.0"))
    dedup_threshold = float(os.getenv("DEDUPLICATION_THRESHOLD", "0.4"))
    match_cfg = MatchConfig(tolerance=match_tolerance, weight=match_weight, deduplication_threshold=dedup_threshold)

    blur_thresh = float(os.getenv("DETECTOR_BLUR_THRESHOLD", "80.0"))
    blur_weight = float(os.getenv("DETECTOR_BLUR_WEIGHT", "0.3"))
    count_max = int(os.getenv("DETECTOR_FACE_COUNT_MAX", "1"))
    count_weight = float(os.getenv("DETECTOR_FACE_COUNT_WEIGHT", "1.0"))
    scale_min = float(os.getenv("DETECTOR_SCALE_MIN", "0.03"))
    scale_weight = float(os.getenv("DETECTOR_SCALE_WEIGHT", "0.5"))

    quality_cfg = QualityConfig(
        blur_threshold=blur_thresh,
        blur_weight=blur_weight,
        face_count_max=count_max,
        face_count_weight=count_weight,
        scale_min=scale_min,
        scale_weight=scale_weight
    )

    heuristic_cfg = HeuristicConfig(
        contrast_multiplier=float(os.getenv("HEURISTIC_CONTRAST_MULTIPLIER", "1.12")),
        brightness_std_multiplier=float(os.getenv("HEURISTIC_BRIGHTNESS_STD_MULTIPLIER", "2.5")),
        min_brightness_floor=float(os.getenv("HEURISTIC_MIN_BRIGHTNESS_FLOOR", "140.0")),
        kernel_size=int(os.getenv("HEURISTIC_KERNEL_SIZE", "7")),
        min_contour_area=float(os.getenv("HEURISTIC_MIN_CONTOUR_AREA", "1.0")),
        clean_side_contrast_threshold=float(os.getenv("HEURISTIC_CLEAN_SIDE_CONTRAST_THRESHOLD", "140.0")),
        nostril_diff_threshold=float(os.getenv("HEURISTIC_NOSTRIL_DIFF_THRESHOLD", "20.0")),
        nostril_min_brightness=float(os.getenv("HEURISTIC_NOSTRIL_MIN_BRIGHTNESS", "150.0"))
    )

    return FilterConfig(
        evaluation_mode=eval_mode,
        global_threshold=threshold,
        valid_extensions=extensions,
        match=match_cfg,
        quality=quality_cfg,
        heuristic=heuristic_cfg
    )

def parse_args() -> argparse.Namespace:
    """Handle CLI argument mapping."""
    parser = argparse.ArgumentParser(description="DDD-compliant dataset filter using src/ layout.")
    
    # Use env vars as defaults if not provided via CLI
    ref_default = os.getenv("REFERENCE_IMAGE")
    input_default = os.getenv("INPUT_DIR")

    parser.add_argument("-r", "--reference", type=Path, default=Path(ref_default) if ref_default else None, required=False, help="Path to the master reference image.")
    parser.add_argument("-i", "--input-dir", type=Path, default=Path(input_default) if input_default else None, required=False, help="Path to the directory containing images to filter.")
    parser.add_argument("--defect-mode", choices=["log", "purge", "rename"], default="log", help="Defect handling mode.")
    parser.add_argument("--match-mode", choices=["noop", "rename", "log"], default="noop", help="Match handling mode.")
    
    args = parser.parse_args()

    if not args.reference or not args.input_dir:
        parser.error("Both --reference and --input-dir must be provided (via CLI or environment variables).")
        
    return args

class Bootstrap:
    """Orchestrates the application startup, configuration, and dependency injection."""

    def run(self) -> None:
        load_dotenv()
        args = parse_args()
        config = build_config_from_env()

        micro_artifact_enabled = os.getenv("MICRO_ARTIFACT_ENABLED", "false").lower() == "true"
        micro_artifact_mode = os.getenv("MICRO_ARTIFACT_MODE", "YOLO").upper()

        defect_strategies = {
            "log": LogStrategy(),
            "purge": PurgeDefectStrategy(),
            "rename": RenameStrategy(),
        }
        match_strategies = {
            "noop": NoOpMatchStrategy(),
            "rename": RenameStrategy(),
            "log": LogStrategy(),
        }

        defect_strategy = defect_strategies[args.defect_mode]
        match_strategy = match_strategies[args.match_mode]

        try:
            # DI Boundary
            engine = RecognitionEngine(reference_path=args.reference, config=config)
            embedding_repo = InMemoryEmbeddingRepository()
            
            detectors = [
                engine,
                FaceCountDetector(config),
                BlurDetector(config),
                ScaleDetector(config)
            ]

            stages: list[PipelineStage] = [
                QualityStage(detectors=detectors, config=config, defect_strategy=defect_strategy),
            ]

            if micro_artifact_enabled:
                print(f"✨ Micro-Artifact Validation ENABLED (Mode: {micro_artifact_mode})")
                
                landmark_provider = LandmarkProvider()

                if micro_artifact_mode == "HEURISTIC":
                    from dataset_filter.infrastructure.validators.heuristic_face_feature_validator import HeuristicFaceFeatureValidator
                    validator = HeuristicFaceFeatureValidator(landmark_provider, config.heuristic)
                else:
                    yolo_model_path = os.getenv("YOLO_MODEL_PATH")
                    if not yolo_model_path:
                        raise ValueError("MICRO_ARTIFACT_ENABLED is true, but YOLO_MODEL_PATH is not set.")
                    from dataset_filter.infrastructure.models.yolo_client import RealYOLOClient
                    yolo_client = RealYOLOClient(model_path=yolo_model_path)
                    validator = MicroArtifactValidator(yolo_client, landmark_provider)

                orchestrator = ValidationOrchestrator([validator])
                
                # Load spec from file
                spec_path = Path(__file__).resolve().parent.parent / "specs" / "smlq_sfw.json"
                if not spec_path.exists():
                    raise FileNotFoundError(f"Character specification not found at: {spec_path}")
                    
                with open(spec_path, "r") as f:
                    spec = CharacterSpecification.from_json(f.read())
                stages.append(MicroArtifactValidationStage(orchestrator, defect_strategy, spec))


            face_validator = next((d for d in detectors if isinstance(d, FaceValidator)), None)
            if not face_validator:
                 raise ValueError("FaceValidator must be provided in detectors for deduplication.")

            stages.append(DeduplicationStage(face_validator=face_validator, embedding_repository=embedding_repo, config=config, defect_strategy=defect_strategy))
            stages.append(MatchStage(match_strategy=match_strategy))

            pipeline = PipelineOrchestrator(stages=stages)
            pipeline.process_directory(args.input_dir, config.valid_extensions)

        except Exception as err:
            tqdm.write(f"❌ Critical system initialization failure: {err}")
