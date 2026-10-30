"""CLI bootstrap entrypoint applying clean Dependency Injection and src layout path resolution."""

import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Resolving the 'src/' layout boundary for direct script execution
SRC_PATH = str(Path(__file__).resolve().parent / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from dataset_filter.application.pipeline_orchestrator import PipelineOrchestrator
from dataset_filter.application.stages.quality_stage import QualityStage
from dataset_filter.application.stages.deduplication_stage import DeduplicationStage
from dataset_filter.application.stages.match_stage import MatchStage
from dataset_filter.application.strategies import (
    LogStrategy, 
    PurgeDefectStrategy, 
    NoOpMatchStrategy, 
    RenameStrategy
)  # noqa: E402
from dataset_filter.domain.models import FilterConfig, EvaluationMode  # noqa: E402
from dataset_filter.domain.models.filter_config import MatchConfig, QualityConfig # noqa: E402
from dataset_filter.domain.interfaces import FaceValidator
from dataset_filter.infrastructure.recognition_engine import RecognitionEngine  # noqa: E402
from dataset_filter.infrastructure.detectors.face_count_detector import FaceCountDetector
from dataset_filter.infrastructure.detectors.blur_detector import BlurDetector
from dataset_filter.infrastructure.detectors.scale_detector import ScaleDetector
from dataset_filter.infrastructure.embedding_repository import InMemoryEmbeddingRepository # noqa: E402

# Try to load .env if it exists (using simple manual parsing to avoid new dependency)
def load_dotenv() -> None:
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
    # Global settings
    mode_str = os.getenv("FILTER_MODE", "strict").upper()
    eval_mode = EvaluationMode.SCORING if mode_str == "SCORING" else EvaluationMode.STRICT
    threshold = float(os.getenv("FILTER_THRESHOLD", "0.65"))
    extensions = tuple(os.getenv("FILTER_EXTENSIONS", ".png,.jpg,.jpeg").split(","))

    # Match Config (Identity)
    match_tolerance = float(os.getenv("DETECTOR_MATCH_TOLERANCE", "0.55"))
    match_weight = float(os.getenv("DETECTOR_MATCH_WEIGHT", "1.0"))
    dedup_threshold = float(os.getenv("DEDUPLICATION_THRESHOLD", "0.4"))
    match_cfg = MatchConfig(tolerance=match_tolerance, weight=match_weight, deduplication_threshold=dedup_threshold)

    # Quality Config (Blur, Count, Scale)
    blur_thresh = float(os.getenv("DETECTOR_BLUR_THRESHOLD", "120.0"))
    blur_weight = float(os.getenv("DETECTOR_BLUR_WEIGHT", "0.3"))
    count_max = int(os.getenv("DETECTOR_FACE_COUNT_MAX", "1"))
    count_weight = float(os.getenv("DETECTOR_FACE_COUNT_WEIGHT", "1.0"))
    scale_min = float(os.getenv("DETECTOR_SCALE_MIN", "0.1"))
    scale_weight = float(os.getenv("DETECTOR_SCALE_WEIGHT", "0.5"))

    quality_cfg = QualityConfig(
        blur_threshold=blur_thresh,
        blur_weight=blur_weight,
        face_count_max=count_max,
        face_count_weight=count_weight,
        scale_min=scale_min,
        scale_weight=scale_weight
    )

    return FilterConfig(
        evaluation_mode=eval_mode,
        global_threshold=threshold,
        valid_extensions=extensions,
        match=match_cfg,
        quality=quality_cfg
    )


def parse_args() -> argparse.Namespace:
    """Handle CLI argument mapping."""
    parser = argparse.ArgumentParser(description="DDD-compliant dataset filter using src/ layout.")
    parser.add_argument(
        "-r",
        "--reference",
        type=Path,
        required=True,
        help="Path to the master reference image.",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Path to the directory containing images to filter.",
    )
    parser.add_argument(
        "--defect-mode",
        choices=["log", "purge", "rename"],
        default="log",
        help="Defect handling mode: 'log' (dry run), 'purge' (delete defects), or 'rename' (rename defects). Default: log.",
    )
    parser.add_argument(
        "--match-mode",
        choices=["noop", "rename", "log"],
        default="noop",
        help="Match handling mode: 'noop' (do nothing), 'rename' (rename matches), or 'log' (log matches). Default: noop.",
    )
    return parser.parse_args()


def main() -> None:
    """Bootstrap application components."""
    load_dotenv()
    args = parse_args()
    config = build_config_from_env()

    # Map modes to strategy pairs
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
        # Dependency Injection boundary
        engine = RecognitionEngine(reference_path=args.reference, config=config)
        embedding_repo = InMemoryEmbeddingRepository()
        
        # Initialize all detectors with the shared config
        detectors = [
            engine,
            FaceCountDetector(config),
            BlurDetector(config),
            ScaleDetector(config)
        ]

        # Find the face validator among detectors to pass it to DeduplicationStage
        face_validator = next((d for d in detectors if isinstance(d, FaceValidator)), None)
        if not face_validator:
             raise ValueError("FaceValidator must be provided in detectors for deduplication.")

        orchestrator = PipelineOrchestrator(stages=[
            QualityStage(detectors=detectors, config=config, defect_strategy=defect_strategy),
            DeduplicationStage(face_validator=face_validator, embedding_repository=embedding_repo, config=config, defect_strategy=defect_strategy),
            MatchStage(match_strategy=match_strategy)
        ])
        
        orchestrator.process_directory(args.input_dir, config.valid_extensions)
    except Exception as err:  # noqa: BLE001
        tqdm.write(f"❌ Critical system initialization failure: {err}")


if __name__ == "__main__":
    main()
