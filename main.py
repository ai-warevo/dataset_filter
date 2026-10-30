"""CLI bootstrap entrypoint applying clean Dependency Injection and src layout path resolution."""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# Resolving the 'src/' layout boundary for direct script execution
# This allows 'import dataset_filter' to work out-of-the-box locally
SRC_PATH = str(Path(__file__).resolve().parent / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from dataset_filter.application.purge_service import DatasetPurgeService  # noqa: E402
from dataset_filter.application.strategies import (
    LogStrategy, 
    PurgeDefectStrategy, 
    NoOpMatchStrategy, 
    RenameStrategy
)  # noqa: E402
from dataset_filter.domain.models import FilterConfig  # noqa: E402
from dataset_filter.infrastructure.recognition_engine import RecognitionEngine  # noqa: E402


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
        "-t",
        "--tolerance",
        type=float,
        default=0.55,
        help="Strictness threshold: lower values mean stricter matching (default: 0.55).",
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
    args = parse_args()
    config = FilterConfig(tolerance=args.tolerance)

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
        orchestrator = DatasetPurgeService(
            detectors=[engine], 
            config=config, 
            defect_strategy=defect_strategy,
            match_strategy=match_strategy
        )
        
        orchestrator.process_directory(args.input_dir)
    except Exception as err:  # noqa: BLE001
        tqdm.write(f"❌ Critical system initialization failure: {err}")


if __name__ == "__main__":
    main()
