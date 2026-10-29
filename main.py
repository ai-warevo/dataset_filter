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
    LogDefectStrategy, 
    PurgeDefectStrategy, 
    NoOpMatchStrategy, 
    RenameMatchStrategy
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
        "--mode",
        choices=["log", "purge", "rename"],
        default="log",
        help="Operation mode: 'log' (dry run), 'purge' (delete defects), or 'rename' (rename matches only). Default: log.",
    )
    return parser.parse_args()


def main() -> None:
    """Bootstrap application components."""
    args = parse_args()
    config = FilterConfig(tolerance=args.tolerance)

    # Map mode to strategy pairs
    if args.mode == "log":
        defect_strategy = LogDefectStrategy()
        match_strategy = NoOpMatchStrategy()
    elif args.mode == "purge":
        defect_strategy = PurgeDefectStrategy()
        match_strategy = NoOpMatchStrategy()
    else:  # mode == "rename"
        defect_strategy = LogDefectStrategy()
        match_strategy = RenameMatchStrategy()

    try:
        # Dependency Injection boundary
        engine = RecognitionEngine(reference_path=args.reference, config=config)
        orchestrator = DatasetPurgeService(
            validator=engine, 
            config=config, 
            defect_strategy=defect_strategy,
            match_strategy=match_strategy
        )
        
        orchestrator.process_directory(args.input_dir)
    except Exception as err:  # noqa: BLE001
        tqdm.write(f"❌ Critical system initialization failure: {err}")


if __name__ == "__main__":
    main()
