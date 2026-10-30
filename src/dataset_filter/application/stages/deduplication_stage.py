"""Implementation of the deduplication stage in the pipeline."""

from __future__ import annotations
from typing import Collection

from tqdm import tqdm

from dataset_filter.domain.interfaces.pipeline_stage import PipelineStage
from dataset_filter.domain.interfaces import DefectStrategy, FaceValidator, IEmbeddingRepository
from dataset_filter.domain.models import FilterConfig, MediaFile


class DeduplicationStage(PipelineStage):
    """Stage that identifies and handles redundant assets using facial embeddings."""

    def __init__(
        self,
        face_validator: FaceValidator,
        embedding_repository: IEmbeddingRepository,
        config: FilterConfig,
        defect_strategy: DefectStrategy,
    ) -> None:
        self._face_validator = face_validator
        self._embedding_repository = embedding_repository
        self._config = config
        self._defect_strategy = defect_strategy

    def execute(self, input_assets: Collection[MediaFile]) -> Collection[MediaFile]:
        """Identifies duplicates within the batch and returns unique assets."""
        survivors: list[MediaFile] = []

        with tqdm(input_assets, desc="Deduplication", unit="img", leave=False) as pbar:
            for item in pbar:
                try:
                    embedding = self._face_validator.get_embedding(item.path)
                    if embedding:
                        # Check if this embedding is already in the repository
                        if self._embedding_repository.is_duplicate(
                            embedding, 
                            self._config.match.deduplication_threshold
                        ):
                            self._defect_strategy.handle(item, reason="Duplicate face detected")
                        else:
                            # It's unique so far, add it to repo and survivors
                            self._embedding_repository.add(embedding)
                            survivors.append(item)
                    else:
                        # No face found - in current logic, we treat no-face as NOT a duplicate 
                        # (it might be handled by QualityStage if that's a requirement)
                        survivors.append(item)
                except Exception as err:
                    self._defect_strategy.handle(item, reason=f"Deduplication error: {err}")

        return survivors
