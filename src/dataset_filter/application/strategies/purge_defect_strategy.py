"""Strategy that permanently deletes the defective file."""

from dataset_filter.domain.models import MediaFile
from dataset_filter.domain.interfaces import DefectStrategy


class PurgeDefectStrategy(DefectStrategy):
    def handle(self, item: MediaFile) -> None:
        item.purge()
