"""Configuration schema for TOS checkpoint persistence."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CheckpointUploadConfig:
    """Optional TOS checkpoint replication settings.

    Local checkpoints are always written first and are never deleted by the
    uploader. ``bucket`` and ``prefix`` may be supplied through the matching
    ``TOS_BUCKET`` and ``TOS_PREFIX`` environment variables.
    """

    # Cluster-only persistence must be enabled explicitly.
    enabled: bool = False
    endpoint: str = "https://tos-cn-beijing.ivolces.com"
    region: str = "cn-beijing"
    bucket: Optional[str] = "ai-dev"
    prefix: Optional[str] = None
    # None derives the run name from the basename of ``training.output_dir``.
    run_name: Optional[str] = None
    asynchronous: bool = True
    part_size_mb: int = 64
    # Number of multipart upload workers used by the TOS SDK.
    task_num: int = 4
    # None stores resumable upload state under ``training.output_dir``.
    state_dir: Optional[str] = None
