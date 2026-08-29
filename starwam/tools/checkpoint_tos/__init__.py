"""TOS checkpoint persistence tools for StarWAM training.

The lightweight configuration schema is safe to import from the shared
configuration module. Backend imports remain lazy so a normal local training
run does not load the TOS implementation or SDK path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import CheckpointUploadConfig as CheckpointUploadConfig

if TYPE_CHECKING:
    from starwam.config import TrainingConfig

    from .backend import TosUploadManager


def validate_checkpoint_upload(training_config: TrainingConfig) -> None:
    from .backend import validate_checkpoint_upload as validate

    validate(training_config)


def build_checkpoint_uploader(
    training_config: TrainingConfig, *, world_size: int
) -> TosUploadManager | None:
    from .backend import build_checkpoint_uploader as build

    return build(training_config, world_size=world_size)
