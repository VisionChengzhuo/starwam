# TOS Checkpoint Persistence

TOS checkpoint replication is optional trainer infrastructure. It applies to
every recipe built with `StarWAMTrainer`; it is not coupled to a benchmark,
model family, or backbone. It is disabled by default so shared recipes remain
runnable on a single machine.

The shared config module keeps only the `training.checkpoint_upload` integration
field. TOS-specific defaults and schema live in
`starwam/tools/checkpoint_tos/config.py`. The backend and TOS SDK path are loaded
only when upload is explicitly enabled. StarWAM continues to use its existing
dataclass, YAML, and `--override` configuration flow; TOS support does not add a
configuration framework.

## Behavior

Local save always completes first. The backend then uploads every payload,
verifies remote size and CRC, and writes the manifest last as the commit marker.
The uploader never deletes a local checkpoint. When local retention is enabled,
a remotely replicated checkpoint becomes eligible for cleanup only after the
atomic `.tos_upload_verified.json` marker exists.

An enabled upload is part of the run's persistence contract. An invalid or
inaccessible destination fails before model construction. A background upload
failure is raised before the next upload or when the trainer closes.

Remote checkpoints use this layout:

```text
tos://<bucket>/<prefix>/<run-name>/checkpoints/<zero-padded-step>/
```

## Installation and access check

Install the optional dependency and provide credentials through the TOS SDK
environment variables:

```bash
pip install -e '.[tos]'

export TOS_ACCESS_KEY=...
export TOS_SECRET_KEY=...
export TOS_BUCKET=ai-dev
export TOS_PREFIX=YOUR_NAME/starwam_checkpoints

python -m starwam.tools.checkpoint_tos doctor \
  --bucket "$TOS_BUCKET" \
  --prefix "$TOS_PREFIX"
```

`doctor` performs a live bucket-access check without uploading an object.

## Enable for a training run

Any training command can enable TOS through the existing override mechanism:

```bash
python -m starwam.training.train \
  --config path/to/recipe.yaml \
  --override training.checkpoint_upload.enabled=true
```

`TOS_BUCKET` and `TOS_PREFIX` supply the destination by default. Additional
settings can be overridden in the same command:

```text
training.checkpoint_upload.endpoint
training.checkpoint_upload.region
training.checkpoint_upload.bucket
training.checkpoint_upload.prefix
training.checkpoint_upload.run_name
training.checkpoint_upload.asynchronous
training.checkpoint_upload.part_size_mb
training.checkpoint_upload.task_num
training.checkpoint_upload.state_dir
```

The schema defaults are documented in
`starwam/tools/checkpoint_tos/config.py`; unknown YAML keys and override paths
continue to fail immediately.
