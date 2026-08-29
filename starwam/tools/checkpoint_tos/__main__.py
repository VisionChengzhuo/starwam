#!/usr/bin/env python
"""Inspect or manually operate the StarWAM TOS checkpoint backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backend import TosCheckpointStore, TosUploadConfig


def _store(args: argparse.Namespace) -> TosCheckpointStore:
    return TosCheckpointStore(
        TosUploadConfig(
            endpoint=args.endpoint,
            region=args.region,
            bucket=args.bucket,
            prefix=args.prefix,
            part_size=args.part_size_mb * 1024 * 1024,
            task_num=args.task_num,
        )
    )


def _add_store_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--endpoint", default="https://tos-cn-beijing.ivolces.com")
    parser.add_argument("--region", default="cn-beijing")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--part-size-mb", type=int, default=64)
    parser.add_argument("--task-num", type=int, default=4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser(
        "doctor", help="validate SDK, credentials, and TOS configuration"
    )
    _add_store_args(doctor)
    upload = commands.add_parser("upload", help="upload one local checkpoint")
    _add_store_args(upload)
    upload.add_argument("--checkpoint-dir", required=True, type=Path)
    upload.add_argument("--run-name", required=True)
    upload.add_argument("--checkpoint-id", required=True)
    upload.add_argument("--source-world-size", required=True, type=int)
    upload.add_argument("--state-root", required=True, type=Path)
    verify = commands.add_parser(
        "verify", help="verify one committed remote checkpoint"
    )
    _add_store_args(verify)
    verify.add_argument("--run-name", required=True)
    verify.add_argument("--checkpoint-id", required=True)
    args = parser.parse_args()

    store = _store(args)
    if args.command == "doctor":
        result = store.check_access()
    elif args.command == "upload":
        result = store.upload_checkpoint(
            args.checkpoint_dir,
            run_name=args.run_name,
            checkpoint_id=args.checkpoint_id,
            source_world_size=args.source_world_size,
            state_root=args.state_root,
        )
    else:
        result = store.verify_checkpoint(args.run_name, args.checkpoint_id)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
