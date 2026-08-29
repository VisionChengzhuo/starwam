"""Verified Volcengine TOS uploads for immutable StarWAM checkpoints.

The design follows the checkpoint lifecycle used by Cosmos: upload every
payload first, verify remote size/CRC, and publish a manifest last as the
commit marker. Local checkpoints are never deleted by this module. Failed or
interrupted uploads therefore leave a complete local checkpoint available for
resume and can be retried idempotently.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from starwam.config import TrainingConfig


logger = logging.getLogger(__name__)
TOS_MANIFEST_NAME = "_starwam_checkpoint_manifest.json"
TOS_UPLOAD_MARKER = ".tos_upload_verified.json"
_MANIFEST_VERSION = 1
_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class TosUploadError(RuntimeError):
    """Raised when a checkpoint cannot be safely committed to TOS."""


def _validate_name(value: str, label: str) -> str:
    if not _NAME_RE.fullmatch(value):
        raise ValueError(
            f"Invalid {label}={value!r}; allowed characters: letters, digits, '_', '.', '-'"
        )
    return value


def _validate_prefix(value: str) -> str:
    if value.startswith(("tos://", "/")) or value.endswith("/"):
        raise ValueError(
            f"Invalid checkpoint_upload.prefix={value!r}; omit the scheme and leading/trailing slash"
        )
    parts = value.split("/")
    if not value or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(
            f"Invalid checkpoint_upload.prefix={value!r}; empty, '.' and '..' segments are not allowed"
        )
    for part in parts:
        _validate_name(part, "TOS prefix segment")
    return value


def _checkpoint_id(value: str | int) -> str:
    text = str(value)
    if not text.isdigit():
        raise ValueError(
            f"Invalid checkpoint_id={value!r}; expected an unsigned integer"
        )
    return f"{int(text):07d}"


@dataclass(frozen=True)
class TosUploadConfig:
    """Validated TOS connection and multipart upload settings."""

    endpoint: str
    region: str
    bucket: str
    prefix: str
    # Multipart chunk size in bytes.
    part_size: int = 64 * 1024 * 1024
    # Number of concurrent multipart workers used by the TOS SDK.
    task_num: int = 4

    def __post_init__(self) -> None:
        parsed = urlparse(self.endpoint)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError(
                f"Invalid checkpoint_upload.endpoint={self.endpoint!r}; expected an HTTPS origin without a path"
            )
        if (
            not self.region
            or "/" in self.region
            or any(char.isspace() for char in self.region)
        ):
            raise ValueError(
                f"Invalid checkpoint_upload.region={self.region!r}; expected a non-empty region name"
            )
        if not _BUCKET_RE.fullmatch(self.bucket):
            raise ValueError(
                f"Invalid checkpoint_upload.bucket={self.bucket!r}; expected a valid lowercase TOS bucket name"
            )
        _validate_prefix(self.prefix)
        if not 5 * 1024 * 1024 <= self.part_size <= 5 * 1024 * 1024 * 1024:
            raise ValueError(
                f"Invalid checkpoint_upload.part_size_mb={self.part_size / 1024 / 1024:g}; "
                "allowed range: 5 MiB to 5 GiB"
            )
        if self.task_num <= 0:
            raise ValueError(
                f"Invalid checkpoint_upload.task_num={self.task_num!r}; expected a positive integer"
            )


def _build_tos_client(config: TosUploadConfig) -> Any:
    try:
        import tos
    except ImportError as exc:
        raise ImportError("TOS upload requires `pip install -e '.[tos]'`") from exc

    missing = [
        name
        for name in ("TOS_ACCESS_KEY", "TOS_SECRET_KEY")
        if not os.environ.get(name, "").strip()
    ]
    if missing:
        raise ValueError(
            "Missing TOS credential environment variable(s): " + ", ".join(missing)
        )
    logging.getLogger("tos").setLevel(logging.WARNING)
    return tos.TosClientV2(
        endpoint=config.endpoint,
        region=config.region,
        credentials_provider=tos.EnvCredentialsProvider(),
        enable_crc=True,
        max_retry_count=3,
        connection_time=10,
        socket_timeout=30,
    )


def _status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _response_bytes(response: Any) -> bytes:
    if hasattr(response, "__iter__"):
        return b"".join(response)
    chunks: list[bytes] = []
    while True:
        chunk = response.read(1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _payload_files(checkpoint_dir: Path) -> list[tuple[str, Path]]:
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise TosUploadError(
            f"checkpoint directory is missing or is a symlink: {checkpoint_dir}"
        )
    files: list[tuple[str, Path]] = []
    for path in sorted(checkpoint_dir.rglob("*")):
        if path.is_symlink():
            raise TosUploadError(
                f"checkpoint payload must not contain symlinks: {path}"
            )
        if not path.is_file() or path.name == TOS_UPLOAD_MARKER:
            continue
        files.append((path.relative_to(checkpoint_dir).as_posix(), path))
    if not files:
        raise TosUploadError(f"checkpoint has no payload files: {checkpoint_dir}")
    return files


def _local_manifest(
    checkpoint_dir: Path,
    *,
    run_name: str,
    checkpoint_id: str | int,
    source_world_size: int,
) -> dict[str, Any]:
    run_name = _validate_name(run_name, "run name")
    checkpoint_id = _checkpoint_id(checkpoint_id)
    if source_world_size <= 0:
        raise TosUploadError("source_world_size must be positive")
    files: dict[str, dict[str, Any]] = {}
    total_size = 0
    for relative, path in _payload_files(checkpoint_dir):
        size = path.stat().st_size
        files[relative] = {"size": size, "sha256": _sha256_file(path)}
        total_size += size
    return {
        "version": _MANIFEST_VERSION,
        "run_name": run_name,
        "checkpoint_id": checkpoint_id,
        "source_world_size": int(source_world_size),
        "files": files,
        "total_files": len(files),
        "total_size": total_size,
    }


def _payload_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": manifest.get("version"),
        "run_name": manifest.get("run_name"),
        "checkpoint_id": manifest.get("checkpoint_id"),
        "source_world_size": manifest.get("source_world_size"),
        "files": {
            key: {"size": value.get("size"), "sha256": value.get("sha256")}
            for key, value in manifest.get("files", {}).items()
        },
        "total_files": manifest.get("total_files"),
        "total_size": manifest.get("total_size"),
    }


class TosCheckpointStore:
    """Upload and verify checkpoint directories using a manifest-last commit."""

    def __init__(self, config: TosUploadConfig, *, client: Any | None = None) -> None:
        self.config = config
        self.client = client if client is not None else _build_tos_client(config)

    def _base_key(self, run_name: str, checkpoint_id: str | int) -> str:
        return f"{self.config.prefix}/{_validate_name(run_name, 'run name')}/checkpoints/{_checkpoint_id(checkpoint_id)}"

    def _manifest_key(self, run_name: str, checkpoint_id: str | int) -> str:
        return f"{self._base_key(run_name, checkpoint_id)}/{TOS_MANIFEST_NAME}"

    def destination(self, run_name: str, checkpoint_id: str | int) -> str:
        return f"tos://{self.config.bucket}/{self._base_key(run_name, checkpoint_id)}/"

    def check_access(self) -> dict[str, str]:
        """Verify that the configured credentials can access the destination bucket."""
        logger.info("Checking TOS access for bucket %s ...", self.config.bucket)
        try:
            self.client.head_bucket(self.config.bucket)
        except Exception as exc:
            raise TosUploadError(
                f"Cannot access checkpoint_upload.bucket={self.config.bucket!r}: {exc}"
            ) from exc
        logger.info("TOS access ready for bucket %s", self.config.bucket)
        return {
            "status": "ready",
            "bucket": self.config.bucket,
            "prefix": self.config.prefix,
        }

    def _get_manifest_bytes(
        self, run_name: str, checkpoint_id: str | int, *, missing_ok: bool
    ) -> bytes | None:
        try:
            response = self.client.get_object(
                self.config.bucket, self._manifest_key(run_name, checkpoint_id)
            )
            return _response_bytes(response)
        except Exception as exc:
            if missing_ok and _status_code(exc) == 404:
                return None
            raise TosUploadError(
                f"cannot read remote checkpoint manifest: {exc}"
            ) from exc

    @staticmethod
    def _parse_manifest(raw: bytes) -> dict[str, Any]:
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TosUploadError(f"invalid remote checkpoint manifest: {exc}") from exc
        if not isinstance(value, dict) or value.get("version") != _MANIFEST_VERSION:
            raise TosUploadError(f"unsupported remote checkpoint manifest: {value!r}")
        return value

    def _verify_remote(self, manifest: dict[str, Any]) -> None:
        base = self._base_key(manifest["run_name"], manifest["checkpoint_id"])
        for relative, entry in manifest["files"].items():
            try:
                output = self.client.head_object(
                    self.config.bucket, f"{base}/{relative}"
                )
            except Exception as exc:
                raise TosUploadError(
                    f"cannot verify remote checkpoint payload {relative}: {exc}"
                ) from exc
            actual_size = getattr(output, "content_length", None)
            actual_crc = getattr(output, "hash_crc64_ecma", None)
            expected_crc = entry.get("crc64_ecma")
            if actual_size != entry["size"] or (
                expected_crc is not None and actual_crc != expected_crc
            ):
                raise TosUploadError(
                    f"remote checkpoint payload mismatch for {relative}: "
                    f"size={actual_size!r}/{entry['size']!r} crc64={actual_crc!r}/{expected_crc!r}"
                )

    def verify_checkpoint(
        self, run_name: str, checkpoint_id: str | int
    ) -> dict[str, Any]:
        logger.info(
            "Verifying TOS checkpoint %s ...", self.destination(run_name, checkpoint_id)
        )
        raw = self._get_manifest_bytes(run_name, checkpoint_id, missing_ok=False)
        assert raw is not None
        manifest = self._parse_manifest(raw)
        self._verify_remote(manifest)
        result = self._result(manifest, status="verified", manifest_bytes=raw)
        logger.info("Verified TOS checkpoint %s", result["destination"])
        return result

    def _upload_payloads(
        self, checkpoint_dir: Path, *, base_key: str, state_root: Path
    ) -> dict[str, int]:
        state_root.mkdir(parents=True, exist_ok=True)
        crc_by_path: dict[str, int] = {}
        for relative, path in _payload_files(checkpoint_dir):
            state_file = (
                state_root / f"{hashlib.sha256(relative.encode()).hexdigest()}.json"
            )
            try:
                output = self.client.upload_file(
                    self.config.bucket,
                    f"{base_key}/{relative}",
                    str(path),
                    part_size=self.config.part_size,
                    task_num=self.config.task_num,
                    enable_checkpoint=True,
                    checkpoint_file=str(state_file),
                )
            except Exception as exc:
                raise TosUploadError(
                    f"cannot upload checkpoint payload {relative}: {exc}"
                ) from exc
            crc = getattr(output, "hash_crc64_ecma", None)
            if not isinstance(crc, int) or isinstance(crc, bool) or crc < 0:
                raise TosUploadError(
                    f"TOS did not return a valid CRC64 for {relative}: {crc!r}"
                )
            crc_by_path[relative] = crc
        return crc_by_path

    def _commit_manifest(
        self,
        manifest: dict[str, Any],
        *,
        run_name: str,
        checkpoint_id: str | int,
    ) -> bytes:
        body = _canonical_json(manifest)
        self._verify_remote(manifest)
        try:
            self.client.put_object(
                self.config.bucket,
                self._manifest_key(run_name, checkpoint_id),
                content=body,
                forbid_overwrite=True,
            )
        except Exception as exc:
            raced = self._get_manifest_bytes(run_name, checkpoint_id, missing_ok=True)
            if raced != body:
                raise TosUploadError(
                    f"cannot commit checkpoint manifest: {exc}"
                ) from exc

        verified_raw = self._get_manifest_bytes(
            run_name, checkpoint_id, missing_ok=False
        )
        if verified_raw != body:
            raise TosUploadError(
                "committed remote manifest differs from the uploaded manifest"
            )
        self._verify_remote(manifest)
        return body

    def upload_checkpoint(
        self,
        checkpoint_dir: str | Path,
        *,
        run_name: str,
        checkpoint_id: str | int,
        source_world_size: int,
        state_root: str | Path,
    ) -> dict[str, Any]:
        checkpoint_dir = Path(checkpoint_dir).resolve()
        destination = self.destination(run_name, checkpoint_id)
        logger.info("Uploading checkpoint %s to %s ...", checkpoint_dir, destination)
        local = _local_manifest(
            checkpoint_dir,
            run_name=run_name,
            checkpoint_id=checkpoint_id,
            source_world_size=source_world_size,
        )
        existing_raw = self._get_manifest_bytes(
            run_name, checkpoint_id, missing_ok=True
        )
        if existing_raw is not None:
            existing = self._parse_manifest(existing_raw)
            if _payload_identity(existing) != _payload_identity(local):
                raise TosUploadError(
                    "remote checkpoint id is already committed to different local content"
                )
            self._verify_remote(existing)
            result = self._result(
                existing, status="reused", manifest_bytes=existing_raw
            )
            self._write_local_marker(checkpoint_dir, result)
            logger.info("Reused verified TOS checkpoint %s", result["destination"])
            return result
        crc_by_path = self._upload_payloads(
            checkpoint_dir,
            base_key=self._base_key(run_name, checkpoint_id),
            state_root=Path(state_root).resolve(),
        )

        current = _local_manifest(
            checkpoint_dir,
            run_name=run_name,
            checkpoint_id=checkpoint_id,
            source_world_size=source_world_size,
        )
        if current != local:
            raise TosUploadError(
                "local checkpoint changed during upload; remote manifest was not committed"
            )

        committed = dict(local)
        committed["files"] = {
            relative: {**entry, "crc64_ecma": crc_by_path[relative]}
            for relative, entry in local["files"].items()
        }
        committed["created_at"] = time.time()
        body = self._commit_manifest(
            committed,
            run_name=run_name,
            checkpoint_id=checkpoint_id,
        )
        result = self._result(committed, status="uploaded", manifest_bytes=body)
        self._write_local_marker(checkpoint_dir, result)
        logger.info("Uploaded and verified TOS checkpoint %s", result["destination"])
        return result

    def _result(
        self, manifest: dict[str, Any], *, status: str, manifest_bytes: bytes
    ) -> dict[str, Any]:
        return {
            "status": status,
            "destination": self.destination(
                manifest["run_name"], manifest["checkpoint_id"]
            ),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "total_files": manifest["total_files"],
            "total_size": manifest["total_size"],
        }

    @staticmethod
    def _write_local_marker(checkpoint_dir: Path, result: dict[str, Any]) -> None:
        marker = checkpoint_dir / TOS_UPLOAD_MARKER
        temporary = marker.with_name(f"{marker.name}.tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, marker)


class TosUploadManager:
    """Serialize checkpoint uploads, optionally on one background worker."""

    def __init__(
        self,
        store: TosCheckpointStore,
        *,
        run_name: str,
        source_world_size: int,
        state_root: str | Path,
        asynchronous: bool = True,
    ) -> None:
        self.store = store
        self.run_name = _validate_name(run_name, "run name")
        self.source_world_size = int(source_world_size)
        self.state_root = Path(state_root)
        self.asynchronous = bool(asynchronous)
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="starwam-tos")
            if asynchronous
            else None
        )
        self._futures: list[Future[dict[str, Any]]] = []
        self._failures: list[Exception] = []
        self._lock = threading.Lock()
        self._closed = False

    def submit(self, checkpoint_dir: str | Path, step: int) -> None:
        if self._closed:
            raise RuntimeError("TOS uploader is already closed")
        self._raise_recorded_failure()
        checkpoint_dir = Path(checkpoint_dir)
        kwargs = {
            "run_name": self.run_name,
            "checkpoint_id": step,
            "source_world_size": self.source_world_size,
            "state_root": self.state_root / _checkpoint_id(step),
        }
        if self._executor is None:
            self.store.upload_checkpoint(checkpoint_dir, **kwargs)
            return
        future = self._executor.submit(
            self.store.upload_checkpoint, checkpoint_dir, **kwargs
        )
        future.add_done_callback(
            lambda item, path=checkpoint_dir: self._report_future(path, item)
        )
        self._futures.append(future)
        logger.info("Queued checkpoint for TOS upload: %s", checkpoint_dir)

    def _report_future(
        self, checkpoint_dir: Path, future: Future[dict[str, Any]]
    ) -> None:
        try:
            future.result()
        except Exception as exc:
            with self._lock:
                self._failures.append(exc)
            logger.error(
                "TOS upload failed; local checkpoint retained at %s: %s",
                checkpoint_dir,
                exc,
            )

    def _raise_recorded_failure(self) -> None:
        with self._lock:
            failure = self._failures[0] if self._failures else None
        if failure is not None:
            raise TosUploadError(
                f"a previous asynchronous TOS upload failed: {failure}"
            ) from failure

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self._raise_recorded_failure()


def _resolve_upload_config(
    training_config: TrainingConfig,
) -> tuple[TosUploadConfig, str, Path]:
    config = training_config.checkpoint_upload
    bucket = str(config.bucket or os.environ.get("TOS_BUCKET", "")).strip()
    prefix = str(config.prefix or os.environ.get("TOS_PREFIX", "")).strip()
    endpoint = config.endpoint.strip()
    region = config.region.strip()
    run_name = str(config.run_name or Path(training_config.output_dir).name)
    state_root = Path(
        config.state_dir or Path(training_config.output_dir) / ".tos-upload-state"
    )
    upload_config = TosUploadConfig(
        endpoint=endpoint,
        region=region,
        bucket=bucket,
        prefix=prefix,
        part_size=int(config.part_size_mb) * 1024 * 1024,
        task_num=int(config.task_num),
    )
    _validate_name(run_name, "checkpoint_upload.run_name")
    return upload_config, run_name, state_root


def validate_checkpoint_upload(training_config: TrainingConfig) -> None:
    """Validate an explicitly enabled TOS destination before expensive model setup."""
    if not training_config.checkpoint_upload.enabled:
        return
    upload_config, _, _ = _resolve_upload_config(training_config)
    TosCheckpointStore(upload_config).check_access()


def build_checkpoint_uploader(
    training_config: TrainingConfig, *, world_size: int
) -> TosUploadManager | None:
    config = training_config.checkpoint_upload
    if not config.enabled:
        return None
    upload_config, run_name, state_root = _resolve_upload_config(training_config)
    return TosUploadManager(
        TosCheckpointStore(upload_config),
        run_name=run_name,
        source_world_size=world_size,
        state_root=state_root,
        asynchronous=config.asynchronous,
    )
