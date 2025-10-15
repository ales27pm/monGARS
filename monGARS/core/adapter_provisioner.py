"""Utilities for provisioning and synchronising adapter artefacts."""

from __future__ import annotations

import logging
import hashlib
import os
import shutil
import stat
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable
from urllib.parse import urlparse
from urllib.request import urlopen

if TYPE_CHECKING:  # pragma: no cover - for static analysis only
    from monGARS.core.model_manager import AdapterDefinition


logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class AdapterSyncResult:
    """Outcome returned after attempting to materialise an adapter."""

    action: str
    detail: str | None
    path: Path


class AdapterProvisioner:
    """Handle adapter download, extraction, and installation into the registry."""

    def __init__(self, registry_root: Path, config_dir: Path) -> None:
        self._registry_root = registry_root
        self._config_dir = config_dir

    def ensure_adapter(
        self,
        role: str,
        adapter: "AdapterDefinition",
        *,
        force: bool,
        allow_download: bool,
    ) -> AdapterSyncResult:
        logger.info(
            "llm.adapters.ensure.start",
            extra={
                "adapter": adapter.name,
                "role": role,
                "force": force,
                "allow_download": allow_download,
            },
        )
        target_path = adapter.resolved_target(self._registry_root)
        action, detail = self._sync_adapter(adapter, target_path, force, allow_download)
        logger.info(
            "llm.adapters.ensure.completed",
            extra={
                "adapter": adapter.name,
                "role": role,
                "action": action,
                "detail": detail,
                "target": str(target_path),
            },
        )
        return AdapterSyncResult(action=action, detail=detail, path=target_path)

    def _sync_adapter(
        self,
        adapter: "AdapterDefinition",
        target_path: Path,
        force: bool,
        allow_download: bool,
    ) -> tuple[str, str | None]:
        target_exists = target_path.exists()
        local_source = self._resolve_adapter_source_path(adapter)
        same_location = (
            local_source is not None
            and target_path.exists()
            and local_source.resolve() == target_path.resolve()
        )

        existing_digest = None
        if target_exists and adapter.checksum:
            existing_digest = self._hash_path(target_path)

        if target_exists and not force:
            if adapter.checksum and existing_digest == adapter.checksum:
                logger.info(
                    "llm.adapters.ensure.noop",
                    extra={
                        "adapter": adapter.name,
                        "target": str(target_path),
                        "reason": "checksum_match",
                    },
                )
                return "exists", target_path.as_posix()
            if not allow_download or not adapter.auto_update:
                logger.info(
                    "llm.adapters.ensure.noop",
                    extra={
                        "adapter": adapter.name,
                        "target": str(target_path),
                        "reason": "updates_disabled",
                    },
                )
                return "exists", target_path.as_posix()

        if target_exists and not allow_download:
            logger.info(
                "llm.adapters.ensure.noop",
                extra={
                    "adapter": adapter.name,
                    "target": str(target_path),
                    "reason": "download_disallowed",
                },
            )
            return "exists", target_path.as_posix()

        if not target_exists and not allow_download:
            logger.info(
                "llm.adapters.ensure.skipped",
                extra={
                    "adapter": adapter.name,
                    "target": str(target_path),
                    "reason": "download_disallowed",
                },
            )
            return "skipped", "auto_download_disabled"

        if adapter.source is None:
            if target_exists:
                logger.warning(
                    "llm.adapters.ensure.source.missing",
                    extra={
                        "adapter": adapter.name,
                        "target": str(target_path),
                        "condition": "reuse_existing",
                    },
                )
                return "exists", target_path.as_posix()
            raise FileNotFoundError(adapter.name)

        if target_exists and force and not same_location:
            logger.info(
                "llm.adapters.ensure.remove_existing",
                extra={
                    "adapter": adapter.name,
                    "target": str(target_path),
                },
            )
            self._remove_path(target_path)
            target_exists = False

        action = "updated" if target_exists else "installed"
        self._materialise_adapter_from_source(adapter, target_path, local_source)
        if adapter.checksum:
            digest = self._hash_path(target_path)
            if digest != adapter.checksum:
                logger.warning(
                    "llm.adapters.ensure.checksum_mismatch",
                    extra={
                        "adapter": adapter.name,
                        "target": str(target_path),
                    },
                )
                raise ValueError("checksum_mismatch")
            logger.debug(
                "llm.adapters.ensure.checksum_verified",
                extra={
                    "adapter": adapter.name,
                    "target": str(target_path),
                },
            )
        return action, target_path.as_posix()

    def _resolve_adapter_source_path(self, adapter: "AdapterDefinition") -> Path | None:
        source = adapter.source
        if source is None:
            return None
        parsed = urlparse(source)
        if parsed.scheme in {"http", "https"}:
            return None
        if parsed.scheme == "file":
            source_path = Path(os.path.join(parsed.netloc, parsed.path))
        else:
            source_path = Path(source)
        source_path = source_path.expanduser()
        if not source_path.is_absolute():
            source_path = (self._config_dir / source_path).resolve()
        else:
            source_path = source_path.resolve()
        logger.debug(
            "llm.adapters.ensure.local_source",
            extra={
                "adapter": adapter.name,
                "source": str(source_path),
            },
        )
        return source_path

    def _materialise_adapter_from_source(
        self,
        adapter: "AdapterDefinition",
        target_path: Path,
        local_source: Path | None,
    ) -> None:
        source = adapter.source
        if source is None:
            raise FileNotFoundError(adapter.name)
        parsed = urlparse(source)
        if parsed.scheme in {"http", "https"}:
            with tempfile.TemporaryDirectory(prefix="adapter_dl_") as tmp_dir:
                tmp_path = Path(tmp_dir)
                filename = Path(parsed.path).name or adapter.name
                download_path = tmp_path / filename
                logger.info(
                    "llm.adapters.download.start",
                    extra={
                        "adapter": adapter.name,
                        "url": source,
                    },
                )
                self._download_remote_file(source, download_path)
                logger.info(
                    "llm.adapters.download.completed",
                    extra={
                        "adapter": adapter.name,
                        "path": str(download_path),
                    },
                )
                self._install_from_filesystem(download_path, target_path)
            return
        source_path = local_source or self._resolve_adapter_source_path(adapter)
        if source_path is None:
            raise FileNotFoundError(adapter.name)
        if source_path.resolve() == target_path.resolve():
            logger.debug(
                "llm.adapters.install.same_location",
                extra={
                    "adapter": adapter.name,
                    "target": str(target_path),
                },
            )
            target_path.mkdir(parents=True, exist_ok=True)
            return
        if not source_path.exists():
            raise FileNotFoundError(str(source_path))
        logger.info(
            "llm.adapters.install.local_source",
            extra={
                "adapter": adapter.name,
                "source": str(source_path),
                "target": str(target_path),
            },
        )
        self._install_from_filesystem(source_path, target_path)

    def _download_remote_file(self, url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("unsupported_url_scheme")
        with urlopen(url, timeout=30) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)

    def _install_from_filesystem(self, source_path: Path, target_path: Path) -> None:
        suffixes = [suffix.lower() for suffix in source_path.suffixes]
        logger.debug(
            "llm.adapters.install.from_filesystem",
            extra={
                "source": str(source_path),
                "target": str(target_path),
                "suffixes": suffixes,
            },
        )
        if source_path.is_dir():
            self._populate_directory_from_source(source_path, target_path)
            return
        if self._is_zip_archive(suffixes):
            self._extract_archive(source_path, target_path, archive_type="zip")
            return
        if self._is_tar_archive(suffixes):
            self._extract_archive(source_path, target_path, archive_type="tar")
            return
        self._populate_directory_from_source(source_path, target_path)

    def _extract_archive(
        self, archive_path: Path, target_path: Path, *, archive_type: str
    ) -> None:
        logger.info(
            "llm.adapters.archive.extract.start",
            extra={
                "archive": str(archive_path),
                "target": str(target_path),
                "archive_type": archive_type,
            },
        )
        with tempfile.TemporaryDirectory(prefix="adapter_archive_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            if archive_type == "zip":
                with zipfile.ZipFile(archive_path) as zip_file:
                    self._safe_extract_zip(zip_file, tmp_path)
            else:
                with tarfile.open(archive_path) as tar_file:
                    self._safe_extract_tar(tar_file, tmp_path)
            root = self._discover_content_root(tmp_path)
            self._populate_directory_from_source(root, target_path)
        logger.info(
            "llm.adapters.archive.extract.completed",
            extra={
                "archive": str(archive_path),
                "target": str(target_path),
                "archive_type": archive_type,
            },
        )

    def _safe_extract_zip(self, zip_file: zipfile.ZipFile, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        safe_infos: list[zipfile.ZipInfo] = []
        members: list[Path] = []
        for info in zip_file.infolist():
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise ValueError("archive_contains_links")
            member_path = destination / info.filename
            safe_infos.append(info)
            members.append(member_path)
        self._validate_archive_targets(destination, members)
        for info in safe_infos:
            member_path = destination / info.filename
            if info.is_dir():
                member_path.mkdir(parents=True, exist_ok=True)
                continue
            member_path.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(info) as source, member_path.open("wb") as target:
                shutil.copyfileobj(source, target)
        logger.debug(
            "llm.adapters.archive.zip.safe_extract",
            extra={"destination": str(destination), "members": len(safe_infos)},
        )

    def _safe_extract_tar(self, tar_file: tarfile.TarFile, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        safe_members: list[tarfile.TarInfo] = []
        members: list[Path] = []
        for member in tar_file.getmembers():
            if member.issym() or member.islnk():
                raise ValueError("archive_contains_links")
            member_path = destination / member.name
            safe_members.append(member)
            members.append(member_path)
        self._validate_archive_targets(destination, members)
        for member in safe_members:
            member_path = destination / member.name
            if member.isdir():
                member_path.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError("unsupported_archive_entry")
            member_path.parent.mkdir(parents=True, exist_ok=True)
            extracted = tar_file.extractfile(member)
            if extracted is None:
                raise ValueError("invalid_archive_entry")
            with extracted, member_path.open("wb") as target:
                shutil.copyfileobj(extracted, target)
        logger.debug(
            "llm.adapters.archive.tar.safe_extract",
            extra={"destination": str(destination), "members": len(safe_members)},
        )

    def _validate_archive_targets(
        self, destination: Path, members: Iterable[Path]
    ) -> None:
        destination_path = destination.resolve()
        for member in members:
            if not self._is_within_directory(destination_path, member):
                logger.warning(
                    "llm.adapters.archive.path_traversal",
                    extra={
                        "destination": str(destination_path),
                        "member": str(member),
                    },
                )
                raise ValueError("archive_path_traversal")

    @staticmethod
    def _is_zip_archive(suffixes: list[str]) -> bool:
        return ".zip" in suffixes

    @staticmethod
    def _is_tar_archive(suffixes: list[str]) -> bool:
        suffix_set = set(suffixes)
        return (
            ".tgz" in suffix_set
            or ".tar" in suffix_set
            or {".tar", ".gz"}.issubset(suffix_set)
        )

    @staticmethod
    def _is_within_directory(directory: Path, target: Path) -> bool:
        directory_path = directory.resolve()
        target_path = target.resolve(strict=False)
        try:
            common = os.path.commonpath([str(directory_path), str(target_path)])
        except ValueError:
            return False
        return common == str(directory_path)

    def _discover_content_root(self, directory: Path) -> Path:
        entries = [
            child for child in sorted(directory.iterdir()) if child.name != "__MACOSX"
        ]
        if not entries:
            return directory
        return entries[0] if len(entries) == 1 else directory

    def _populate_directory_from_source(self, source: Path, destination: Path) -> None:
        if destination.exists():
            self._remove_path(destination)
        if source.is_symlink():
            raise ValueError("archive_contains_links")
        if source.is_dir():
            for path in source.rglob("*"):
                if path.is_symlink():
                    raise ValueError("archive_contains_links")
            destination.mkdir(parents=True, exist_ok=True)
            for child in sorted(source.iterdir()):
                target = destination / child.name
                if child.is_dir():
                    shutil.copytree(child, target, dirs_exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(child, target)
        else:
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination / source.name)
        logger.debug(
            "llm.adapters.install.populated",
            extra={
                "source": str(source),
                "destination": str(destination),
            },
        )

    def _remove_path(self, path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()
        logger.debug(
            "llm.adapters.install.removed",
            extra={"path": str(path)},
        )

    def _hash_path(self, path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(str(path))
        digest = hashlib.sha256()
        if path.is_file():
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(65536), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        for file_path in sorted(
            p for p in path.rglob("*") if p.is_file() and not p.is_symlink()
        ):
            digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
            with file_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(65536), b""):
                    digest.update(chunk)
        logger.debug(
            "llm.adapters.install.hash_computed",
            extra={"path": str(path)},
        )
        return digest.hexdigest()


__all__ = ["AdapterProvisioner", "AdapterSyncResult"]
