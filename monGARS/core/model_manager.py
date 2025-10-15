"""Centralised management of LLM model configuration and provisioning."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from monGARS.config import Settings, get_settings
from monGARS.core.adapter_provisioner import AdapterProvisioner

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency during tests
    import ollama
except ImportError:  # pragma: no cover - allow lightweight deployments without Ollama
    ollama = None


@dataclass(slots=True, frozen=True)
class AdapterDefinition:
    """Description of a LoRA/adapter artifact backing a logical model role."""

    name: str
    source: str | None = None
    target: str | None = None
    checksum: str | None = None
    auto_update: bool = True
    description: str | None = None

    def resolved_target(self, registry_root: Path) -> Path:
        """Return the destination path for the adapter within the registry."""

        if self.target:
            target_path = Path(self.target).expanduser()
        else:
            target_path = Path(self.name)
        if target_path.is_absolute():
            return target_path
        return (registry_root / target_path).resolve()

    def to_payload(self) -> dict[str, Any]:
        """Serialise the adapter definition for API responses or logging."""

        return {
            "name": self.name,
            "source": self.source,
            "target": self.target,
            "checksum": self.checksum,
            "auto_update": self.auto_update,
            "description": self.description,
        }


@dataclass(slots=True, frozen=True)
class ModelDefinition:
    """Description of a single logical model role."""

    role: str
    name: str
    provider: str = "ollama"
    parameters: Mapping[str, Any] = field(default_factory=dict)
    auto_download: bool = True
    description: str | None = None
    adapters: tuple[AdapterDefinition, ...] = ()

    def merge_parameters(self, base: Mapping[str, Any]) -> dict[str, Any]:
        """Merge model-specific overrides on top of ``base`` options."""

        merged = dict(base)
        for key, value in self.parameters.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
        return merged

    def with_name(self, name: str) -> "ModelDefinition":
        """Return a copy of the definition with ``name`` updated."""

        return replace(self, name=name)

    def to_payload(self) -> dict[str, Any]:
        """Serialise the definition for API responses or logging."""

        return {
            "role": self.role,
            "name": self.name,
            "provider": self.provider,
            "parameters": dict(self.parameters),
            "auto_download": self.auto_download,
            "description": self.description,
            "adapters": [adapter.to_payload() for adapter in self.adapters],
        }


@dataclass(slots=True)
class ModelProfile:
    """Collection of model definitions grouped under a profile name."""

    name: str
    models: dict[str, ModelDefinition]

    def get(self, role: str) -> ModelDefinition | None:
        return self.models.get(role.lower())

    def to_payload(self) -> dict[str, dict[str, Any]]:
        """Return a JSON-serialisable snapshot of configured models."""

        return {
            role: definition.to_payload() for role, definition in self.models.items()
        }


@dataclass(slots=True)
class ModelProvisionStatus:
    """Status entry returned after attempting to provision a model."""

    role: str
    name: str
    provider: str
    action: str
    detail: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Serialise the provisioning status for external consumers."""

        return {
            "role": self.role,
            "name": self.name,
            "provider": self.provider,
            "action": self.action,
            "detail": self.detail,
        }


@dataclass(slots=True)
class ModelProvisionReport:
    """Summary of provisioning actions performed for a batch of roles."""

    statuses: list[ModelProvisionStatus]

    def __bool__(self) -> bool:  # pragma: no cover - convenience only
        return bool(self.statuses)

    def actions_by_role(self) -> dict[str, str]:
        """Return a mapping of role -> action for quick inspection."""

        return {status.role: status.action for status in self.statuses}

    def to_payload(self) -> dict[str, list[dict[str, Any]]]:
        """Return a JSON representation suitable for API responses."""

        return {"statuses": [status.to_payload() for status in self.statuses]}


_DEFAULT_MODELS: dict[str, ModelDefinition] = {
    "general": ModelDefinition(
        role="general",
        name="dolphin3",
        description="Dolphin 3.0 Llama 3.1 8B chat checkpoint",
    ),
    "coding": ModelDefinition(
        role="coding",
        name="dolphin3-llm2vec",
        description="LLM2Vec-wrapped Dolphin 3.0 build for coding tasks",
    ),
}


class LLMModelManager:
    """Manage model metadata and ensure required weights are available locally."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        config_path: str | Path | None = None,
        profile: str | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._config_path = Path(
            config_path
            if config_path is not None
            else self._settings.llm_models_config_path
        ).expanduser()
        if not self._config_path.is_absolute():
            self._config_path = self._config_path.resolve()
        self._config_dir = self._config_path.parent
        self._profile_name = (
            profile or self._settings.llm_models_profile
        ).strip() or "default"
        self._auto_download = bool(self._settings.llm_models_auto_download)
        self._adapter_registry_path = Path(
            self._settings.llm_adapter_registry_path
        ).expanduser()
        if not self._adapter_registry_path.is_absolute():
            self._adapter_registry_path = self._adapter_registry_path.resolve()
        self._adapter_registry_path.mkdir(parents=True, exist_ok=True)
        self._adapter_provisioner = AdapterProvisioner(
            self._adapter_registry_path, self._config_dir
        )
        self._profiles = self._load_profiles(self._config_path)
        base_profile = self._select_profile(self._profile_name)
        self._profile = self._apply_overrides(base_profile)
        self._ensure_lock = asyncio.Lock()
        self._ensured_roles: set[str] = set()

    def get_model_definition(self, role: str) -> ModelDefinition:
        """Return the configured definition for ``role`` with fallbacks."""

        normalized = role.lower()
        definition = self._profile.get(normalized)
        if definition:
            return definition
        if normalized != "general":
            fallback = self._profile.get("general")
            if fallback:
                return fallback
        return _DEFAULT_MODELS.get(normalized, _DEFAULT_MODELS["general"])

    def get_model_name(self, role: str) -> str:
        return self.get_model_definition(role).name

    def get_model_parameters(self, role: str) -> dict[str, Any]:
        definition = self.get_model_definition(role)
        if not definition.parameters:
            return {}
        return dict(definition.parameters)

    def resolve_parameter(self, role: str, key: str, default: Any) -> Any:
        params = self.get_model_parameters(role)
        if key not in params or params[key] is None:
            return default
        return params[key]

    def active_profile_name(self) -> str:
        """Return the name of the currently selected profile."""

        return self._profile.name

    def available_profile_names(self) -> list[str]:
        """Return all profile identifiers discovered in the manifest."""

        return sorted(self._profiles)

    def get_profile_snapshot(self, name: str | None = None) -> ModelProfile:
        """Return a defensive copy of the requested profile configuration."""

        if name is None:
            profile = self._profile
        else:
            profile = self._profiles.get(name)
            if profile is None:
                raise KeyError(name)
        return ModelProfile(name=profile.name, models=dict(profile.models))

    async def ensure_models_installed(
        self, roles: Iterable[str] | None = None, *, force: bool = False
    ) -> ModelProvisionReport:
        """Ensure local providers have the required models for ``roles``."""

        requested_roles = list(
            dict.fromkeys(role.lower() for role in (roles or self._all_roles()))
        )
        statuses: list[ModelProvisionStatus] = []
        async with self._ensure_lock:
            for role in requested_roles:
                definition = self.get_model_definition(role)
                if not force and role in self._ensured_roles:
                    statuses.append(
                        ModelProvisionStatus(
                            role=role,
                            name=definition.name,
                            provider=definition.provider,
                            action="skipped",
                            detail="already_ensured",
                        )
                    )
                    continue
                role_statuses = await self._ensure_provider(definition, force=force)
                statuses.extend(role_statuses)
                if any(
                    status.role == definition.role
                    and status.action in {"exists", "installed", "updated"}
                    for status in role_statuses
                ):
                    self._ensured_roles.add(role)
        return ModelProvisionReport(statuses=statuses)

    def _all_roles(self) -> list[str]:
        roles = set(_DEFAULT_MODELS.keys()) | set(self._profile.models.keys())
        return sorted(roles)

    async def _ensure_provider(
        self, definition: ModelDefinition, *, force: bool
    ) -> list[ModelProvisionStatus]:
        provider = definition.provider.lower()
        statuses: list[ModelProvisionStatus] = []
        if provider == "ollama":
            statuses.append(await self._ensure_ollama_model(definition, force=force))
        else:
            logger.info(
                "llm.models.provider.skipped",
                extra={"provider": definition.provider, "role": definition.role},
            )
            statuses.append(
                ModelProvisionStatus(
                    role=definition.role,
                    name=definition.name,
                    provider=definition.provider,
                    action="skipped",
                    detail="unsupported_provider",
                )
            )
        statuses.extend(await self._ensure_role_adapters(definition, force=force))
        return statuses

    async def _ensure_ollama_model(
        self, definition: ModelDefinition, *, force: bool
    ) -> ModelProvisionStatus:
        if not ollama:
            logger.warning(
                "llm.models.ollama.unavailable",
                extra={"role": definition.role, "model": definition.name},
            )
            return ModelProvisionStatus(
                role=definition.role,
                name=definition.name,
                provider=definition.provider,
                action="unavailable",
                detail="ollama_client_missing",
            )
        try:
            existing = await asyncio.to_thread(self._ollama_list_models)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "llm.models.ollama.list_failed",
                extra={"role": definition.role, "model": definition.name},
                exc_info=exc,
            )
            return ModelProvisionStatus(
                role=definition.role,
                name=definition.name,
                provider=definition.provider,
                action="error",
                detail="list_failed",
            )
        if definition.name in existing and not force:
            logger.info(
                "llm.models.present",
                extra={"role": definition.role, "model": definition.name},
            )
            return ModelProvisionStatus(
                role=definition.role,
                name=definition.name,
                provider=definition.provider,
                action="exists",
            )
        if not (self._auto_download and definition.auto_download):
            logger.info(
                "llm.models.download.skipped",
                extra={
                    "role": definition.role,
                    "model": definition.name,
                    "auto_download": self._auto_download,
                    "model_auto_download": definition.auto_download,
                },
            )
            return ModelProvisionStatus(
                role=definition.role,
                name=definition.name,
                provider=definition.provider,
                action="skipped",
                detail="auto_download_disabled",
            )
        action = "updated" if definition.name in existing else "installed"
        try:
            await asyncio.to_thread(ollama.pull, definition.name)
        except Exception as exc:  # pragma: no cover - unexpected provider failure
            logger.warning(
                "llm.models.download.failed",
                extra={"role": definition.role, "model": definition.name},
                exc_info=exc,
            )
            return ModelProvisionStatus(
                role=definition.role,
                name=definition.name,
                provider=definition.provider,
                action="error",
                detail="download_failed",
            )
        logger.info(
            "llm.models.download.completed",
            extra={"role": definition.role, "model": definition.name, "action": action},
        )
        return ModelProvisionStatus(
            role=definition.role,
            name=definition.name,
            provider=definition.provider,
            action=action,
        )

    async def _ensure_role_adapters(
        self, definition: ModelDefinition, *, force: bool
    ) -> list[ModelProvisionStatus]:
        if not definition.adapters:
            return []
        statuses: list[ModelProvisionStatus] = []
        for adapter in definition.adapters:
            statuses.append(
                await self._ensure_adapter(definition.role, adapter, force=force)
            )
        return statuses

    async def _ensure_adapter(
        self, role: str, adapter: AdapterDefinition, *, force: bool
    ) -> ModelProvisionStatus:
        target_path = adapter.resolved_target(self._adapter_registry_path)
        allow_download = self._auto_download and adapter.auto_update
        try:
            result = await asyncio.to_thread(
                self._adapter_provisioner.ensure_adapter,
                role,
                adapter,
                force=force,
                allow_download=allow_download,
            )
            action, detail = result.action, result.detail
            target_path = result.path
        except FileNotFoundError:
            logger.warning(
                "llm.models.adapter.missing_source",
                extra={
                    "role": role,
                    "adapter": adapter.name,
                    "source": adapter.source,
                },
            )
            return ModelProvisionStatus(
                role=f"{role}:{adapter.name}",
                name=adapter.name,
                provider="adapter",
                action="error",
                detail="source_missing",
            )
        except ValueError as exc:
            logger.warning(
                "llm.models.adapter.invalid",
                extra={
                    "role": role,
                    "adapter": adapter.name,
                    "target": str(target_path),
                },
                exc_info=exc,
            )
            return ModelProvisionStatus(
                role=f"{role}:{adapter.name}",
                name=adapter.name,
                provider="adapter",
                action="error",
                detail=str(exc) or "invalid_adapter",
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "llm.models.adapter.failed",
                extra={
                    "role": role,
                    "adapter": adapter.name,
                    "target": str(target_path),
                },
                exc_info=exc,
            )
            return ModelProvisionStatus(
                role=f"{role}:{adapter.name}",
                name=adapter.name,
                provider="adapter",
                action="error",
                detail="adapter_install_failed",
            )
        logger.info(
            "llm.models.adapter.status",
            extra={
                "role": role,
                "adapter": adapter.name,
                "target": str(target_path),
                "action": action,
            },
        )
        return ModelProvisionStatus(
            role=f"{role}:{adapter.name}",
            name=adapter.name,
            provider="adapter",
            action=action,
            detail=detail,
        )

    def _ollama_list_models(self) -> set[str]:
        response = ollama.list()
        models = response.get("models") if isinstance(response, Mapping) else response
        names: set[str] = set()
        if isinstance(models, Mapping):
            models = models.values()
        if not models:
            return names
        for item in models:
            if isinstance(item, Mapping):
                name = item.get("name") or item.get("model")
            else:
                name = str(item)
            if name:
                names.add(str(name))
        return names

    def _load_profiles(self, path: Path) -> dict[str, ModelProfile]:
        if not path.exists():
            logger.info(
                "llm.models.config.missing",
                extra={"path": str(path)},
            )
            return {
                "default": ModelProfile(name="default", models=_DEFAULT_MODELS.copy())
            }
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            logger.warning(
                "llm.models.config.invalid",
                extra={"path": str(path)},
                exc_info=exc,
            )
            return {
                "default": ModelProfile(name="default", models=_DEFAULT_MODELS.copy())
            }
        return self._parse_profiles(data)

    def _parse_profiles(self, payload: Any) -> dict[str, ModelProfile]:
        profiles: dict[str, ModelProfile] = {}
        if isinstance(payload, Mapping):
            raw_profiles = payload.get("profiles")
            if isinstance(raw_profiles, Mapping):
                for name, profile_payload in raw_profiles.items():
                    profile = self._parse_profile(name, profile_payload)
                    if profile:
                        profiles[profile.name] = profile
            if not profiles:
                profile = self._parse_profile("default", payload)
                if profile:
                    profiles[profile.name] = profile
        if not profiles:
            profiles["default"] = ModelProfile(
                name="default", models=_DEFAULT_MODELS.copy()
            )
        return profiles

    def _parse_profile(self, name: str, payload: Any) -> ModelProfile | None:
        if not isinstance(payload, Mapping):
            return None
        models_payload = payload.get("models")
        if not isinstance(models_payload, Mapping):
            return None
        models: dict[str, ModelDefinition] = {}
        for role, definition_payload in models_payload.items():
            definition = self._parse_model_definition(role, definition_payload)
            if definition:
                models[role.lower()] = definition
        if not models:
            return None
        return ModelProfile(name=name, models=models)

    def _parse_model_definition(
        self, role: str, payload: Any
    ) -> ModelDefinition | None:
        if isinstance(payload, str):
            role_key = role.lower()
            base_definition = _DEFAULT_MODELS.get(role_key, _DEFAULT_MODELS["general"])
            return replace(base_definition, role=role_key, name=str(payload))
        if not isinstance(payload, Mapping):
            return None
        name_value = payload.get("name") or payload.get("model") or payload.get("id")
        if not name_value:
            return None
        provider = str(payload.get("provider", "ollama"))
        raw_parameters = payload.get("parameters") or payload.get("options") or {}
        parameters: dict[str, Any]
        if isinstance(raw_parameters, Mapping):
            parameters = {str(key): raw_parameters[key] for key in raw_parameters}
        else:
            parameters = {}
        auto_download = payload.get("auto_download")
        if auto_download is None:
            auto_download_flag = True
        elif isinstance(auto_download, str):
            auto_download_flag = auto_download.strip().lower() in {
                "true",
                "1",
                "yes",
                "on",
            }
        else:
            auto_download_flag = bool(auto_download)
        description = payload.get("description")
        adapters_payload = payload.get("adapters") or payload.get("adapter")
        if isinstance(adapters_payload, (list, tuple)):
            adapters_iterable = adapters_payload
        elif adapters_payload is None:
            adapters_iterable = []
        else:
            adapters_iterable = [adapters_payload]
        adapters: list[AdapterDefinition] = []
        for adapter_payload in adapters_iterable:
            if adapter_definition := self._parse_adapter_definition(
                role, adapter_payload
            ):
                adapters.append(adapter_definition)
        return ModelDefinition(
            role=role.lower(),
            name=str(name_value),
            provider=provider,
            parameters=parameters,
            auto_download=auto_download_flag,
            description=str(description) if description else None,
            adapters=tuple(adapters),
        )

    def _parse_adapter_definition(
        self, role: str, payload: Any
    ) -> AdapterDefinition | None:
        if isinstance(payload, str):
            name = Path(payload).stem or payload
            return AdapterDefinition(name=name, source=str(payload))
        if not isinstance(payload, Mapping):
            return None
        name_value = (
            payload.get("name")
            or payload.get("id")
            or payload.get("adapter")
            or payload.get("target")
            or f"{role}-adapter"
        )
        auto_update_value = payload.get("auto_update")
        if auto_update_value is None:
            auto_update_flag = True
        elif isinstance(auto_update_value, str):
            auto_update_flag = auto_update_value.strip().lower() in {
                "true",
                "1",
                "yes",
                "on",
            }
        else:
            auto_update_flag = bool(auto_update_value)
        source_value = (
            payload.get("source")
            or payload.get("url")
            or payload.get("location")
            or payload.get("download")
        )
        target_value = (
            payload.get("target")
            or payload.get("destination")
            or payload.get("install_path")
        )
        path_value = payload.get("path")
        if source_value is None and path_value is not None:
            source_value = path_value
        elif target_value is None and path_value is not None:
            target_value = path_value
        checksum_value = payload.get("checksum")
        description_value = payload.get("description")
        return AdapterDefinition(
            name=str(name_value),
            source=str(source_value) if source_value else None,
            target=str(target_value) if target_value else None,
            checksum=str(checksum_value) if checksum_value else None,
            auto_update=auto_update_flag,
            description=str(description_value) if description_value else None,
        )

    def _select_profile(self, name: str) -> ModelProfile:
        profile = self._profiles.get(name)
        if profile:
            return profile
        logger.warning(
            "llm.models.profile.defaulted",
            extra={
                "requested_profile": name,
                "available_profiles": list(self._profiles),
            },
        )
        return self._profiles.get("default") or next(iter(self._profiles.values()))

    def _apply_overrides(self, profile: ModelProfile) -> ModelProfile:
        models = dict(profile.models)
        general_override = (self._settings.llm_general_model or "").strip()
        if general_override:
            base_general = models.get("general", _DEFAULT_MODELS["general"])
            models["general"] = base_general.with_name(general_override)
        coding_override = (self._settings.llm_coding_model or "").strip()
        if coding_override:
            base_coding = (
                models.get("coding")
                or models.get("general")
                or _DEFAULT_MODELS.get("coding", _DEFAULT_MODELS["general"])
            )
            models["coding"] = base_coding.with_name(coding_override)
        return ModelProfile(name=profile.name, models=models)


__all__ = [
    "LLMModelManager",
    "AdapterDefinition",
    "ModelDefinition",
    "ModelProfile",
    "ModelProvisionReport",
    "ModelProvisionStatus",
]
