from __future__ import annotations

from typing import Any

class _DeploymentSpec:
    replicas: int | None

class _Deployment:
    spec: _DeploymentSpec

class AppsV1Api:
    def read_namespaced_deployment(
        self,
        name: str,
        namespace: str,
        *args: Any,
        **kwargs: Any,
    ) -> _Deployment: ...

    def patch_namespaced_deployment(
        self,
        name: str,
        namespace: str,
        body: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...

class _ApiException(Exception):
    ...

class _ClientExceptionsModule:
    ApiException: type[_ApiException]

class _ClientModule:
    AppsV1Api: type[AppsV1Api]
    exceptions: _ClientExceptionsModule

class _ConfigException(Exception):
    ...

class _ConfigModule:
    ConfigException: type[_ConfigException]

    def load_incluster_config(self) -> None: ...

    def load_kube_config(self) -> None: ...

client: _ClientModule
config: _ConfigModule

__all__ = ["client", "config", "AppsV1Api"]
