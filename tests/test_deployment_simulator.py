"""Tests for the deployment simulator tooling."""

from __future__ import annotations

import yaml

from scripts.deployment_simulator import DeploymentSimulator, _extract_host_port


def test_evaluate_compose_detects_common_misconfigurations(tmp_path) -> None:
    simulator = DeploymentSimulator(root=tmp_path)

    compose_dict = {
        "services": {
            "api": {
                "image": "example/api:latest",
                "ports": ["8000:8000"],
                "env_file": ["./.env.api"],
                "volumes": ["./data:/var/data"],
            },
            "worker": {
                "build": {"context": "./worker"},
                "ports": ["127.0.0.1:8000:8001"],
            },
        }
    }

    compose_path = tmp_path / "docker-compose.yml"
    compose_path.write_text(yaml.safe_dump(compose_dict))

    issues = simulator._evaluate_compose_dict(compose_dict, compose_path)

    messages = {issue.message for issue in issues}
    assert any("env_file" in issue.message for issue in issues)
    assert any("Host volume path" in issue.message for issue in issues)
    assert any("Host port is published" in issue.message for issue in issues)
    assert "Service is missing both image and build directives." not in messages


def test_simulate_settings_flags_missing_secret(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SECRET_KEY", "")
    (tmp_path / ".env").write_text("")
    simulator = DeploymentSimulator(root=tmp_path)
    issues = simulator._simulate_settings()

    assert any(
        issue.severity == "error"
        and ("SECRET_KEY" in issue.message or "SECRET_KEY" in repr(issue.context))
        for issue in issues
    ), [issue.as_dict() for issue in issues]


def test_kubernetes_validation_warns_on_missing_health_probes(tmp_path) -> None:
    manifest = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "mongars"},
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "api",
                            "image": "example/api:latest",
                        }
                    ]
                }
            }
        },
    }

    simulator = DeploymentSimulator(root=tmp_path)
    issues = simulator._evaluate_kubernetes_deployment(manifest, tmp_path / "k8s.yaml")

    assert any("resource" in issue.message for issue in issues)
    assert any("health probes" in issue.message for issue in issues)


def test_extract_host_port_handles_various_bindings() -> None:
    assert _extract_host_port("8080:8000") == "8080"
    assert _extract_host_port("127.0.0.1:9000:9000") == "9000"
    assert _extract_host_port({"published": 7000, "target": 7000}) == "7000"
    assert _extract_host_port({"host_port": "6000"}) == "6000"
