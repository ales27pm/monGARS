"""Tests for the Unsloth diagnostics helper."""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest

from monGARS.mlops.diagnostics import analysis, cli


def _bytes_payload(num_bytes: int) -> dict[str, float]:
    return {
        "bytes": float(num_bytes),
        "mib": float(num_bytes) / 1024**2,
        "gib": float(num_bytes) / 1024**3,
    }


def _install_fake_unsloth(monkeypatch: Any) -> None:
    module = types.ModuleType("unsloth")
    module.__version__ = "1.2.3"
    module.__file__ = "/tmp/unsloth/__init__.py"
    monkeypatch.setitem(sys.modules, "unsloth", module)


def _install_fake_llm_integration(
    monkeypatch: Any, *, return_value: dict[str, Any]
) -> None:
    core_pkg = types.ModuleType("monGARS.core")
    core_pkg.__path__ = []  # mark as package

    pkg = types.ModuleType("monGARS")
    pkg.__path__ = []
    pkg.core = core_pkg

    llm_module = types.ModuleType("monGARS.core.llm_integration")

    def _fake_initialize_unsloth(force: bool = False) -> dict[str, Any]:
        _fake_initialize_unsloth.last_force = force  # type: ignore[attr-defined]
        return return_value

    llm_module.initialize_unsloth = _fake_initialize_unsloth  # type: ignore[attr-defined]

    core_pkg.llm_integration = llm_module

    monkeypatch.setitem(sys.modules, "monGARS", pkg)
    monkeypatch.setitem(sys.modules, "monGARS.core", core_pkg)
    monkeypatch.setitem(sys.modules, "monGARS.core.llm_integration", llm_module)


def test_main_outputs_extended_payload(monkeypatch, capsys):
    _install_fake_llm_integration(
        monkeypatch, return_value={"available": True, "patched": True}
    )
    _install_fake_unsloth(monkeypatch)

    original_import_optional = cli.import_optional
    monkeypatch.setattr(
        cli,
        "import_optional",
        lambda name: None if name == "torch" else original_import_optional(name),
    )

    exit_code = cli.main(["--no-cuda"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["unsloth"]["patched"] is True
    assert payload["environment"]["unsloth"]["available"] is True
    assert payload["environment"]["torch"]["available"] is False
    assert payload["cuda"] is None
    assert payload["analysis"]["oom_risk"]["status"] == "unknown"
    assert payload["analysis"]["oom_risk"]["reason"] == "cuda_diagnostics_disabled"


def test_cli_rejects_non_positive_thresholds():
    with pytest.raises(SystemExit):
        cli._parse_args(["--min-free-gib", "0"])

    with pytest.raises(SystemExit):
        cli._parse_args(["--min-free-ratio", "0"])


def test_force_flag_is_forwarded(monkeypatch, capsys):
    return_state = {"available": True, "patched": False}
    _install_fake_llm_integration(monkeypatch, return_value=return_state)

    exit_code = cli.main(["--no-cuda", "--force"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["unsloth"]["patched"] is False
    fake_module = sys.modules["monGARS.core.llm_integration"]
    assert getattr(fake_module.initialize_unsloth, "last_force") is True  # type: ignore[attr-defined]
    assert payload["analysis"]["oom_risk"]["status"] == "unknown"


def test_oom_analysis_classifies_critical():
    cuda_payload = {
        "devices": [
            {
                "index": 0,
                "memory_bytes": {
                    "free": _bytes_payload(256 * 1024 * 1024),
                    "total": _bytes_payload(8 * 1024 * 1024 * 1024),
                    "reserved": _bytes_payload(6 * 1024 * 1024 * 1024),
                    "allocated": _bytes_payload(5 * 1024 * 1024 * 1024),
                },
            }
        ]
    }

    oom_analysis = analysis.analyse_cuda_state(
        cuda_payload,
        min_free_gib=1.0,
        min_free_ratio=0.1,
        skip_reason=None,
    )

    device_report = oom_analysis["devices"][0]
    assert oom_analysis["status"] == "critical"
    assert device_report["status"] == "critical"
    assert any(
        "max_seq_length" in recommendation
        for recommendation in device_report["recommendations"]
    )


def test_oom_analysis_classifies_warning():
    cuda_payload = {
        "devices": [
            {
                "index": 0,
                "memory_bytes": {
                    "free": _bytes_payload(int(0.25 * 1024**3)),
                    "total": _bytes_payload(1 * 1024**3),
                    "reserved": _bytes_payload(300 * 1024**2),
                    "allocated": _bytes_payload(200 * 1024**2),
                },
            }
        ]
    }

    result = analysis.analyse_cuda_state(
        cuda_payload,
        min_free_gib=0.2,
        min_free_ratio=0.2,
        skip_reason=None,
    )

    device_report = result["devices"][0]
    assert result["status"] == "warning"
    assert device_report["status"] == "warning"
    assert any("offloading" in rec.lower() for rec in device_report["recommendations"])


def test_oom_analysis_classifies_ok():
    cuda_payload = {
        "devices": [
            {
                "index": 0,
                "memory_bytes": {
                    "free": _bytes_payload(2 * 1024**3),
                    "total": _bytes_payload(4 * 1024**3),
                    "reserved": _bytes_payload(1 * 1024**3),
                    "allocated": _bytes_payload(512 * 1024**2),
                },
            }
        ]
    }

    result = analysis.analyse_cuda_state(
        cuda_payload,
        min_free_gib=1.0,
        min_free_ratio=0.3,
        skip_reason=None,
    )

    device_report = result["devices"][0]
    assert result["status"] == "ok"
    assert device_report["status"] == "ok"
    assert not device_report["recommendations"]


def test_oom_analysis_surfaces_invalid_indices():
    cuda_payload = {
        "devices": [
            {
                "index": 0,
                "memory_bytes": {
                    "free": _bytes_payload(2 * 1024**3),
                    "total": _bytes_payload(4 * 1024**3),
                    "reserved": _bytes_payload(3 * 1024**3),
                    "allocated": _bytes_payload(2 * 1024**3),
                },
            }
        ],
        "invalid_indices": [5],
    }

    result = analysis.analyse_cuda_state(
        cuda_payload,
        min_free_gib=0.5,
        min_free_ratio=0.2,
        skip_reason=None,
    )

    assert result["invalid_indices"] == [5]
