"""Tests for the Unsloth diagnostics helper."""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import scripts.diagnose_unsloth as diagnose


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

    original_import_optional = diagnose._import_optional

    def _fake_import_optional(name: str):
        if name == "torch":
            return None
        return original_import_optional(name)

    monkeypatch.setattr(diagnose, "_import_optional", _fake_import_optional)

    exit_code = diagnose.main(["--no-cuda"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["unsloth"]["patched"] is True
    assert payload["environment"]["unsloth"]["available"] is True
    assert payload["environment"]["torch"]["available"] is False
    assert payload["cuda"] is None
    assert payload["analysis"]["oom_risk"]["status"] == "unknown"
    assert payload["analysis"]["oom_risk"]["reason"] == "cuda_diagnostics_disabled"


def test_force_flag_is_forwarded(monkeypatch, capsys):
    return_state = {"available": True, "patched": False}
    _install_fake_llm_integration(monkeypatch, return_value=return_state)

    exit_code = diagnose.main(["--no-cuda", "--force"])

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
                    "free": diagnose._format_bytes(256 * 1024 * 1024),
                    "total": diagnose._format_bytes(8 * 1024 * 1024 * 1024),
                    "reserved": diagnose._format_bytes(6 * 1024 * 1024 * 1024),
                    "allocated": diagnose._format_bytes(5 * 1024 * 1024 * 1024),
                },
            }
        ]
    }

    analysis = diagnose._analyse_cuda_state(
        cuda_payload,
        min_free_gib=1.0,
        min_free_ratio=0.1,
        skip_reason=None,
    )

    device_report = analysis["devices"][0]
    assert analysis["status"] == "critical"
    assert device_report["status"] == "critical"
    assert any(
        "max_seq_length" in recommendation
        for recommendation in device_report["recommendations"]
    )
