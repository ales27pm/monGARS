"""Tests for the QLoRA build-and-wrap pipeline helpers."""

from __future__ import annotations

import pytest

from build_and_wrap import evaluate_oom_headroom


class _CudaShim:
    def __init__(self, available: bool = True, device_count: int = 1) -> None:
        self._available = available
        self._device_count = device_count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._device_count


class _TorchShim:
    def __init__(self, *, available: bool = True, device_count: int = 1) -> None:
        self.cuda = _CudaShim(available=available, device_count=device_count)


def test_evaluate_oom_headroom_handles_cuda_unavailable() -> None:
    captured = {}

    def fake_analyse(
        payload, *, min_free_gib: float, min_free_ratio: float, skip_reason: str | None
    ) -> dict[str, object]:
        captured["payload"] = payload
        captured["skip_reason"] = skip_reason
        return {"status": "unknown", "reason": skip_reason, "thresholds": {}}

    analysis = evaluate_oom_headroom(
        torch_module=_TorchShim(available=False),
        gather_metrics=lambda module: {"devices": []},
        analyse_state=fake_analyse,
        fail_on_critical=False,
    )

    assert analysis["status"] == "unknown"
    assert captured["payload"] is None
    assert captured["skip_reason"] == "cuda_unavailable"


def test_evaluate_oom_headroom_handles_torch_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("build_and_wrap.torch", None)
    captured = {}

    def fake_analyse(
        payload, *, min_free_gib: float, min_free_ratio: float, skip_reason: str | None
    ) -> dict[str, object]:
        captured["payload"] = payload
        captured["skip_reason"] = skip_reason
        return {"status": "unknown", "reason": skip_reason, "thresholds": {}}

    analysis = evaluate_oom_headroom(
        torch_module=None,
        gather_metrics=lambda module: {"devices": []},
        analyse_state=fake_analyse,
        fail_on_critical=False,
    )

    assert analysis["status"] == "unknown"
    assert captured["payload"] is None
    assert captured["skip_reason"] == "torch_missing"


def test_evaluate_oom_headroom_handles_cuda_interface_missing() -> None:
    class TorchMissingCuda:
        pass

    captured = {}

    def fake_analyse(
        payload, *, min_free_gib: float, min_free_ratio: float, skip_reason: str | None
    ) -> dict[str, object]:
        captured["payload"] = payload
        captured["skip_reason"] = skip_reason
        return {"status": "unknown", "reason": skip_reason, "thresholds": {}}

    analysis = evaluate_oom_headroom(
        torch_module=TorchMissingCuda(),
        gather_metrics=lambda module: {"devices": []},
        analyse_state=fake_analyse,
        fail_on_critical=False,
    )

    assert analysis["status"] == "unknown"
    assert captured["payload"] is None
    assert captured["skip_reason"] == "cuda_interface_missing"


def test_evaluate_oom_headroom_raises_on_critical() -> None:
    torch_shim = _TorchShim()

    def fake_gather(module: _TorchShim) -> dict[str, object]:
        return {
            "devices": [
                {
                    "index": 0,
                    "memory_bytes": {
                        "free": {"gib": 0.1},
                        "total": {"gib": 10.0},
                        "reserved": {"bytes": 2},
                        "allocated": {"bytes": 1},
                    },
                }
            ]
        }

    def fake_analyse(
        payload, *, min_free_gib: float, min_free_ratio: float, skip_reason: str | None
    ) -> dict[str, object]:
        assert payload is not None
        return {
            "status": "critical",
            "devices": [
                {
                    "index": 0,
                    "free_gib": 0.1,
                    "free_ratio": 0.01,
                    "status": "critical",
                    "recommendations": ["Reduce batch size"],
                }
            ],
            "thresholds": {
                "min_free_gib": min_free_gib,
                "min_free_ratio": min_free_ratio,
            },
        }

    with pytest.raises(RuntimeError) as excinfo:
        evaluate_oom_headroom(
            torch_module=torch_shim,
            gather_metrics=fake_gather,
            analyse_state=fake_analyse,
            fail_on_critical=True,
            min_free_gib=1.0,
            min_free_ratio=0.1,
        )

    message = str(excinfo.value)
    assert "critical" in message.lower()
    assert "reduce batch size" in message.lower()
