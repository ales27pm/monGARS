import os

os.environ.setdefault("DEBUG", "1")
os.environ.setdefault("SECRET_KEY", "test-secret")

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import AsyncMock

import pytest

from modules.evolution_engine.hardware import HardwareProfile
from modules.evolution_engine.orchestrator import EvolutionOrchestrator
from monGARS.config import HardwareHeuristics, get_settings
from monGARS.core.evolution_engine import EvolutionEngine, PerformanceIssue
from monGARS.core.monitor import SystemStats

settings = get_settings()


class DummyWorkflowBackend:
    def __init__(self) -> None:
        self.flow: Callable[..., Any] | None = None
        self.schedule_parameters: dict[str, Any] | None = None
        self.run_parameters: list[dict[str, Any]] = []

    def build_flow(
        self, func: Callable[..., Any], *, name: str
    ) -> Callable[..., Any]:  # noqa: D401 - signature parity
        self.flow = func
        return func

    def ensure_schedule(
        self, flow: Callable[..., Any], *, parameters: dict[str, Any]
    ) -> None:
        self.schedule_parameters = dict(parameters)

    def run(
        self, flow: Callable[..., Any], *, parameters: dict[str, Any]
    ) -> Any:
        self.run_parameters.append(dict(parameters))
        return flow(**parameters)


def _mock_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "modules.evolution_engine.orchestrator.psutil.cpu_percent",
        lambda interval=None: 5.0,
        raising=False,
    )
    monkeypatch.setattr(
        "modules.evolution_engine.orchestrator.psutil.virtual_memory",
        lambda: SimpleNamespace(percent=10.0),
        raising=False,
    )
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(
        "modules.evolution_engine.orchestrator.torch", fake_torch, raising=False
    )


def _mock_torch_vram(
    monkeypatch: pytest.MonkeyPatch,
    *,
    allocated_gb: float,
    fail_stage: str | None = None,
) -> None:
    allocated_bytes = allocated_gb * (1024**3)

    class _FakeCuda:
        def is_available(self) -> bool:
            return True

        def device_count(self) -> int:
            return 1

        def get_device_properties(self, index: int) -> SimpleNamespace:
            if fail_stage == "properties":
                raise RuntimeError("properties unavailable")
            return SimpleNamespace(name=f"cuda:{index}")

        @contextmanager
        def device(self, index: int):  # type: ignore[override]
            if fail_stage == "device":
                raise RuntimeError("device not accessible")
            yield None

        def memory_allocated(self) -> float:
            if fail_stage == "memory":
                raise RuntimeError("memory query failed")
            return allocated_bytes

    fake_torch = SimpleNamespace(cuda=_FakeCuda())
    monkeypatch.setattr(
        "modules.evolution_engine.orchestrator.torch", fake_torch, raising=False
    )


 def test_orchestrator_registers_interval_schedule(
     monkeypatch: pytest.MonkeyPatch,
 ) -> None:
     backend = DummyWorkflowBackend()

     class _NoopTrainer:
         def __init__(
             self, training_config_path: str, output_dir: str
         ) -> None:  # noqa: D401
             self.training_config_path = training_config_path
             self.output_dir = output_dir

         def fit(self, dataset: Any) -> dict[str, Any]:  # pragma: no cover - unused
             return {}

     monkeypatch.setenv("USE_RAY_SERVE", "false")
     orchestrator = EvolutionOrchestrator(
         workflow_backend=backend,
         trainer_cls=_NoopTrainer,
         data_collector=lambda: [],
         slot_manager_cls=None,
     )

     assert backend.schedule_parameters == {"force": False}
     assert backend.flow is not None
    assert orchestrator.workflow_backend is backend

def test_orchestrator_skips_training_when_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = DummyWorkflowBackend()

    monkeypatch.setattr(
        "modules.evolution_engine.orchestrator.psutil.cpu_percent",
        lambda interval=None: 95.0,
        raising=False,
    )
    monkeypatch.setattr(
        "modules.evolution_engine.orchestrator.psutil.virtual_memory",
        lambda: SimpleNamespace(percent=40.0),
        raising=False,
    )
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(
        "modules.evolution_engine.orchestrator.torch", fake_torch, raising=False
    )

    orchestrator = EvolutionOrchestrator(
        workflow_backend=backend,
        trainer_cls=lambda *args, **kwargs: None,  # type: ignore[arg-type]
        slot_manager_cls=None,
        data_collector=lambda: ["data"],
    )

    assert orchestrator.run_training_cycle() is None


def test_orchestrator_skips_when_vram_pressure(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = DummyWorkflowBackend()
    _mock_idle(monkeypatch)
    _mock_torch_vram(monkeypatch, allocated_gb=8.5)

    orchestrator = EvolutionOrchestrator(
        workflow_backend=backend,
        trainer_cls=lambda *args, **kwargs: None,  # type: ignore[arg-type]
        slot_manager_cls=None,
        data_collector=lambda: ["sample"],
    )

    assert orchestrator.run_training_cycle() is None


def test_orchestrator_trains_when_vram_query_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = DummyWorkflowBackend()
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    class RecordingTrainer:
        def __init__(self, training_config_path: str, output_dir: str) -> None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        def fit(self, dataset: Any) -> dict[str, Any]:
            return {"status": "success", "artifacts": {"adapter": "stub"}}

    _mock_idle(monkeypatch)
    _mock_torch_vram(monkeypatch, allocated_gb=1.0, fail_stage="properties")

    manifest_path = tmp_path / "manifest.json"

    def fake_update_manifest(
        registry: Path, summary: dict[str, Any], history_limit: int = 10
    ):
        manifest_path.write_text("{}", encoding="utf-8")
        return SimpleNamespace(path=manifest_path, build_payload=lambda: summary)

    monkeypatch.setattr(
        "modules.evolution_engine.orchestrator.update_manifest",
        fake_update_manifest,
    )
    monkeypatch.setattr(
        "modules.evolution_engine.orchestrator.update_ray_deployment",
        lambda payload: None,
    )

    orchestrator = EvolutionOrchestrator(
        workflow_backend=backend,
        training_config_path=str(config_path),
        trainer_cls=RecordingTrainer,
        slot_manager_cls=None,
        data_collector=lambda: ["data"],
    )

    run_dir = orchestrator.run_training_cycle(force=True)
    assert run_dir is not None


def test_orchestrator_runs_cycle_and_rolls_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = DummyWorkflowBackend()
    registry_path = tmp_path / "registry"
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    class DummySlotManager:
        enter_calls = 0

        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def __enter__(self) -> tuple[object, object]:
            DummySlotManager.enter_calls += 1
            return object(), object()

        def __exit__(self, exc_type, exc, tb) -> None:
            pass

    class RecordingTrainer:
        def __init__(self, training_config_path: str, output_dir: str) -> None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        def fit(self, dataset: Any) -> dict[str, Any]:
            adapter_dir = self.output_dir / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            weights_path = adapter_dir / "adapter.bin"
            weights_path.write_text("weights", encoding="utf-8")
            return {
                "status": "success",
                "artifacts": {
                    "adapter": str(adapter_dir),
                    "weights": str(weights_path),
                },
                "version": "cycle-1",
                "metrics": {"loss": 0.1},
            }

    rollout_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "modules.evolution_engine.orchestrator.update_ray_deployment",
        lambda payload: rollout_calls.append(payload),
    )

    class _Settings:
        USE_RAY_SERVE = True

    monkeypatch.setattr(
        "modules.evolution_engine.orchestrator.get_settings",
        lambda: _Settings(),
    )

    manifest_path = registry_path / "adapter_manifest.json"

    def fake_update_manifest(
        registry: Path, summary: dict[str, Any], history_limit: int = 10
    ):
        manifest_path.write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            path=manifest_path,
            build_payload=lambda: {
                "adapter_path": summary["artifacts"]["adapter"],
                "version": summary.get("version", ""),
            },
        )

    monkeypatch.setattr(
        "modules.evolution_engine.orchestrator.update_manifest",
        fake_update_manifest,
    )

    _mock_idle(monkeypatch)

    orchestrator = EvolutionOrchestrator(
        workflow_backend=backend,
        model_registry_path=str(registry_path),
        training_config_path=str(config_path),
        trainer_cls=RecordingTrainer,
        slot_manager_cls=DummySlotManager,
        data_collector=lambda: ["sample"],
    )

    run_dir = orchestrator.run_training_cycle()
    assert run_dir is not None
    summary_path = run_dir / "training_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "success"
    assert rollout_calls
    assert rollout_calls[0]["adapter_path"] == summary["artifacts"]["adapter"]
    assert DummySlotManager.enter_calls == 1


@pytest.mark.asyncio
async def test_diagnose_performance_detects_cpu_pressure() -> None:
    engine = EvolutionEngine()
    engine._stat_history.clear()
    engine._stat_history.extend(
        [
            SystemStats(cpu_usage=90.0, memory_usage=65.0, disk_usage=40.0),
            SystemStats(cpu_usage=92.0, memory_usage=66.0, disk_usage=41.0),
        ]
    )

    engine.monitor = AsyncMock()
    engine.monitor.get_system_stats = AsyncMock(
        return_value=SystemStats(
            cpu_usage=96.0,
            memory_usage=67.0,
            disk_usage=42.0,
            gpu_usage=None,
            gpu_memory_usage=None,
        )
    )

    issues = await engine.diagnose_performance()
    identifiers = {issue.identifier for issue in issues}

    assert "cpu_sustained_high" in identifiers


@pytest.mark.asyncio
async def test_apply_optimizations_scales_down_when_underutilized() -> None:
    engine = EvolutionEngine()
    engine._last_scale_timestamp = 0.0

    engine.diagnose_performance = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            PerformanceIssue(
                "workers_underutilized",
                "info",
                {"cpu_average": 10.0, "mem_average": 20.0, "window": 3},
            )
        ]
    )
    engine._get_worker_replicas = AsyncMock(return_value=3)  # type: ignore[method-assign]
    engine._scale_workers = AsyncMock()  # type: ignore[method-assign]
    engine._clear_caches = AsyncMock()  # type: ignore[method-assign]

    await engine.apply_optimizations()

    engine._scale_workers.assert_awaited_once()
    await_call = engine._scale_workers.await_args
    assert await_call.args[0] == -1


@pytest.mark.asyncio
async def test_apply_optimizations_clears_cache_for_memory_pressure() -> None:
    engine = EvolutionEngine()
    engine._last_scale_timestamp = 0.0

    engine.diagnose_performance = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            PerformanceIssue(
                "memory_pressure",
                "high",
                {"average": 93.0, "latest": 94.0},
            )
        ]
    )
    engine._get_worker_replicas = AsyncMock(return_value=2)  # type: ignore[method-assign]
    engine._scale_workers = AsyncMock()  # type: ignore[method-assign]
    engine._clear_caches = AsyncMock()  # type: ignore[method-assign]

    await engine.apply_optimizations()

    engine._clear_caches.assert_awaited_once()


def test_constrain_scale_delta_respects_hardware_bounds() -> None:
    engine = EvolutionEngine(
        orchestrator_factory=lambda: None, peer_communicator=None  # type: ignore[arg-type]
    )
    engine._hardware_profile = HardwareProfile(
        physical_cores=4, logical_cpus=8, total_memory_gb=4.0, gpu_count=0
    )
    max_workers = engine._hardware_profile.max_recommended_workers(settings.workers)

    assert engine._constrain_scale_delta(5, max_workers - 1) == 1
    assert engine._constrain_scale_delta(1, max_workers) == 0

    min_workers = engine._hardware_profile.min_recommended_workers()
    assert engine._constrain_scale_delta(-5, min_workers) == 0
    assert engine._constrain_scale_delta(-5, min_workers + 2) == (
        min_workers - (min_workers + 2)
    )


@pytest.mark.asyncio
async def test_train_cycle_executes_training_and_broadcasts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    summary_payload: dict[str, Any] = {
        "status": "success",
        "artifacts": {
            "adapter": str(run_dir / "adapter"),
            "weights": str(run_dir / "weights"),
        },
        "metrics": {"loss": 0.1},
    }

    class _StubOrchestrator:
        def __init__(self, output: Path, payload: dict[str, Any]) -> None:
            self._output = output
            self._payload = payload

        def trigger_encoder_training_pipeline(self) -> str:
            self._output.mkdir(parents=True, exist_ok=True)
            (self._output / "training_summary.json").write_text(
                json.dumps(self._payload)
            )
            (self._output / "energy_report.json").write_text(
                json.dumps({"energy_wh": 1.25, "backend": "psutil"})
            )
            return str(self._output)

    class _RecorderCommunicator:
        def __init__(self) -> None:
            self.local_snapshots: list[dict[str, Any]] = []
            self.broadcasts: list[dict[str, Any]] = []

        def update_local_telemetry(self, snapshot: dict[str, Any]) -> None:
            self.local_snapshots.append(snapshot)

        async def broadcast_telemetry(self, snapshot: dict[str, Any]) -> bool:
            self.broadcasts.append(snapshot)
            return True

    class _StubEventBus:
        def __init__(self) -> None:
            self.events: list[Any] = []

        async def publish(self, event: Any) -> None:
            self.events.append(event)

    bus = _StubEventBus()
    monkeypatch.setattr(
        "monGARS.core.evolution_engine.event_bus", lambda: bus, raising=False
    )

    communicator = _RecorderCommunicator()

    engine = EvolutionEngine(
        orchestrator_factory=lambda: _StubOrchestrator(run_dir, summary_payload),
        peer_communicator=communicator,  # type: ignore[arg-type]
    )
    engine.apply_optimizations = AsyncMock()

    await engine.train_cycle(user_id="user-123", version="v-test")

    assert communicator.local_snapshots
    assert communicator.broadcasts
    broadcast = communicator.broadcasts[-1]
    assert broadcast["training_version"] == "v-test"
    assert broadcast["energy"]["energy_wh"] == 1.25

    assert bus.events
    event = bus.events[-1]
    assert event.data["energy"]["energy_wh"] == 1.25


def test_hardware_profile_uses_configured_heuristics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARDWARE_HEURISTICS__BASE_POWER_DRAW", "55.5")
    monkeypatch.setenv("HARDWARE_HEURISTICS__GPU_WORKER_BONUS", "4")
    get_settings.cache_clear()
    try:
        profile = HardwareProfile.detect()
        assert profile.heuristics.base_power_draw == 55.5
        assert profile.heuristics.gpu_worker_bonus == 4
    finally:
        get_settings.cache_clear()


def test_custom_heuristics_affect_estimations() -> None:
    heuristics = HardwareHeuristics(
        base_power_draw=10.0,
        power_per_core=1.0,
        power_per_gpu=50.0,
        minimum_power_draw=5.0,
        low_memory_power_threshold_gb=32.0,
        low_memory_power_scale=0.5,
        cpu_capacity_divisor=1,
        gpu_worker_bonus=5,
        worker_low_memory_soft_limit_gb=32.0,
        worker_memory_floor_gb=16.0,
        worker_low_memory_increment=3,
        worker_default_increment=6,
        warm_pool_memory_threshold_gb=1.0,
        warm_pool_divisor=1,
        warm_pool_cap=10,
        warm_pool_floor=2,
    )
    profile = HardwareProfile(
        physical_cores=2,
        logical_cpus=4,
        total_memory_gb=16.0,
        gpu_count=1,
        heuristics=heuristics,
    )

    assert profile.estimate_training_power_draw() == pytest.approx(31.0)
    assert profile.max_recommended_workers(configured_default=1) == 8
    assert profile.min_recommended_workers() == 2
