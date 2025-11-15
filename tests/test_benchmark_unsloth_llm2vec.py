from __future__ import annotations

from pathlib import Path

from scripts import benchmark_unsloth_llm2vec as benchmark


def _snapshot_with_gpu(
    total_memory_mb: float, *, ram_gb: float = 32.0
) -> benchmark.HardwareSnapshot:
    return benchmark.HardwareSnapshot(
        cpu_model="Test CPU",
        physical_cores=8,
        logical_cores=16,
        memory_total_gb=ram_gb,
        memory_available_gb=ram_gb - 4,
        gpus=[
            benchmark.GPUStatus(
                name="Test GPU",
                total_memory_mb=total_memory_mb,
                free_memory_mb=total_memory_mb - 512,
                temperature_c=60.0,
                utilisation_percent=75.0,
            )
        ],
        cuda_available=True,
        cuda_device_count=1,
        torch_version="2.2.0",
        cuda_version="12.1",
    )


def test_auto_tune_hyperparameters_scales_to_gpu_capacity() -> None:
    snapshot = _snapshot_with_gpu(6144.0)
    params = {
        "vram_budget_mb": 9000,
        "batch_size": 2,
        "grad_accum": 4,
        "activation_buffer_mb": 2048,
    }

    tuned, adjustments = benchmark.auto_tune_hyperparameters(snapshot, params)

    assert tuned["vram_budget_mb"] <= 5632
    assert tuned["batch_size"] == 1
    assert tuned["grad_accum"] == 8
    assert tuned["activation_buffer_mb"] <= 2048
    assert any("vram_budget_mb" in entry for entry in adjustments)
    assert any("batch_size" in entry for entry in adjustments)


def test_benchmark_report_serialisation(tmp_path: Path) -> None:
    snapshot = _snapshot_with_gpu(12288.0)
    runs = [
        benchmark.RunMetrics(
            run_index=1,
            duration_seconds=10.0,
            cpu_user_seconds=5.0,
            cpu_system_seconds=2.0,
            rss_start_mb=100.0,
            rss_end_mb=150.0,
            max_cuda_reserved_mb=512.0,
            max_cuda_allocated_mb=480.0,
            dataset_size=100,
            eval_dataset_size=20,
            evaluation_metrics={"loss": 0.1},
            result_paths={"output_dir": tmp_path},
        ),
        benchmark.RunMetrics(
            run_index=2,
            duration_seconds=8.0,
            cpu_user_seconds=4.0,
            cpu_system_seconds=1.5,
            rss_start_mb=110.0,
            rss_end_mb=140.0,
            max_cuda_reserved_mb=600.0,
            max_cuda_allocated_mb=500.0,
            dataset_size=80,
            eval_dataset_size=16,
            evaluation_metrics={"loss": 0.09},
            result_paths={"output_dir": tmp_path / "run2"},
        ),
    ]
    report = benchmark.BenchmarkReport(
        hardware=snapshot,
        requested_args={"batch_size": 2},
        tuned_args={"batch_size": 1},
        adjustments=["batch size reduced"],
        runs=runs,
    )

    data = report.to_dict()

    assert data["summary"]["run_count"] == 2
    assert data["summary"]["duration_seconds"]["max"] == 10.0
    assert data["summary"]["examples_per_second"]["mean"] == 10.0
    assert data["runs"][0]["result_paths"]["output_dir"] == str(tmp_path)
    assert data["hardware"]["gpus"][0]["name"] == "Test GPU"
