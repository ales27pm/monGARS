"""Unit tests for the pgvector restoration migration helpers."""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path
from typing import Any

import pytest

alembic_module = sys.modules.get("alembic")
if alembic_module is None or not hasattr(alembic_module, "op"):
    stub = types.ModuleType("alembic")
    stub.__path__ = []  # type: ignore[attr-defined]
    op_stub = types.ModuleType("alembic.op")
    stub.op = op_stub
    sys.modules["alembic"] = stub
    sys.modules["alembic.op"] = op_stub

spec = importlib.util.spec_from_file_location(
    "restore_pgvector_migration",
    Path(__file__).resolve().parents[1]
    / "alembic"
    / "versions"
    / "20250308_01_restore_pgvector.py",
)
assert spec is not None and spec.loader is not None
migration = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = migration
spec.loader.exec_module(migration)


@pytest.mark.parametrize(
    "payload,dimensions,expected",
    [
        ("[1, 2, 3]", 4, [1.0, 2.0, 3.0, 0.0]),
        ([0.1, 0.2, 0.3, 0.4, 0.5], 3, [0.1, 0.2, 0.3]),
        (("1", "2"), 2, [1.0, 2.0]),
        ([-1, -2, -3], 3, [-1.0, -2.0, -3.0]),
        ([float("nan"), 1], 2, [float("nan"), 1.0]),
        ([float("inf"), float("-inf")], 2, [float("inf"), float("-inf")]),
        (
            [float("nan"), float("inf"), -5],
            4,
            [float("nan"), float("inf"), -5.0, 0.0],
        ),
    ],
)
def test_normalise_vector_success_cases(
    payload: Any, dimensions: int, expected: list[float]
) -> None:
    result = migration._normalise_vector(payload, dimensions=dimensions)
    assert result is not None
    assert len(result) == len(expected)
    for actual_value, expected_value in zip(result, expected):
        if math.isnan(expected_value):
            assert math.isnan(actual_value)
        elif math.isinf(expected_value):
            assert actual_value == expected_value
        else:
            assert actual_value == pytest.approx(expected_value)


@pytest.mark.parametrize(
    "payload",
    [None, "", "not-json", "{}", object()],
)
def test_normalise_vector_invalid_payloads(payload: Any) -> None:
    result = migration._normalise_vector(payload, dimensions=4)
    assert result is None


def test_normalise_vector_rejects_non_numeric_values() -> None:
    with pytest.raises(TypeError):
        migration._normalise_vector(["a", "b"], dimensions=2)
    with pytest.raises(TypeError):
        migration._normalise_vector([1, "a", 2], dimensions=3)


def test_normalise_vector_handles_objects_with_tolist() -> None:
    class ArrayLike:
        def __init__(self, values: list[float]):
            self._values = values

        def tolist(self) -> list[float]:
            return self._values

    payload = ArrayLike([1.0, 2.0, 3.0])
    result = migration._normalise_vector(payload, dimensions=5)
    assert result is not None
    assert len(result) == 5
    assert result[:3] == pytest.approx([1.0, 2.0, 3.0])
    assert all(math.isclose(value, 0.0) for value in result[3:])
