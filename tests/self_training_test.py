import sys
import types

import pytest

# Provide a minimal EmbeddingSystem to satisfy imports
module = types.ModuleType("monGARS.core.neurones")
module.EmbeddingSystem = object
sys.modules["monGARS.core.neurones"] = module

from monGARS.core.self_training import SelfTrainingEngine


@pytest.mark.asyncio
async def test_run_training_cycle_creates_version():
    engine = SelfTrainingEngine()
    await engine.training_queue.put({"data": 1})
    await engine._run_training_cycle()
    assert "v1" in engine.model_versions
    assert engine.model_versions["v1"]["data_count"] == 1


@pytest.mark.asyncio
async def test_run_training_cycle_no_data():
    engine = SelfTrainingEngine()
    await engine._run_training_cycle()
    assert engine.model_versions == {}
