import asyncio
import logging

import uvicorn

from init_db import init_db
from monGARS.config import get_settings
from monGARS.core.monitor import SystemMonitor
from monGARS.core.orchestrator import Orchestrator
from monGARS.core.self_training import SelfTrainingEngine
from monGARS.mlops.training_pipeline import training_workflow

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
settings = get_settings()


async def main():
    await init_db()
    monitor = SystemMonitor()
    trainer = SelfTrainingEngine()
    orchestrator = Orchestrator()
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(monitor.get_system_stats())
            tg.create_task(trainer.auto_improve())
            tg.create_task(training_workflow())
            tg.create_task(orchestrator.run_loop())
    except Exception as e:
        logging.error(f"Error in task group: {e}")
    uvicorn.run("monGARS.api.web_api:app", host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    asyncio.run(main())
