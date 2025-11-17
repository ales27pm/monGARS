import logging
import time

import ray
from ray import serve

from monGARS.core.llm_integration import LLMIntegration
from monGARS.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


@serve.deployment(
    ray_actor_options={
        "num_gpus": 0.5 if settings.llm.use_gpu else 0,
        "memory": 8 * 1024 * 1024 * 1024,  # 8GB
    },
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_num_ongoing_requests_per_replica": 5,
    },
)
class RayLLMDeployment:
    def __init__(self):
        self.llm = LLMIntegration.instance()
        self.health_check_timestamp = time.time()

    async def health_check(self):
        """Verify model is responsive"""
        if time.time() - self.health_check_timestamp > 300:  # 5 minutes
            test_response = await self.__call__(
                {"prompt": "health check", "max_new_tokens": 10}
            )
            self.health_check_timestamp = time.time()
        return {"status": "healthy", "last_check": self.health_check_timestamp}

    async def __call__(self, request_data: dict):
        prompt = request_data.get("prompt", "")
        max_tokens = request_data.get("max_new_tokens", settings.llm.max_tokens)

        if not prompt:
            return {"error": "empty_prompt", "message": "Prompt cannot be empty"}

        try:
            response = self.llm.generate(prompt, max_new_tokens=max_tokens)
            return {
                "response": response,
                "tokens_used": len(self.llm.tokenizer.tokenize(prompt + response)),
                "model": settings.llm.model_name,
            }
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return {"error": "generation_failed", "message": str(e)}


deployment = RayLLMDeployment.bind()
