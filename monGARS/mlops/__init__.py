"""Operational tooling for background training workflows and artefacts."""

from .artifacts import (
    WrapperConfig,
    render_output_bundle_readme,
    render_project_wrapper,
    render_wrapper_readme,
    write_wrapper_bundle,
)
from .dataset import prepare_instruction_dataset
from .exporters import export_gguf, merge_lora_adapters, run_generation_smoke_test
from .model import (
    detach_sequences,
    load_4bit_causal_lm,
    move_to_cpu,
    summarise_device_map,
)
from .training import (
    LoraHyperParams,
    TrainerConfig,
    disable_training_mode,
    prepare_lora_model,
    prepare_lora_model_light,
    run_embedding_smoke_test,
    save_lora_artifacts,
    train_qlora,
)
from .training_pipeline import training_workflow
from .utils import (
    configure_cuda_allocator,
    describe_environment,
    ensure_dependencies,
)
from .wrapper_loader import WrapperBundle, WrapperBundleError, load_wrapper_bundle

__all__ = [
    "WrapperConfig",
    "render_output_bundle_readme",
    "render_project_wrapper",
    "render_wrapper_readme",
    "training_workflow",
    "write_wrapper_bundle",
    "prepare_instruction_dataset",
    "export_gguf",
    "merge_lora_adapters",
    "run_generation_smoke_test",
    "load_4bit_causal_lm",
    "summarise_device_map",
    "move_to_cpu",
    "detach_sequences",
    "LoraHyperParams",
    "TrainerConfig",
    "prepare_lora_model_light",
    "prepare_lora_model",
    "train_qlora",
    "disable_training_mode",
    "save_lora_artifacts",
    "run_embedding_smoke_test",
    "configure_cuda_allocator",
    "ensure_dependencies",
    "describe_environment",
    "WrapperBundle",
    "WrapperBundleError",
    "load_wrapper_bundle",
]
