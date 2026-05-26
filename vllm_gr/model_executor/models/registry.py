# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
GR Model Registry.

Registers GR-specific model architectures with vLLM's global ModelRegistry
using the stable public API (ModelRegistry.register_model).

Called during activation when the PatchPlan has ``needs_model_registry=True``
(i.e. ``gr_recommend`` family).  In the main process this is triggered by
``GREngineArgs.create_engine_config()``.

Usage:
    from vllm_gr.runtime.patch_plan import build_patch_plan
    from vllm_gr.runtime.activation import activate

    plan = build_patch_plan(selection)
    activate(plan)

    # Or manually:
    from vllm_gr.model_executor.models.registry import (
        register_gr_model_architectures,
    )
    register_gr_model_architectures()
"""

import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# vLLM Registry Import (public API only)
# ============================================================================

try:
    from vllm.model_executor.models.registry import (
        ModelRegistry as VLLMModelRegistry,
    )
    _HAS_VLLM_REGISTRY = True
except ImportError:
    _HAS_VLLM_REGISTRY = False
    VLLMModelRegistry = None

# ============================================================================
# GR Model Definitions
# ============================================================================

# architecture -> (module_relative_name, class_name)
# Module path is relative to vllm_gr.model_executor.models
#
# "HSTUEmbeddingModel" is the canonical HuggingFace architecture name
# declared in HSTUConfig.architectures and every config.json.  vLLM's
# ModelRegistry resolves the model class by this name.  The "Embedding"
# suffix follows the vLLM convention for pooling-runner models.
_GR_MODEL_ARCHITECTURES: Dict[str, Tuple[str, str]] = {
    "HSTUEmbeddingModel": ("hstu_vllm", "HSTUForRecommendation"),
}

# ============================================================================
# Registration State
# ============================================================================

_GR_MODEL_ARCHITECTURES_REGISTERED = False


# ============================================================================
# Public API
# ============================================================================

def register_gr_model_architectures() -> bool:
    """
    Register GR model architectures into vLLM's global ModelRegistry.

    Uses the stable ``ModelRegistry.register_model(arch, "module:class")``
    API which supports lazy loading and avoids premature CUDA init.

    Returns:
        True if registration succeeded or already registered
    """
    global _GR_MODEL_ARCHITECTURES_REGISTERED

    if _GR_MODEL_ARCHITECTURES_REGISTERED:
        return True

    if not _HAS_VLLM_REGISTRY:
        logger.warning("vLLM registry not available")
        return False

    try:
        supported_archs = VLLMModelRegistry.get_supported_archs()

        for arch, (mod_name, cls_name) in _GR_MODEL_ARCHITECTURES.items():
            if arch not in supported_archs:
                VLLMModelRegistry.register_model(
                    arch,
                    f"vllm_gr.model_executor.models.{mod_name}:{cls_name}"
                )
                logger.debug(f"Registered {arch} with vLLM ModelRegistry")

        _GR_MODEL_ARCHITECTURES_REGISTERED = True
        logger.info("GR model architectures registered with vLLM ModelRegistry")
        return True

    except Exception as e:
        logger.error(f"Failed to register GR model architectures: {e}")
        return False


def register_gr_models_to_vllm() -> bool:
    """Backward-compatible alias for the old registry entrypoint."""
    return register_gr_model_architectures()


def is_registered() -> bool:
    """Check if GR models are registered with vLLM."""
    return _GR_MODEL_ARCHITECTURES_REGISTERED


def get_gr_architectures() -> list:
    """Get list of supported GR architectures."""
    return list(_GR_MODEL_ARCHITECTURES.keys())


def is_gr_model(arch: str) -> bool:
    """Check if an architecture is a GR model."""
    return arch in _GR_MODEL_ARCHITECTURES
