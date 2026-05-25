# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
HSTU Model for vLLM-GR.

Unified HSTU model class that:
1. Is compatible with vLLM's ModelRegistry
2. Delegates embedding lookup and preprocess to DlrmHSTU (via EmbeddingCollection)
3. Accepts KeyedJaggedTensor inputs from GRModelRunner

HSTU is a recommendation model (not LLM):
- Uses Target-Aware Attention instead of autoregressive generation
- Processes user history + candidates in a single forward pass
- Returns ranking scores instead of token probabilities

Runtime topology:
- Sparse (embedding) weights stay on the embedding device
- Dense (MLP/Attention) weights run on the active accelerator
- Forward pass: preprocess(embedding device) -> transfer -> main_forward(dense device)
"""

from functools import partial
import importlib
import inspect
import types
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from vllm.logger import init_logger

try:
    import vllm as _vllm_pkg

    if not hasattr(_vllm_pkg, "forward_context"):
        try:
            _vllm_pkg.forward_context = importlib.import_module("vllm.forward_context")
        except Exception:
            _vllm_pkg.forward_context = types.SimpleNamespace(
                is_forward_context_available=lambda: False,
                get_forward_context=lambda: None,
            )
except Exception:
    pass

from vllm_gr.utils import (
    move_hstu_sparse_inputs_to_device,
    move_hstu_sparse_outputs_to_device,
    resolve_hstu_runtime_devices,
)

logger = init_logger(__name__)


def _get_callable_keyword_names(callable_obj: Any) -> tuple[set[str], bool]:
    """Return accepted keyword names and whether ``**kwargs`` is supported."""
    signature = inspect.signature(callable_obj)
    keyword_names: set[str] = set()
    accepts_var_kwargs = False

    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_var_kwargs = True
        elif parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            keyword_names.add(parameter.name)

    return keyword_names, accepts_var_kwargs


class HSTUPooler(nn.Module):
    """
    Pooler module for HSTU model — provides vLLM pooling model compatibility.

    For recommendation models, the "pooling" operation is essentially
    returning the item scores/predictions rather than text embeddings.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.dense = nn.Identity()

    def get_supported_tasks(self) -> set:
        """Valid PoolingTask values: "embed", "classify", "score", etc."""
        return {"embed"}

    def get_pooling_updates(self, task: str) -> Any:
        from vllm.model_executor.layers.pooler.common import PoolingParamsUpdate
        return PoolingParamsUpdate()

    def forward(
        self,
        hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        pooling_metadata: Optional[Any] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(hidden_states, list):
            return [self.dense(h) for h in hidden_states]
        return self.dense(hidden_states)


class HSTUForRecommendation(nn.Module):
    """
    Unified HSTU model for recommendation.

    vLLM-compatible wrapper around DlrmHSTU.  DlrmHSTU owns the
    EmbeddingCollection internally and handles preprocess + forward.
    This class only:
      - translates HSTUConfig → DlrmHSTUConfig + EmbeddingConfig
      - provides the vLLM forward / load_weights interface
    """

    supports_multimodal = False
    supports_pp = False
    is_pooling_model = True
    is_text_generation_model = False
    is_attention_free = False

    def __init__(
        self,
        *,
        vllm_config: Any = None,
        prefix: str = "",
        config: Any = None,
        cache_config: Any = None,
        quant_config: Any = None,
        lora_config: Any = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__()
        self.prefix = prefix
        self.vllm_config = vllm_config

        if vllm_config is not None:
            config = vllm_config.model_config.hf_config

        self.config = config
        self.enable_kv_cache = bool(getattr(self.config, "enable_kv_cache", False))
        self.kv_cache_dtype = (
            getattr(self.config, "kv_cache_dtype", None)
            or getattr(self.config, "torch_dtype", torch.bfloat16)
        )
        self._hstu_kv_cache_layers = nn.ModuleList()
        dense_device_override = (
            torch.device(device) if device is not None else None
        )
        self._set_runtime_devices(
            dense_device_override=dense_device_override,
        )

        self._create_hstu_model()
        self._init_hstu_kv_cache_layers()

        self._pooler = HSTUPooler(self.config.hstu_transducer_embedding_dim)

        logger.info(
            "HSTUForRecommendation initialized: layers=%d, heads=%d, dim=%d, "
            "dense_device=%s, embedding_device=%s, enable_kv_cache=%s",
            self.config.hstu_attn_num_layers,
            self.config.hstu_num_heads,
            self.config.hstu_transducer_embedding_dim,
            self._dense_device,
            self._embedding_device,
            self.enable_kv_cache,
        )

    @property
    def pooler(self):
        return self._pooler

    @staticmethod
    def _split_pooling_predictions(
        predictions: torch.Tensor,
        num_candidates: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Return one pooling output per request for vLLM's scheduler contract.

        HSTU predicts over the concatenated candidate axis for the whole batch.
        vLLM pooling expects an iterable whose length matches the scheduled
        request count, so we split the last dimension back into per-request
        tensors using the candidate counts produced during preprocess().
        """
        if num_candidates.numel() == 0:
            return []

        split_sizes = [int(size) for size in num_candidates.reshape(-1).tolist()]
        total_candidates = sum(split_sizes)

        if predictions.shape[-1] != total_candidates:
            raise ValueError(
                "HSTU pooling split mismatch: "
                f"predictions.shape={tuple(predictions.shape)}, "
                f"sum(num_candidates)={total_candidates}"
            )

        return list(torch.split(predictions, split_sizes, dim=-1))

    def _build_embedding_configs(self) -> Dict[str, "EmbeddingConfig"]:
        """Convert HSTUConfig.embedding_tables (list of dicts) to TorchRec EmbeddingConfig dict."""
        from torchrec.modules.embedding_configs import EmbeddingConfig, DataType

        def _resolve_data_type(raw_value: Any) -> DataType:
            if isinstance(raw_value, DataType):
                return raw_value
            normalized = str(raw_value).upper()
            bf16_dtype = getattr(DataType, "BF16", None)
            aliases = {
                "FP16": DataType.FP16,
                "FLOAT16": DataType.FP16,
                "HALF": DataType.FP16,
                "FP32": DataType.FP32,
                "FLOAT32": DataType.FP32,
            }
            if bf16_dtype is not None:
                aliases["BF16"] = bf16_dtype
                aliases["BFLOAT16"] = bf16_dtype
            elif normalized in {"BF16", "BFLOAT16"}:
                raise ValueError(
                    "embedding table data_type 'bf16' is not supported by the "
                    "current torchrec DataType runtime"
                )
            try:
                return aliases[normalized]
            except KeyError as exc:
                raise ValueError(f"Unsupported embedding table data_type: {raw_value!r}") from exc

        configs: Dict[str, EmbeddingConfig] = {}
        default_data_type = getattr(
            self.config,
            "embedding_table_data_type",
            "FP16",
        )
        accepted_kwargs, accepts_var_kwargs = _get_callable_keyword_names(
            EmbeddingConfig,
        )
        for table_cfg in self.config.embedding_tables:
            name = table_cfg["name"]
            embedding_kwargs: Dict[str, Any] = {
                "num_embeddings": table_cfg["num_embeddings"],
                "embedding_dim": table_cfg["embedding_dim"],
                "name": name,
                "data_type": _resolve_data_type(
                    table_cfg.get("data_type", default_data_type)
                ),
                "feature_names": table_cfg.get("feature_names", [name]),
            }

            for key, value in table_cfg.items():
                if key in {
                    "name",
                    "num_embeddings",
                    "embedding_dim",
                    "data_type",
                    "feature_names",
                }:
                    continue
                if accepts_var_kwargs or key in accepted_kwargs:
                    embedding_kwargs[key] = value

            configs[name] = EmbeddingConfig(
                **embedding_kwargs,
            )
        return configs

    def _get_hstu_attention_metadata(
        self,
        uih_seq_lengths: Optional[torch.Tensor] = None,
        num_targets: Optional[torch.Tensor] = None,
    ) -> Optional[Any]:
        if not self.enable_kv_cache or len(self._hstu_kv_cache_layers) == 0:
            return None

        try:
            from vllm.forward_context import (
                get_forward_context,
                is_forward_context_available,
            )
        except ImportError:
            return None

        if not is_forward_context_available():
            return None

        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, list):
            if not attn_metadata:
                return None
            attn_metadata = attn_metadata[0]
        if not isinstance(attn_metadata, dict):
            return None

        for cache_layer in self._hstu_kv_cache_layers:
            layer_name = getattr(cache_layer, "layer_name", None)
            if layer_name is None:
                continue
            metadata = attn_metadata.get(layer_name)
            if metadata is not None:
                runtime_uih_seq_lengths = (
                    uih_seq_lengths.to(
                        device=metadata.history_lengths.device,
                        dtype=metadata.history_lengths.dtype,
                    )
                    if uih_seq_lengths is not None
                    else None
                )
                runtime_num_targets = (
                    num_targets.to(
                        device=metadata.num_targets.device,
                        dtype=metadata.num_targets.dtype,
                    )
                    if num_targets is not None
                    else None
                )
                if (
                    runtime_uih_seq_lengths is not None
                    and int(metadata.history_lengths.to(dtype=torch.int64).sum().item()) == 0
                    and int(runtime_uih_seq_lengths.to(dtype=torch.int64).sum().item()) > 0
                ):
                    metadata.history_lengths = runtime_uih_seq_lengths
                if (
                    runtime_num_targets is not None
                    and int(metadata.num_targets.to(dtype=torch.int64).sum().item()) == 0
                    and int(runtime_num_targets.to(dtype=torch.int64).sum().item()) > 0
                ):
                    metadata.num_targets = runtime_num_targets
                if (
                    int(getattr(metadata, "contextual_seq_len", 0)) == 0
                    and getattr(self.config, "contextual_feature_to_max_length", None)
                ):
                    metadata.contextual_seq_len = int(
                        sum(self.config.contextual_feature_to_max_length.values())
                    )

                total_delta_tokens = int(
                    metadata.num_targets.to(dtype=torch.int64).sum().item()
                )
                expected_total_tokens = int(
                    (
                        metadata.history_lengths.to(dtype=torch.int64)
                        + int(getattr(metadata, "contextual_seq_len", 0))
                        + metadata.num_targets.to(dtype=torch.int64)
                    ).sum().item()
                )
                slot_mapping = getattr(
                    getattr(metadata, "common_attn_metadata", None),
                    "slot_mapping",
                    None,
                )
                num_computed_tokens = getattr(
                    getattr(metadata, "common_attn_metadata", None),
                    "_num_computed_tokens_cpu",
                    None,
                )
                actual_token_slots = int(slot_mapping.numel()) if slot_mapping is not None else 0
                actual_computed_tokens = (
                    int(num_computed_tokens.to(dtype=torch.int64).sum().item())
                    if num_computed_tokens is not None
                    else 0
                )
                if actual_token_slots < expected_total_tokens:
                    if (
                        actual_computed_tokens == 0
                        and actual_token_slots == total_delta_tokens
                    ):
                        logger.debug_once(
                            "Skip native HSTU KV cache for this cold-start step "
                            "because scheduler-visible token slots only cover "
                            "the newly scheduled delta tokens: slots=%d "
                            "computed=%d delta=%d expected=%d",
                            actual_token_slots,
                            actual_computed_tokens,
                            total_delta_tokens,
                            expected_total_tokens,
                        )
                    elif (
                        actual_token_slots + actual_computed_tokens
                        >= expected_total_tokens
                    ):
                        logger.debug_once(
                            "Skip native HSTU KV cache for this step because "
                            "scheduler-visible token slots only cover the new "
                            "delta while previously computed/prefix-cached "
                            "tokens already cover the rest of the logical HSTU "
                            "sequence: slots=%d computed=%d expected=%d",
                            actual_token_slots,
                            actual_computed_tokens,
                            expected_total_tokens,
                        )
                    else:
                        logger.warning_once(
                            "Disable native HSTU KV cache for this step because "
                            "scheduler-visible token slots and previously "
                            "computed tokens still do not cover the full HSTU "
                            "sequence: slots=%d computed=%d expected=%d",
                            actual_token_slots,
                            actual_computed_tokens,
                            expected_total_tokens,
                        )
                    return None
                metadata.cache_layers = list(self._hstu_kv_cache_layers)
                return metadata
        return None

    def _init_hstu_kv_cache_layers(self) -> None:
        if not self.enable_kv_cache or self.vllm_config is None:
            return

        from vllm_gr.model_executor.layers.hstu_attention_layer import (
            HSTUKVCacheAttentionLayer,
        )

        layer_prefix = (
            f"{self.prefix}.hstu_kv_cache_layers"
            if self.prefix
            else "hstu_kv_cache_layers"
        )
        self._hstu_kv_cache_layers = nn.ModuleList(
            [
                HSTUKVCacheAttentionLayer(
                    layer_name=f"{layer_prefix}.{idx}",
                    num_heads=self.config.hstu_num_heads,
                    head_size=self.config.hstu_attn_qk_dim,
                    head_size_v=self.config.hstu_attn_linear_dim,
                    dtype=self.kv_cache_dtype,
                    vllm_config=self.vllm_config,
                    enable_kv_cache=self.enable_kv_cache,
                )
                for idx in range(self.config.hstu_attn_num_layers)
            ]
        )

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------

    def _create_hstu_model(self) -> None:
        """Create the underlying DlrmHSTU model with EmbeddingCollection."""
        from vllm_gr.models.dlrm_v3.modules.dlrm_hstu import (
            DlrmHSTU, DlrmHSTUConfig,
        )
        from vllm_gr.models.dlrm_v3.modules.multitask_module import (
            TaskConfig,
        )

        task_names = self.config.task_names
        task_types = self.config.task_types
        multitask_configs = []

        for i, (task_name, task_type) in enumerate(zip(task_names, task_types)):
            task_config = TaskConfig(
                task_name=task_name,
                task_weight=1 << i,
                task_type=task_type,
            )
            multitask_configs.append(task_config)

        merge_mapping = [tuple(pair) for pair in self.config.merge_uih_candidate_feature_mapping]

        hstu_config = DlrmHSTUConfig(
            max_seq_len=self.config.max_seq_len,
            max_num_candidates=self.config.max_num_candidates,
            max_num_candidates_inference=self.config.max_num_candidates_inference,
            hstu_num_heads=self.config.hstu_num_heads,
            hstu_attn_linear_dim=self.config.hstu_attn_linear_dim,
            hstu_attn_qk_dim=self.config.hstu_attn_qk_dim,
            hstu_attn_num_layers=self.config.hstu_attn_num_layers,
            hstu_embedding_table_dim=self.config.hstu_embedding_table_dim,
            hstu_preprocessor_hidden_dim=self.config.hstu_preprocessor_hidden_dim,
            hstu_transducer_embedding_dim=self.config.hstu_transducer_embedding_dim,
            hstu_group_norm=self.config.hstu_group_norm,
            hstu_input_dropout_ratio=self.config.hstu_input_dropout_ratio,
            hstu_linear_dropout_rate=self.config.hstu_linear_dropout_rate,
            contextual_feature_to_max_length=self.config.contextual_feature_to_max_length,
            contextual_feature_to_min_uih_length=self.config.contextual_feature_to_min_uih_length,
            candidates_weight_feature_name=self.config.candidates_weight_feature_name,
            candidates_watchtime_feature_name=self.config.candidates_watchtime_feature_name,
            candidates_querytime_feature_name=self.config.candidates_querytime_feature_name,
            multitask_configs=multitask_configs,
            user_embedding_feature_names=self.config.user_embedding_feature_names,
            item_embedding_feature_names=self.config.item_embedding_feature_names,
            uih_post_id_feature_name=self.config.uih_post_id_feature_name,
            uih_action_time_feature_name=self.config.uih_action_time_feature_name,
            uih_weight_feature_name=self.config.uih_weight_feature_name,
            hstu_uih_feature_names=self.config.hstu_uih_feature_names,
            hstu_candidate_feature_names=self.config.hstu_candidate_feature_names,
            merge_uih_candidate_feature_mapping=merge_mapping,
            action_weights=self.config.action_weights or None,
            action_embedding_init_std=getattr(self.config, "action_embedding_init_std", 0.1),
            causal_multitask_weights=getattr(self.config, "causal_multitask_weights", 0.2),
            enable_postprocessor=getattr(self.config, "enable_postprocessor", True),
            use_layer_norm_postprocessor=getattr(self.config, "use_layer_norm_postprocessor", False),
            enable_kv_cache=getattr(self.config, "enable_kv_cache", False),
        )

        embedding_configs = self._build_embedding_configs()

        self.hstu = DlrmHSTU(
            hstu_configs=hstu_config,
            embedding_tables=embedding_configs,
            is_dense=not embedding_configs,
            embedding_device=self._embedding_device,
        )
        self._patch_embedding_collection_for_inference()

    def _patch_embedding_collection_for_inference(self) -> None:
        """Align inference embedding lookup behavior with the golden sparse path."""

        embedding_collection = getattr(self.hstu, "_embedding_collection", None)
        if embedding_collection is None:
            return

        from vllm_gr.model_executor.models.embedding_collection_patch import (
            ec_patched_forward_wo_embedding_copy,
        )

        embedding_collection.forward = partial(
            ec_patched_forward_wo_embedding_copy,
            embedding_collection,
        )

    def _get_dense_forward_dtype(self) -> torch.dtype:
        """Return the dense-phase dtype expected by downstream HSTU modules."""
        item_mlp = getattr(self.hstu, "_item_embedding_mlp", None)
        if item_mlp is not None and len(item_mlp) > 0:
            first_layer = item_mlp[0]
            weight = getattr(first_layer, "weight", None)
            if weight is not None:
                return weight.dtype

        for name, param in self.hstu.named_parameters():
            if "embedding_collection" not in name:
                return param.dtype

        return torch.float32

    # ------------------------------------------------------------------
    # vLLM interface
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[Any] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        uih_features: Optional[Any] = None,
        candidates_features: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Two-phase forward pass with sparse/dense device separation.

        Phase 1 (embedding device): preprocess() — embedding lookups via
                       EmbeddingCollection on the sparse runtime device
                       (CPU by default). Produces seq_embeddings +
                       payload_features.
        Transfer:      move_sparse_output_to_device() — embedding device ->
                       dense device.
        Phase 2 (dense device): main_forward() — attention, MLPs,
                       multitask prediction on the active accelerator.

        When uih_features/candidates_features are None (vLLM warmup),
        returns a zero tensor on the dense execution device.
        """
        dense_device = self._get_device()
        if uih_features is None or candidates_features is None:
            total_tokens = input_ids.numel() if input_ids is not None else 1
            return torch.zeros(
                total_tokens, self.config.hstu_transducer_embedding_dim,
                device=dense_device,
            )

        # Phase 1: explicitly re-home sparse inputs onto the embedding device.
        uih_features, candidates_features = move_hstu_sparse_inputs_to_device(
            uih_features=uih_features,
            candidates_features=candidates_features,
            device=self._embedding_device,
        )

        # Sparse preprocess on the embedding device.
        (
            seq_embeddings,
            payload_features,
            max_uih_len,
            uih_seq_lengths,
            max_num_candidates,
            num_candidates,
        ) = self.hstu.preprocess(
            uih_features=uih_features,
            candidates_features=candidates_features,
        )

        # Transfer: embedding device -> dense execution device.
        seq_embeddings, payload_features, uih_seq_lengths, num_candidates = (
            move_hstu_sparse_outputs_to_device(
                seq_embeddings=seq_embeddings,
                payload_features=payload_features,
                uih_seq_lengths=uih_seq_lengths,
                num_candidates=num_candidates,
                device=dense_device,
                source_device=self._embedding_device,
                target_dtype=self._get_dense_forward_dtype(),
            )
        )

        # Phase 2: dense forward on the active accelerator.
        hstu_attn_metadata = self._get_hstu_attention_metadata()
        predictions = self.hstu.main_forward(
            seq_embeddings=seq_embeddings,
            payload_features=payload_features,
            max_uih_len=max_uih_len,
            uih_seq_lengths=uih_seq_lengths,
            max_num_candidates=max_num_candidates,
            num_candidates=num_candidates,
            hstu_attn_metadata=hstu_attn_metadata,
        )

        return self._split_pooling_predictions(predictions, num_candidates)

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> Set[str]:
        """Load model weights with sparse/dense device separation.

        - Sparse weights (embedding_collection) stay on the embedding device
        - Dense weights go to the active accelerator
        - Registered buffers are loaded manually (AutoWeightsLoader
          only handles nn.Module and nn.Parameter)

        Device placement is automatic: parameters were created on their
        target device (embedding device for sparse weights, dense device
        for everything else via vLLM's default device context).
        param.data.copy_() copies to wherever the parameter already lives.
        """
        from vllm.model_executor.models.utils import AutoWeightsLoader

        buffer_names = {name for name, _ in self.named_buffers()}

        buffer_weights: List[Tuple[str, torch.Tensor]] = []
        param_weights: List[Tuple[str, torch.Tensor]] = []
        sparse_count = 0
        dense_count = 0

        for name, tensor in weights:
            is_sparse = "embedding_collection" in name
            if name in buffer_names:
                buffer_weights.append((name, tensor))
            else:
                param_weights.append((name, tensor))

            if is_sparse:
                sparse_count += 1
            else:
                dense_count += 1

        loaded: Set[str] = set()

        if buffer_weights:
            buf_dict = dict(self.named_buffers())
            for name, tensor in buffer_weights:
                buf_dict[name].copy_(tensor)
                loaded.add(name)

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["_pooler."],
        )
        loaded |= loader.load_weights(iter(param_weights))

        logger.info(
            "Loaded %d weights: %d sparse (%s), %d dense (%s), "
            "%d buffers",
            len(loaded),
            sparse_count,
            getattr(self.hstu, "_embedding_device", torch.device("cpu")).type,
            dense_count,
            self._get_device().type,
            len(buffer_weights),
        )
        return loaded

    @classmethod
    def from_model_config(
        cls,
        config: Any,
        device: torch.device,
        **kwargs,
    ) -> "HSTUForRecommendation":
        return cls(config=config, device=device, **kwargs)

    def _get_device(self) -> torch.device:
        """Return the dense execution device for HSTU."""
        return self._dense_device

    def _set_runtime_devices(
        self,
        *,
        dense_device_override: Optional[torch.device] = None,
    ) -> None:
        runtime_devices = resolve_hstu_runtime_devices(
            explicit_dense_device=dense_device_override,
        )
        self._dense_device = runtime_devices.dense_device
        self._embedding_device = runtime_devices.embedding_device
