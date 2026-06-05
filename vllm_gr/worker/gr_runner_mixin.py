# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared GR-specific model runner logic.

Delegates feature decoding and model-kwargs building to the injected adapter.
HSTU model execution infrastructure (static forward context sync, KV cache
spec fallback, attention builder preparation) remains here as it is tied to
the HSTU model's execution lifecycle.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import ModelRunnerOutput

from vllm_gr.engine.gr_features_codec import decode_gr_features_payload
from vllm_gr.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class GRFeatureRunnerMixin:
    """Inject GR feature decode and model-kwargs building into a runner.

    The ``_adapter`` attribute must be set before ``_update_states`` is called.
    """

    def __init__(self, *args, adapter=None, **kwargs):
        # adapter is consumed here; not passed to super()
        super().__init__(*args, **kwargs)
        self._adapter = adapter
        self._gr_features_cache: dict[str, dict[str, Any]] = {}
        self._gr_feature_errors: dict[str, str] = {}

    def _apply_hstu_runtime_policy(self) -> None:
        gr_config = getattr(getattr(self, "vllm_config", None), "gr_config", None)
        if gr_config is None:
            return

        from vllm_gr.ops.hstu_attention import configure_hstu_npu_fused_dispatch
        from vllm_gr.utils.npu_adapter import is_npu_device

        device = getattr(self, "device", None)
        is_npu_runtime = bool(device is not None and is_npu_device(device))

        fused_prefill = (
            is_npu_runtime
            if gr_config.enable_npu_fused_prefill is None
            else bool(gr_config.enable_npu_fused_prefill)
        )
        force_hammer_npu = (
            is_npu_runtime
            if gr_config.force_hammer_npu is None
            else bool(gr_config.force_hammer_npu)
        )

        configure_hstu_npu_fused_dispatch(
            prefill_enabled=fused_prefill,
        )

        logger.info(
            "Applied HSTU runtime policy: is_npu_runtime=%s prefill=%s force_hammer_npu=%s",
            is_npu_runtime,
            fused_prefill,
            force_hammer_npu,
        )

        if not force_hammer_npu:
            return

        from vllm_gr.models.dlrm_v3.common import HammerKernel, HammerModule

        model = getattr(self, "model", None)
        if model is None:
            return

        changed = 0
        for module in model.modules():
            if isinstance(module, HammerModule):
                module.set_hammer_kernel(HammerKernel.NPU)
                changed += 1

        logger.info(
            "Forced HammerKernel.NPU on %d HammerModule instance(s)",
            changed,
        )

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            gr_features = getattr(new_req_data, "gr_features", None)
            if gr_features is not None:
                try:
                    payload = decode_gr_features_payload(
                        gr_features,
                        self._adapter.payload_type,
                    )
                    decoded = self._adapter.decode_payload(payload)
                    self._gr_features_cache[req_id] = decoded
                    self._gr_feature_errors.pop(req_id, None)
                    logger.debug(
                        "Decoded GR features for request %s: uih_keys=%s, cand_keys=%s",
                        req_id,
                        decoded.get("uih_keys"),
                        decoded.get("cand_keys"),
                    )
                except Exception as e:
                    self._gr_features_cache.pop(req_id, None)
                    self._gr_feature_errors[req_id] = str(e)
                    logger.error(
                        "Error decoding GR features for %s: %s",
                        req_id,
                        e,
                    )
            else:
                logger.warning(
                    "Request %s has no gr_features; monkey-patch may not be active.",
                    req_id,
                )

        for req_id in scheduler_output.finished_req_ids:
            self._gr_features_cache.pop(req_id, None)
            self._gr_feature_errors.pop(req_id, None)

        super()._update_states(scheduler_output)

    def _sync_hstu_static_forward_context(self) -> int:
        compilation_config = getattr(
            getattr(self, "vllm_config", None),
            "compilation_config",
            None,
        )
        static_forward_context = getattr(
            compilation_config,
            "static_forward_context",
            None,
        )
        model = getattr(self, "model", None)
        hstu_kv_layers = getattr(model, "_hstu_kv_cache_layers", None)
        if static_forward_context is None or not hstu_kv_layers:
            return 0

        inserted = 0
        for layer in hstu_kv_layers:
            layer_name = getattr(layer, "layer_name", None)
            if layer_name is None:
                continue
            if static_forward_context.get(layer_name) is layer:
                continue
            static_forward_context[layer_name] = layer
            inserted += 1

        if inserted:
            logger.debug(
                "Registered %d HSTU KV cache layers into static_forward_context.",
                inserted,
            )
        return inserted

    def _build_request_lengths(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        """Build request-length tensors for the current batch via adapter."""
        if not self._gr_features_cache:
            return None

        batch_req_ids = list(getattr(self.input_batch, "req_ids", ()))
        if not batch_req_ids:
            return None

        features_list: list[dict[str, Any]] = []
        for req_id in batch_req_ids:
            features = self._gr_features_cache.get(req_id)
            if features is None:
                return None
            features_list.append(features)

        return self._adapter.build_request_lengths(features_list)

    def _prepare_hstu_attention_builders(self) -> None:
        from vllm_gr.attention.backends.hstu_attn import (
            HSTUAttentionMetadataBuilder,
        )

        hstu_builders: list[HSTUAttentionMetadataBuilder] = []
        for kv_cache_groups in getattr(self, "attn_groups", ()):
            for attn_group in kv_cache_groups:
                for builder in attn_group.metadata_builders:
                    if isinstance(builder, HSTUAttentionMetadataBuilder):
                        hstu_builders.append(builder)

        if not hstu_builders:
            return

        request_lengths = self._build_request_lengths()
        for builder in hstu_builders:
            if request_lengths is None:
                builder.clear_request_lengths()
            else:
                num_targets, history_lengths, contextual_seq_len = request_lengths
                builder.set_request_lengths(
                    num_targets=num_targets,
                    history_lengths=history_lengths,
                    contextual_seq_len=contextual_seq_len,
                )

    def _prepare_inputs(self, *args, **kwargs):
        # Both GPUModelRunner and vllm-ascend NPUModelRunner build attention
        # metadata from inside `_prepare_inputs`, so inject HSTU request-length
        # state here to keep the builder path shared across backends.
        self._prepare_hstu_attention_builders()
        return super()._prepare_inputs(*args, **kwargs)

    def _get_hstu_kv_cache_spec_fallback(self) -> dict[str, Any]:
        self._sync_hstu_static_forward_context()

        model = getattr(self, "model", None)
        hstu_kv_layers = getattr(model, "_hstu_kv_cache_layers", None)
        if not hstu_kv_layers:
            return {}

        kv_cache_spec: dict[str, Any] = {}
        for layer in hstu_kv_layers:
            layer_name = getattr(layer, "layer_name", None)
            if layer_name is None:
                continue
            spec_getter = getattr(layer, "get_kv_cache_spec", None)
            if not callable(spec_getter):
                continue
            spec = spec_getter(self.vllm_config)
            if spec is not None:
                kv_cache_spec[layer_name] = spec
        return kv_cache_spec

    def _preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        num_input_tokens: int,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple:
        result = super()._preprocess(
            scheduler_output,
            num_input_tokens,
            intermediate_tensors,
        )

        (
            input_ids,
            inputs_embeds,
            positions,
            intermediate,
            model_kwargs,
            ec_connector_output,
        ) = result

        gr_kwargs = self._build_gr_model_kwargs()
        if gr_kwargs:
            model_kwargs.update(gr_kwargs)

        return (
            input_ids,
            inputs_embeds,
            positions,
            intermediate,
            model_kwargs,
            ec_connector_output,
        )

    def _build_gr_model_kwargs(self) -> dict[str, Any]:
        batch_req_ids = list(self.input_batch.req_ids)
        failed_req_ids = [req_id for req_id in batch_req_ids if req_id in self._gr_feature_errors]
        if failed_req_ids:
            error_text = ", ".join(
                f"{req_id}: {self._gr_feature_errors[req_id]}" for req_id in failed_req_ids
            )
            raise ValueError(f"GR feature decode failed for scheduled requests: {error_text}")

        if not self._gr_features_cache:
            return {}

        gr_reqs = [
            (req_id, self._gr_features_cache[req_id])
            for req_id in batch_req_ids
            if req_id in self._gr_features_cache
        ]
        if not gr_reqs:
            return {}
        if len(gr_reqs) != len(batch_req_ids):
            missing_req_ids = [
                req_id for req_id in batch_req_ids if req_id not in self._gr_features_cache
            ]
            raise ValueError(
                "Incomplete GR feature batch: missing decoded features for "
                f"requests {missing_req_ids}"
            )

        self._validate_batch_schema(gr_reqs)

        device = torch.device("cpu")
        return self._adapter.build_model_kwargs(gr_reqs, device)

    def _pool(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
        kv_connector_output: Any,
    ) -> ModelRunnerOutput:
        """Route pooling based on hidden_states type.

        The type of `hidden_states` determines the execution path:

        - torch.Tensor: Standard vLLM pooling path. Delegates to the parent
          GPUModelRunner._pool(), which assumes hidden_states is indexed by
          scheduled token positions (used by embedding models like BERT).

        - list[torch.Tensor]: GR recommendation path. Each tensor in the list
          is a per-request score tensor from HSTU. We bypass the token-centric
          base implementation and construct pooler_output directly.

        This dispatch mechanism allows GRModelRunner to handle both standard
        pooling requests and GR recommendation outputs without modifying the
        vLLM core.
        """
        if isinstance(hidden_states, torch.Tensor):
            return super()._pool(
                hidden_states,
                num_scheduled_tokens,
                num_scheduled_tokens_np,
                kv_connector_output,
            )

        num_reqs = self.input_batch.num_reqs
        assert num_reqs == len(self.input_batch.pooling_params), (
            "Either all or none of the requests in a batch must be pooling request"
        )

        per_req_hidden_states = list(hidden_states[:num_reqs])
        seq_lens_cpu = self.seq_lens.cpu[:num_reqs]

        pool_device = next(
            (tensor.device for tensor in per_req_hidden_states if isinstance(tensor, torch.Tensor)),
            torch.device("cpu"),
        )
        pooling_metadata = self.input_batch.get_pooling_metadata()
        pooling_metadata.build_pooling_cursor(
            num_scheduled_tokens_np,
            seq_lens_cpu,
            device=pool_device,
        )

        raw_pooler_output = self.model.pooler(
            hidden_states=per_req_hidden_states,
            pooling_metadata=pooling_metadata,
        )

        finished_mask = [
            seq_len == prompt_len
            for seq_len, prompt_len in zip(seq_lens_cpu, pooling_metadata.prompt_lens)
        ]

        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids.copy(),
            req_id_to_index=self.input_batch.req_id_to_index.copy(),
            kv_connector_output=kv_connector_output,
        )

        if raw_pooler_output is None or not any(finished_mask):
            model_runner_output.pooler_output = [None] * num_reqs
            return model_runner_output

        if isinstance(raw_pooler_output, torch.Tensor):
            raw_outputs: list[torch.Tensor | None] = [raw_pooler_output]
        else:
            raw_outputs = list(raw_pooler_output)

        if len(raw_outputs) != num_reqs:
            raise ValueError(
                f"GR pooling output count mismatch: num_reqs={num_reqs}, outputs={len(raw_outputs)}"
            )

        raw_outputs = [
            None if tensor is None else tensor.to("cpu", non_blocking=True)
            for tensor in raw_outputs
        ]
        model_runner_output.pooler_output = [
            output if include else None for output, include in zip(raw_outputs, finished_mask)
        ]

        sync_device = getattr(self, "_sync_device", None)
        if callable(sync_device):
            sync_device()

        return model_runner_output

    def _validate_batch_schema(
        self,
        gr_reqs: list[tuple[str, dict[str, Any]]],
    ) -> None:
        """Fail fast when mixed GR requests carry different schemas."""
        base_req_id, base_features = gr_reqs[0]
        expected = str(base_features.get("schema_fingerprint", ""))
        if not expected:
            return

        if "uih_keys" not in base_features:
            mismatches: list[str] = []
            for req_id, features in gr_reqs[1:]:
                actual = str(features.get("schema_fingerprint", ""))
                if actual != expected:
                    mismatches.append(f"{req_id}={actual}")
            if mismatches:
                mismatch_text = ", ".join(mismatches)
                raise ValueError(
                    "Mixed GR batch schema mismatch: "
                    f"expected {base_req_id}={expected}; got {mismatch_text}"
                )
            return

        expected_full_uih_keys = list(
            base_features.get(
                "full_uih_keys",
                base_features["uih_keys"] + base_features["ctx_keys"],
            )
        )
        mismatches = []

        for req_id, features in gr_reqs[1:]:
            actual = str(features["schema_fingerprint"])
            actual_full_uih_keys = list(
                features.get(
                    "full_uih_keys",
                    features["uih_keys"] + features["ctx_keys"],
                )
            )
            if actual != expected or actual_full_uih_keys != expected_full_uih_keys:
                mismatch_desc = f"{req_id}={actual}"
                if actual_full_uih_keys != expected_full_uih_keys:
                    mismatch_desc += f" full_uih_keys={actual_full_uih_keys!r}"
                mismatches.append(mismatch_desc)

        if mismatches:
            mismatch_text = ", ".join(mismatches)
            raise ValueError(
                "Mixed GR batch schema mismatch: "
                f"expected {base_req_id}={expected}; got {mismatch_text}"
            )
