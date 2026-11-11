// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
/* Copyright 2025 The xLLM Authors. All Rights Reserved.
 * Copyright 2026 vLLM-GR contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Adapted from xLLM-AI/xllm-ops test/python_test/RegisterOps.cpp.
// This extension adds an output-buffer binding; it does not replace the kernel.
#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>

#include "pytorch_npu_helper.hpp"

namespace {

pybind11::dict BindingInfo() {
  pybind11::dict providers;
  for (const char* name :
       {"aclnnXAttention", "aclnnXAttentionGetWorkspaceSize"}) {
    void* address = GetOpApiFuncAddr(name);
    TORCH_CHECK(address != nullptr, "Cannot resolve ", name);
    Dl_info info{};
    TORCH_CHECK(dladdr(address, &info) != 0 && info.dli_fname != nullptr,
                "Cannot identify provider for ", name);
    providers[name] = info.dli_fname;
  }
  pybind11::dict result;
  result["abi_version"] = 1;
  result["providers"] = providers;
  return result;
}

void CheckInput(const at::Tensor& tensor, const at::Tensor& query,
                const at::Tensor& output, const char* name) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.device() == query.device(), name,
              " must be on the query NPU device");
  // Reject even disjoint views of the same allocation. This conservative
  // binding must not overwrite any input allocation.
  TORCH_CHECK(tensor.storage().unsafeGetStorageImpl() !=
                  output.storage().unsafeGetStorageImpl(),
              "output must not share storage with ", name);
  at::assert_no_overlap(output, tensor);
}

void XAttentionOut(const at::Tensor& query, const at::Tensor& key_cache,
                   const at::Tensor& value_cache,
                   const at::Tensor& unshared_key,
                   const at::Tensor& unshared_value,
                   const c10::optional<at::Tensor>& shared_block_tables,
                   const c10::optional<at::Tensor>& unshared_block_tables,
                   const at::Tensor& actual_shared_kvlen,
                   const at::Tensor& decode_step, double scale_value,
                   at::Tensor output) {
  TORCH_CHECK(query.defined() && output.defined(),
              "query and output must be defined");
  TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1,
              "query must be on NPU");
  TORCH_CHECK(output.device() == query.device(),
              "output must be on the query NPU device");
  TORCH_CHECK(query.dim() == 3 && query.numel() != 0,
              "query must have nonempty [rows, heads, head_dim] shape");
  TORCH_CHECK(
      query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
      "query must have float16 or bfloat16 dtype");
  TORCH_CHECK(output.sizes() == query.sizes(),
              "output shape must match query shape");
  TORCH_CHECK(output.scalar_type() == query.scalar_type(),
              "output dtype must match query dtype");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(at_npu::native::get_npu_format(output) == ACL_FORMAT_ND,
              "output must use ACL_FORMAT_ND (2); no implicit format cast");
  at::assert_no_internal_overlap(output);

  CheckInput(query, query, output, "query");
  CheckInput(key_cache, query, output, "key_cache");
  CheckInput(value_cache, query, output, "value_cache");
  CheckInput(unshared_key, query, output, "unshared_key");
  CheckInput(unshared_value, query, output, "unshared_value");
  CheckInput(actual_shared_kvlen, query, output, "actual_shared_kvlen");
  CheckInput(decode_step, query, output, "decode_step");
  if (shared_block_tables.has_value()) {
    CheckInput(*shared_block_tables, query, output, "shared_block_tables");
  }
  if (unshared_block_tables.has_value()) {
    CheckInput(*unshared_block_tables, query, output, "unshared_block_tables");
  }

  const c10::DeviceGuard guard(query.device());
  // Keep the deployed scale-enabled ACLNN ABI and argument order unchanged.
  // No empty_like, resize, clone, contiguous, or copy is performed here.
  EXEC_NPU_CMD(aclnnXAttention, query, key_cache, value_cache, unshared_key,
               unshared_value, unshared_block_tables, actual_shared_kvlen,
               decode_step, shared_block_tables, scale_value, output);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("binding_info", &BindingInfo,
             "Describe the binding ABI and resolved operator providers");
  module.def("x_attention_out", &XAttentionOut,
             "Run XAttention into a caller-owned contiguous ND output",
             pybind11::arg("query"), pybind11::arg("key_cache"),
             pybind11::arg("value_cache"), pybind11::arg("unshared_key"),
             pybind11::arg("unshared_value"),
             pybind11::arg("shared_block_tables"),
             pybind11::arg("unshared_block_tables"),
             pybind11::arg("actual_shared_kvlen"), pybind11::arg("decode_step"),
             pybind11::arg("scale_value"), pybind11::arg("output"));
}
