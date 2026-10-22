// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "pytorch_npu_helper.hpp"

#include <c10/core/DeviceGuard.h>
#include <limits>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

namespace {
using namespace gr_ffn_detail;

constexpr int64_t kHidden = 1024;
constexpr int64_t kIntermediate = 3072;

void CheckMatrix(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name,
              " must be on NPU");
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16 && tensor.dim() == 2, name,
              " must be a rank-2 BF16 tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  const auto format = at_npu::native::get_npu_format(tensor);
  TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW, name,
              " must have a base ND format (no implicit NZ conversion)");
}

void CheckInputs(const at::Tensor& x, const at::Tensor& weight, int64_t k,
                 int64_t n, int64_t minM) {
  CheckMatrix(x, "x");
  CheckMatrix(weight, "weight");
  TORCH_CHECK(x.device() == weight.device(),
              "x and weight must share a device");
  TORCH_CHECK(x.size(1) == k && weight.size(0) == n && weight.size(1) == k,
              "GrGateUpSwiGlu requires x[M,1024] and weight[6144,1024]");
  TORCH_CHECK(
      x.size(0) >= minM && x.size(0) <= std::numeric_limits<uint32_t>::max(),
      "M outside the release kernel range; use the model's native fallback");
}

}  // namespace
