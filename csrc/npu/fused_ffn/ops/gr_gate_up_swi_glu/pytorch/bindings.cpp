// SPDX-License-Identifier: Apache-2.0
// Adapted from the user-supplied 15.diff.txt GateUp binding. Down stays native.
#include "pytorch.h"

namespace {
void Preflight() {
  // Use the execution path's library handle and symbol resolver without
  // allocating tensors, selecting a device, or launching an ACLNN command.
  (void)GetOpApiFuncAddr("aclnnGrGateUpSwiGluGetWorkspaceSize");
  (void)GetOpApiFuncAddr("aclnnGrGateUpSwiGlu");
}

at::Tensor GateUpSwiGlu(const at::Tensor& x, const at::Tensor& weight) {
  CheckInputs(x, weight, kHidden, 2 * kIntermediate, 1);
  const c10::DeviceGuard guard(x.device());
  auto output = at::empty({x.size(0), kIntermediate}, x.options());
  EXEC_NPU_CMD(aclnnGrGateUpSwiGlu, x, weight, output);
  return output;
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Experimental BF16 GrGateUpSwiGlu H1024/I3072; Down remains native";
  m.def("preflight", &Preflight,
        "Load the dedicated OPAPI library and resolve its required symbols");
  m.def("fused_gate_up_swiglu", &GateUpSwiGlu, pybind11::arg("x"),
        pybind11::arg("weight"));
}
