#pragma once
#include <limits>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace gr_ffn_host {
template <int64_t K, int64_t N, int64_t MinM, class TilingData, class SelectKey>
ge::graphStatus Tiling(gert::TilingContext* context, SelectKey selectKey,
                       size_t workspace) {
  const auto& x = context->GetInputShape(0)->GetOriginShape();
  const auto& w = context->GetInputShape(1)->GetOriginShape();
  if (x.GetDimNum() != 2 || w.GetDimNum() != 2 || x.GetDim(1) != K ||
      w.GetDim(0) != N || w.GetDim(1) != K || x.GetDim(0) < MinM ||
      x.GetDim(0) > std::numeric_limits<uint32_t>::max())
    return ge::GRAPH_FAILED;
  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  if (platform.GetSocVersion() != platform_ascendc::SocVersion::ASCEND910B ||
      platform.GetCoreNumAic() < 20)
    return ge::GRAPH_FAILED;

  const uint32_t m = x.GetDim(0);
  const uint64_t key = selectKey(m);
  TilingData data;
  data.set_m(m);
  data.SaveToBuffer(context->GetRawTilingData()->GetData(),
                    context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(data.GetDataSize());
  // The kernels partition ownership and ring slots over exactly 20 AIC blocks.
  // More physical cores are permitted, but are deliberately not launched.
  context->SetBlockDim(20);
  context->SetTilingKey(key);
  context->GetWorkspaceSizes(1)[0] =
      platform.GetLibApiWorkSpaceSize() + workspace;
  return ge::GRAPH_SUCCESS;
}

template <int64_t OutputN>
ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const auto* x = context->GetInputShape(0);
  if (x->GetDimNum() != 2) return ge::GRAPH_FAILED;
  *context->GetOutputShape(0) = *x;
  context->GetOutputShape(0)->SetDim(1, OutputN);
  return ge::GRAPH_SUCCESS;
}
inline ge::graphStatus InferType(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0, ge::DT_BF16);
  return ge::GRAPH_SUCCESS;
}
}  // namespace gr_ffn_host
