#include "host.h"
#include "gr_gate_up_swi_glu_tiling.h"

namespace {
uint64_t SelectKey(uint32_t m) {
  if (m <= 256) return 3;
  return m % 128 >= 1 && m % 128 < 16 ? 2 : 1;
}

ge::graphStatus Tiling(gert::TilingContext* context) {
  return gr_ffn_host::Tiling<1024, 6144, 1, optiling::GrGateUpSwiGluTilingData>(
      context, SelectKey, size_t(20) * 2 * 128 * 256 * sizeof(uint16_t));
}
}  // namespace

namespace ops {
class GrGateUpSwiGlu : public OpDef {
 public:
  explicit GrGateUpSwiGlu(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("weight")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->SetInferShape(gr_ffn_host::InferShape<3072>)
        .SetInferDataType(gr_ffn_host::InferType);
    this->AICore().SetTiling(Tiling).AddConfig("ascend910b");
  }
};
OP_ADD(GrGateUpSwiGlu);
}  // namespace ops
