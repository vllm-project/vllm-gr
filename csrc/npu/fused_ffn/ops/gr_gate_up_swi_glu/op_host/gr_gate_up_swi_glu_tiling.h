#pragma once
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GrGateUpSwiGluTilingData)
TILING_DATA_FIELD_DEF(uint32_t, m);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GrGateUpSwiGlu, GrGateUpSwiGluTilingData)
}  // namespace optiling
