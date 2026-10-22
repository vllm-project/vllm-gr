#include "gate_up_swi_glu.h"
#include "small_gate_up.h"

template <bool LogicalRows>
__aicore__ inline void RunGateUp(GM_ADDR x, GM_ADDR w, GM_ADDR y, GM_ADDR ring,
                                 uint32_t m) {
  using namespace Catlass;
  using namespace gr_ffn::swiglu_impl;
  using Mmad =
      Gemm::Block::BlockMmad<GemmShape<128, 256, 256>, GemmShape<128, 256, 64>,
                             NativeNdGateUpTileCopy>;
  using Scheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
  using Kernel =
      MatmulSwiGluRing<Mmad, Scheduler, BlockSwiGluEpilogue<LogicalRows>>;
  typename Kernel::Arguments args{GemmCoord{m, 6144, 1024}, x, w, ring, y};
  Kernel op;
  op(Kernel::ToUnderlyingArguments(args, nullptr));
}

extern "C" __global__ __aicore__ void gr_gate_up_swi_glu(GM_ADDR x, GM_ADDR w,
                                                         GM_ADDR y,
                                                         GM_ADDR workspace,
                                                         GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  GET_TILING_DATA(data, tiling);
  auto ring = AscendC::GetUserWorkspace(workspace);
  if (TILING_KEY_IS(1))
    RunGateUp<false>(x, w, y, ring, data.m);
  else if (TILING_KEY_IS(2))
    RunGateUp<true>(x, w, y, ring, data.m);
  else if (TILING_KEY_IS(3))
    gr_ffn::small_gateup::Run(x, w, y, ring, data.m);
}
