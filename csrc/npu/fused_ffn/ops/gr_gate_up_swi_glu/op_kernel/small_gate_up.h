// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gate_up_swi_glu.h"
namespace gr_ffn::small_gateup {
using namespace AscendC;
using namespace Catlass;
using namespace gr_ffn::swiglu_impl;

// Paired gate/up N16 ownership and exact M16 balance. Compact the valid
// paired columns in L1/L0; the physical GM ring remains 32768 BF16 elements.
// M<=128 uses 128x256 and bypasses single-pass weights. The rest of short
// Prefill uses 256x128, reusing each weight across more M rows.
template <uint32_t BM, uint32_t BN, bool Bypass, uint32_t Shift = 0>
struct Implementation {
  static constexpr uint32_t K = 1024, I = 3072, C = 20, PAIR = BN / 2,
                            NB = PAIR / 16, KW = 256, BK = 64, G = K / KW,
                            S = KW / BK;
  struct Ownership {
    uint32_t core, totalM, begin, end, firstEnd, firstTasks, tasks;
    __aicore__ inline Ownership(uint32_t c, uint32_t m) : core(c), totalM(m) {
      uint32_t total = ((m + BM - 1) / BM) * (I / 16), q = total / C,
               r = total % C;
      begin = c * q + (c < r ? c : r);
      end = begin + q + (c < r);
      firstEnd = (begin / (I / 16) + 1) * (I / 16);
      uint32_t firstCount = (end < firstEnd ? end : firstEnd) - begin;
      firstTasks = (firstCount + NB - 1) / NB;
      tasks =
          firstTasks + (end > firstEnd ? (end - firstEnd + NB - 1) / NB : 0);
    }
    __aicore__ inline uint32_t Micro(uint32_t t) const {
      return t < firstTasks ? begin + t * NB : firstEnd + (t - firstTasks) * NB;
    }
    __aicore__ inline uint32_t Col(uint32_t t) const {
      return Micro(t) % (I / 16) * 16;
    }
    __aicore__ inline uint32_t Cols(uint32_t t) const {
      uint32_t start = Micro(t), left = end - start,
               untilRow = I / 16 - start % (I / 16);
      uint32_t n = left < NB ? left : NB;
      return (n < untilRow ? n : untilRow) * 16;
    }
    __aicore__ inline uint32_t Row(uint32_t t) const {
      uint32_t tile = Micro(t) / (I / 16), tiles = (totalM + BM - 1) / BM;
      uint32_t q = ((totalM + 15) / 16) / tiles,
               r = ((totalM + 15) / 16) % tiles;
      return (tile * q + (tile < r ? tile : r)) * 16;
    }
    __aicore__ inline uint32_t Rows(uint32_t t) const {
      uint32_t tile = Micro(t) / (I / 16), tiles = (totalM + BM - 1) / BM;
      uint32_t q = ((totalM + 15) / 16) / tiles,
               r = ((totalM + 15) / 16) % tiles;
      uint32_t rows = (q + (tile < r)) * 16, left = totalM - Row(t);
      return left < rows ? left : rows;
    }
  };

  using Scheduler = Ownership;
  class StreamingCube {
    LocalTensor<bfloat16_t> a, b, a0, b0;
    LocalTensor<float> c;
    GlobalTensor<bfloat16_t> x, w, ring;
    Scheduler own;
    __aicore__ inline void Preload(uint32_t panel) {
      if (panel >= own.tasks * G) return;
      uint32_t s = panel % 2, t = panel / G, g = panel % G, rows = own.Rows(t),
               ma = (rows + 15) / 16 * 16;
      // Rotate A and its paired gate/up B panels together. Every task
      // still consumes all K panels once; cores start at different K.
      uint32_t kOffset = (g * KW + own.core * Shift) % K;
      WaitFlag<HardEvent::MTE1_MTE2>((event_t)s);
      WaitFlag<HardEvent::MTE1_MTE2>((event_t)(2 + s));
      Nd2NzParams p;
      p.ndNum = 1;
      p.nValue = rows;
      p.dValue = KW;
      p.srcNdMatrixStride = 0;
      p.srcDValue = K;
      p.dstNzC0Stride = ma;
      p.dstNzNStride = 1;
      p.dstNzMatrixStride = 0;
      DataCopy(a[s * BM * KW], x[own.Row(t) * K + kOffset], p);
      SetFlag<HardEvent::MTE2_MTE1>((event_t)(2 + s));
      p.nValue = own.Cols(t);
      p.dstNzC0Stride = 2 * own.Cols(t);
      DataCopy(b[s * BN * KW], w[own.Col(t) * K + kOffset], p);
      DataCopy(b[s * BN * KW + (own.Cols(t)) * 16],
               w[(I + own.Col(t)) * K + kOffset], p);
      SetFlag<HardEvent::MTE2_MTE1>((event_t)s);
    }

   public:
    __aicore__ inline StreamingCube(Arch::Resource<Arch::AtlasA2>& r,
                                    GM_ADDR xp, GM_ADDR wp, GM_ADDR rp,
                                    uint32_t m)
        : own(GetBlockIdx(), m) {
      a = r.l1Buf.GetBufferByByte<bfloat16_t>(0);
      b = r.l1Buf.GetBufferByByte<bfloat16_t>(2 * BM * KW * 2);
      a0 = r.l0ABuf.GetBufferByByte<bfloat16_t>(0);
      b0 = r.l0BBuf.GetBufferByByte<bfloat16_t>(0);
      c = r.l0CBuf.GetBufferByByte<float>(0);
      x.SetGlobalBuffer((__gm__ bfloat16_t*)xp);
      w.SetGlobalBuffer((__gm__ bfloat16_t*)wp);
      if constexpr (Bypass) w.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
      ring.SetGlobalBuffer((__gm__ bfloat16_t*)rp);
    }
    __aicore__ inline void Run() {
      for (uint32_t s = 0; s < 2; ++s) {
        SetFlag<HardEvent::MTE1_MTE2>((event_t)s);
        SetFlag<HardEvent::MTE1_MTE2>((event_t)(2 + s));
        SetFlag<HardEvent::M_MTE1>((event_t)s);
      }
      SetFlag<HardEvent::FIX_M>(EVENT_ID0);
      Preload(0);
      Preload(1);
      for (uint32_t t = 0; t < own.tasks; ++t) {
        uint32_t rows = own.Rows(t), ma = (rows + 15) / 16 * 16,
                 tn = 2 * own.Cols(t);
        constexpr uint8_t pad[4] = {0, 0, 0, 0};
        SetFmatrix(1, ma, pad, FmatrixMode::FMATRIX_LEFT);
        for (uint32_t g = 0; g < G; ++g) {
          uint32_t panel = t * G + g, ls = panel % 2;
          WaitFlag<HardEvent::MTE2_MTE1>((event_t)ls);
          WaitFlag<HardEvent::MTE2_MTE1>((event_t)(2 + ls));
          for (uint32_t ki = 0; ki < S; ++ki) {
            uint32_t s = ki % 2;
            WaitFlag<HardEvent::M_MTE1>((event_t)s);
            LoadData3DParamsV2Pro ap;
            ap.channelSize = KW;
            ap.extConfig =
                (uint64_t(ki * BK) << 32) | (uint64_t(ma) << 16) | BK;
            LoadData<bfloat16_t>(a0[s * BM * BK], a[ls * BM * KW], ap);
            LoadData2DParams bp{0, uint8_t(tn * BK / 256), 1, 0, 0, false, 0};
            LoadData(b0[s * BN * BK], b[ls * BN * KW + ki * tn * BK], bp);
            SetFlag<HardEvent::MTE1_M>((event_t)s);
            if (ki + 1 == S) {
              SetFlag<HardEvent::MTE1_MTE2>((event_t)(2 + ls));
              SetFlag<HardEvent::MTE1_MTE2>((event_t)ls);
              Preload(panel + 2);
            }
            WaitFlag<HardEvent::MTE1_M>((event_t)s);
            if (g == 0 && ki == 0) WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
            MmadParams mp;
            mp.m = ma;
            mp.n = tn;
            mp.k = BK;
            mp.cmatrixInitVal = g == 0 && ki == 0;
            mp.unitFlag = 0;
            Mmad(c, a0[s * BM * BK], b0[s * BN * BK], mp);
            if ((ma / 16) * (tn / 16) < 10) PipeBarrier<PIPE_M>();
            SetFlag<HardEvent::M_MTE1>((event_t)s);
          }
        }
        if (t >= 2) CrossCoreWaitFlag(1);
        SetFlag<HardEvent::M_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
        FixpipeParamsV220 fp;
        fp.nSize = tn;
        fp.mSize = rows < 16 ? 16 : rows;
        fp.srcStride = ma;
        fp.dstStride = BN;
        fp.quantPre = QuantMode_t::F322BF16;
        fp.reluEn = false;
        fp.unitFlag = 0;
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
            ring[WorkspaceTileOffset(own.core, t % 2)], c, fp);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        CrossCoreSetFlag<0x2, PIPE_FIX>(0);
      }
      for (uint32_t s = 0; s < 2; ++s) {
        WaitFlag<HardEvent::MTE1_MTE2>((event_t)s);
        WaitFlag<HardEvent::MTE1_MTE2>((event_t)(2 + s));
        WaitFlag<HardEvent::M_MTE1>((event_t)s);
      }
      WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
      PipeBarrier<PIPE_ALL>();
    }
  };

  __aicore__ static inline void Run(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                                    GM_ADDR ring, uint32_t m) {
    Arch::Resource<Arch::AtlasA2> resource;
    if constexpr (g_coreType == AscendC::AIC) {
      StreamingCube cube(resource, x, w, ring, m);
      cube.Run();
    } else {
      Scheduler own(GetBlockIdx() / GetSubBlockNum(), m);
      BlockSwiGluEpilogue<true, BM, BN> vector(resource,
                                               {y, layout::RowMajor{m, I}});
      GlobalTensor<bfloat16_t> r;
      r.SetGlobalBuffer((__gm__ bfloat16_t*)ring);
      for (uint32_t t = 0; t < own.tasks; ++t) {
        uint32_t rows = own.Rows(t);
        CrossCoreWaitFlag(0);
        vector(GemmCoord{own.Row(t) / BM, 0, 0},
               GemmCoord{rows < 16 ? 16 : rows, BN, K},
               r[WorkspaceTileOffset(own.core, t % 2)], own.Cols(t), own.Col(t),
               own.Cols(t), own.Row(t));
        Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
        CrossCoreSetFlag<0x2, PIPE_MTE3>(1);
      }
      PipeBarrier<PIPE_ALL>();
    }
  }
};
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w, GM_ADDR y, GM_ADDR ring,
                           uint32_t m) {
  if (m <= 128)
    Implementation<128, 256, true>::Run(x, w, y, ring, m);
  else if (m <= 384)
    Implementation<256, 128, false>::Run(x, w, y, ring, m);
  else
    Implementation<256, 128, false, 256>::Run(x, w, y, ring, m);
}
}  // namespace gr_ffn::small_gateup
