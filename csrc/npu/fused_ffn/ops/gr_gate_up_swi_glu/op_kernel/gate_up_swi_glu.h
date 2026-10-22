#pragma once
/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include "matmul.h"

namespace Catlass::Arch {

using FlagID = uint16_t;
constexpr FlagID AIV_INTER_SUBBLOCK_BARRIER = 10;

struct CrossCoreFlag {
  CATLASS_DEVICE
  CrossCoreFlag() : id(0) {}

  CATLASS_DEVICE
  CrossCoreFlag(FlagID id) : id(id) {}

  FlagID id;
};

template <uint8_t MODE, int32_t CORE_TYPE>
struct BarrierFlag {
  static_assert(
      MODE != MODE,
      "Unsupported cross core barrier flag, can not find the specialization.");
};

template <>
struct BarrierFlag<0x1, AscendC::AIV> {
  static constexpr FlagID ID = AIV_INTER_SUBBLOCK_BARRIER;
};

template <uint8_t MODE, pipe_t PIPE>
CATLASS_DEVICE void CrossCoreBarrier() {
  constexpr FlagID flagId = BarrierFlag<MODE, g_coreType>::ID;
  AscendC::CrossCoreSetFlag<MODE, PIPE>(flagId);
  AscendC::CrossCoreWaitFlag(flagId);
}

template <uint8_t MODE, pipe_t PIPE>
CATLASS_DEVICE void CrossCoreSetFlag(CrossCoreFlag& flag) {
  AscendC::CrossCoreSetFlag<MODE, PIPE>(flag.id);
}

CATLASS_DEVICE
void CrossCoreWaitFlag(CrossCoreFlag& flag) {
  AscendC::CrossCoreWaitFlag(flag.id);
}

}  // namespace Catlass::Arch

namespace Catlass::Epilogue::Tile {

template <class ArchTag, class GmType>
struct CopyGm2UbAligned {
  static_assert(
      DEPENDENT_FALSE<ArchTag>,
      "Unsupported copy gm to ub aligned, can not find the specialization.");
};

template <typename Element>
struct CopyGm2UbAligned<Arch::AtlasA2,
                        Gemm::GemmType<Element, layout::RowMajor>> {
  using LayoutSrc = layout::RowMajor;
  using LayoutDst = layout::RowMajor;

  static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);
  static constexpr uint32_t BLOCK_LEN_LIMIT = 65536;
  static constexpr uint32_t MAX_REPEAT = 4095;
  static constexpr uint32_t STRIDE_LIMIT = 65536;

  CATLASS_DEVICE
  CopyGm2UbAligned() = default;

  CATLASS_DEVICE
  void operator()(AscendC::LocalTensor<Element> const& dstTensor,
                  AscendC::GlobalTensor<Element> const& srcTensor,
                  layout::RowMajor const& layoutDst,
                  layout::RowMajor const& layoutSrc) {
    uint32_t rows = layoutSrc.shape(0);
    uint32_t cols = layoutSrc.shape(1);
    uint32_t srcStride =
        (layoutSrc.stride(0) - layoutSrc.shape(1)) / ELE_NUM_PER_BLK;
    uint32_t dstStride =
        (layoutDst.stride(0) - layoutDst.shape(1)) / ELE_NUM_PER_BLK;

    if ((layoutSrc.shape(1) == layoutSrc.stride(0)) &&
        (layoutDst.shape(1) == layoutDst.stride(0))) {
      DataCopy(dstTensor, srcTensor, rows * cols);
    } else if (srcStride < STRIDE_LIMIT && dstStride < STRIDE_LIMIT &&
               (cols / ELE_NUM_PER_BLK) < BLOCK_LEN_LIMIT) {
      uint32_t rLoops = CeilDiv(rows, MAX_REPEAT);
      for (uint32_t i = 0; i < rLoops; ++i) {
        uint32_t rActual =
            (i < rLoops - 1) ? MAX_REPEAT : rows - i * MAX_REPEAT;
        AscendC::DataCopyParams dataCopyParams(rActual, cols / ELE_NUM_PER_BLK,
                                               srcStride, dstStride);
        DataCopy(dstTensor[i * MAX_REPEAT * layoutDst.stride(0)],
                 srcTensor[i * MAX_REPEAT * layoutSrc.stride(0)],
                 dataCopyParams);
      }
    } else {
      for (uint32_t i = 0; i < rows; ++i) {
        DataCopy(dstTensor[i * layoutDst.stride(0)],
                 srcTensor[i * layoutSrc.stride(0)], cols);
      }
    }
  };
};

template <class ArchTag, class GmType>
struct CopyUb2GmAligned {
  static_assert(
      DEPENDENT_FALSE<ArchTag>,
      "Unsupported copy ub to gm aligned, can not find the specialization.");
};

template <typename Element>
struct CopyUb2GmAligned<Arch::AtlasA2,
                        Gemm::GemmType<Element, layout::RowMajor>> {
  using LayoutSrc = layout::RowMajor;
  using LayoutDst = layout::RowMajor;

  static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);
  static constexpr uint32_t BLOCK_LEN_LIMIT = 65536;
  static constexpr uint32_t MAX_REPEAT = 4095;
  static constexpr uint32_t STRIDE_LIMIT = 65536;

  CATLASS_DEVICE
  CopyUb2GmAligned() = default;

  CATLASS_DEVICE
  void operator()(AscendC::GlobalTensor<Element> const& dstTensor,
                  AscendC::LocalTensor<Element> const& srcTensor,
                  layout::RowMajor const& layoutDst,
                  layout::RowMajor const& layoutSrc) {
    uint32_t rows = layoutDst.shape(0);
    uint32_t cols = layoutDst.shape(1);
    uint32_t srcStride =
        (layoutSrc.stride(0) - layoutSrc.shape(1)) / ELE_NUM_PER_BLK;
    uint32_t dstStride =
        (layoutDst.stride(0) - layoutDst.shape(1)) / ELE_NUM_PER_BLK;

    if ((layoutSrc.shape(1) == layoutSrc.stride(0)) &&
        (layoutDst.shape(1) == layoutDst.stride(0))) {
      DataCopy(dstTensor, srcTensor, rows * cols);
    } else if (srcStride < STRIDE_LIMIT && dstStride < STRIDE_LIMIT &&
               (cols / ELE_NUM_PER_BLK) < BLOCK_LEN_LIMIT) {
      uint32_t rLoops = CeilDiv(rows, MAX_REPEAT);
      for (uint32_t i = 0; i < rLoops; ++i) {
        uint32_t rActual =
            (i < rLoops - 1) ? MAX_REPEAT : rows - i * MAX_REPEAT;
        AscendC::DataCopyParams dataCopyParams(rActual, cols / ELE_NUM_PER_BLK,
                                               srcStride, dstStride);
        DataCopy(dstTensor[i * MAX_REPEAT * layoutDst.stride(0)],
                 srcTensor[i * MAX_REPEAT * layoutSrc.stride(0)],
                 dataCopyParams);
      }
    } else {
      for (uint32_t i = 0; i < rows; ++i) {
        DataCopy(dstTensor[i * layoutDst.stride(0)],
                 srcTensor[i * layoutSrc.stride(0)], cols);
      }
    }
  };
};

}  // namespace Catlass::Epilogue::Tile

namespace gr_ffn {
namespace swiglu_impl {

using namespace Catlass;

using Element = bfloat16_t;
// The ring and final activation use BF16 row-major storage. The paired
// weight reader below consumes the model's unchanged [2I, H] ND weight.
using CType = Gemm::GemmType<Element, layout::RowMajor>;
using DType = Gemm::GemmType<Element, layout::RowMajor>;

constexpr uint32_t kTileM = 128;
constexpr uint32_t kTileN = 256;
constexpr uint32_t kPairN = 128;
constexpr uint32_t kGateUpHalfN = 3072;
constexpr uint32_t kWorkspaceStages = 2;
constexpr uint32_t kWorkspaceTileElements = kTileM * kTileN;

// The Atlas A2 row-major Fixpipe path is not reliable when mSize is below one
// 16-row fractal.  The fused kernel writes into a physically complete
// 128x256 GM ring tile, so the final store may safely cover 16 rows. The AIC
// scheduler still computes the true logical row count; only the AIV view of
// the final ring fragment is widened to 16 rows. In particular, this keeps a
// 1..15-row M tail inside the original MIX kernel; it does not pad/copy the
// input or launch a second kernel.
template <class BaseCopyL0CToGm>
struct CopyL0CToRingGm : BaseCopyL0CToGm {
  using LayoutDst = typename BaseCopyL0CToGm::LayoutDst;
  using LayoutSrc = typename BaseCopyL0CToGm::LayoutSrc;

  CATLASS_DEVICE CopyL0CToRingGm() = default;

  template <class DstTensor, class SrcTensor>
  CATLASS_DEVICE void operator()(DstTensor const& dst, SrcTensor const& src,
                                 LayoutDst const& dstLayout,
                                 LayoutSrc const& srcLayout,
                                 uint8_t unitFlag = 0) {
    if (dstLayout.shape(0) >= C0_NUM_PER_FRACTAL) {
      BaseCopyL0CToGm::operator()(dst, src, dstLayout, srcLayout, unitFlag);
      return;
    }

    LayoutDst fixpipeLayout{C0_NUM_PER_FRACTAL, dstLayout.shape(1),
                            dstLayout.stride(0)};
    BaseCopyL0CToGm::operator()(dst, src, fixpipeLayout, srcLayout, unitFlag);
  }
};

// Direct v3 ND-weight reader.  A paired MMAD task consumes 128 gate columns
// and the matching 128 up columns.  They are two distant row ranges in the
// existing [2I, H] ND tensor, therefore two ND2NZ transfers write directly
// into the two halves of the original 256-column L1 nZ tile.  No GM staging,
// persistent packed copy, or change to the MMAD/Vector pipeline is required.
struct CopyNativeNdGateUpGmToL1B {
  using LayoutDst = layout::nZ;
  using LayoutSrc = layout::ColumnMajor;

  static constexpr uint32_t kElementsPerC0 = BYTE_PER_C0 / sizeof(Element);

  CATLASS_DEVICE CopyNativeNdGateUpGmToL1B() {}

  CATLASS_DEVICE
  void operator()(AscendC::LocalTensor<Element> const& dstTensor,
                  AscendC::GlobalTensor<Element> const& srcTensor,
                  LayoutDst const& layoutDst, LayoutSrc const& layoutSrc) {
    AscendC::Nd2NzParams params;
    params.ndNum = 1;
    params.nValue = kPairN;
    params.dValue = layoutSrc.shape(0);
    params.srcNdMatrixStride = 0;
    params.srcDValue = layoutSrc.stride(1);
    params.dstNzC0Stride = layoutDst.stride(1) / kElementsPerC0;
    params.dstNzNStride = layoutDst.stride(2) / kElementsPerC0;
    params.dstNzMatrixStride = 0;

    AscendC::DataCopy(dstTensor, srcTensor, params);
    constexpr uint32_t kUpDstOffset = kPairN * kElementsPerC0;
    const uint64_t upSrcOffset =
        static_cast<uint64_t>(kGateUpHalfN) * layoutSrc.stride(1);
    AscendC::DataCopy(dstTensor[kUpDstOffset], srcTensor[upSrcOffset], params);
  }
};

struct NativeNdGateUpTileCopy : Gemm::Tile::FfnTileCopy {
  using CopyL0CToGm = CopyL0CToRingGm<Gemm::Tile::FfnTileCopy::CopyL0CToGm>;
  using CopyGmToL1B = CopyNativeNdGateUpGmToL1B;
};

// AIC and AIV share two GM tiles per AIC, with per-stage lifetime handshakes.
CATLASS_DEVICE
int64_t WorkspaceTileOffset(uint32_t aicoreIndex, uint32_t stage) {
  return static_cast<int64_t>(aicoreIndex * kWorkspaceStages + stage) *
         kWorkspaceTileElements;
}

template <bool LogicalVectorRows = false, uint32_t TileM = 128,
          uint32_t TileN = 256>
class BlockSwiGluEpilogue {
 public:
  static constexpr uint32_t kTileM = TileM, kTileN = TileN, kPairN = TileN / 2;
  using ArchTag = Arch::AtlasA2;
  using ElementC = Element;
  using LayoutC = layout::RowMajor;
  using ElementD = Element;
  using LayoutD = layout::RowMajor;
  using CopyGmToUbC = Epilogue::Tile::CopyGm2UbAligned<ArchTag, CType>;
  using CopyUbToGmD = Epilogue::Tile::CopyUb2GmAligned<ArchTag, DType>;

  struct Params {
    GM_ADDR ptrD;
    LayoutD layoutD;

    CATLASS_HOST_DEVICE Params() {}
    CATLASS_HOST_DEVICE Params(GM_ADDR ptrD_, LayoutD const& layoutD_)
        : ptrD(ptrD_), layoutD(layoutD_) {}
  };

  CATLASS_DEVICE
  BlockSwiGluEpilogue(Arch::Resource<ArchTag>& resource, Params const& params_)
      : params(params_) {
    // Each AIV subcore owns half of its AIC tile. Keep all UB regions
    // disjoint for both the 128x256 and 256x128 paired tiles.
    constexpr uint32_t inputElements = (kTileM / 2) * kTileN;
    constexpr uint32_t outputElements = (kTileM / 2) * kPairN;
    ubC = resource.ubBuf.template GetBufferByByte<Element>(0);
    ubGate = resource.ubBuf.template GetBufferByByte<float>(inputElements *
                                                            sizeof(Element));
    ubUp = resource.ubBuf.template GetBufferByByte<float>(
        inputElements * sizeof(Element) + outputElements * sizeof(float));
    ubTmp = resource.ubBuf.template GetBufferByByte<float>(
        inputElements * sizeof(Element) + 2 * outputElements * sizeof(float));
    ubD = resource.ubBuf.template GetBufferByByte<Element>(
        inputElements * sizeof(Element) + 3 * outputElements * sizeof(float));

    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
  }

  CATLASS_DEVICE
  ~BlockSwiGluEpilogue() {
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
  }

  CATLASS_DEVICE
  void operator()(GemmCoord const& blockCoordMNK,
                  GemmCoord const& actualBlockShapeMNK,
                  AscendC::GlobalTensor<Element> const& gmWorkspaceTile,
                  uint32_t validColumns = kPairN,
                  uint32_t outputColumn = UINT32_MAX,
                  uint32_t pairStride = kPairN,
                  uint32_t outputRow = UINT32_MAX) {
    using namespace AscendC;

    MatrixCoord actualBlockShape = actualBlockShapeMNK.GetCoordMN();
    MatrixCoord subblockShape{CeilDiv(actualBlockShape.row(),
                                      static_cast<uint32_t>(GetSubBlockNum())),
                              actualBlockShape.column()};
    MatrixCoord subblockCoord{GetSubBlockIdx(), 0};
    MatrixCoord subblockOffset = subblockCoord * subblockShape;
    MatrixCoord actualSubblockShape =
        MatrixCoord::Min(subblockShape, actualBlockShape - subblockOffset);
    if (actualSubblockShape.row() <= 0 || actualSubblockShape.column() <= 0) {
      return;
    }

    // The ring slot is always physically 128x256, including the final M
    // tail. Keep that fixed stride while copying only valid rows.
    layout::RowMajor layoutWorkspaceTile{kTileM, kTileN};
    auto gmSubblockC =
        gmWorkspaceTile[layoutWorkspaceTile.GetOffset(subblockOffset)];
    auto layoutSubblockC =
        layoutWorkspaceTile.GetTileLayout(actualSubblockShape);

    MatrixCoord computeOutputShape{actualSubblockShape.row(), kPairN};
    MatrixCoord blockCoord = blockCoordMNK.GetCoordMN();
    MatrixCoord outputOffset{
        (outputRow == UINT32_MAX ? blockCoord.row() * kTileM : outputRow) +
            subblockOffset.row(),
        outputColumn == UINT32_MAX ? blockCoord.column() * kPairN
                                   : outputColumn};
    const int64_t logicalRowsLeft =
        params.layoutD.shape(0) > outputOffset.row()
            ? params.layoutD.shape(0) - outputOffset.row()
            : 0;
    const int64_t validRows = logicalRowsLeft < actualSubblockShape.row()
                                  ? logicalRowsLeft
                                  : actualSubblockShape.row();
    MatrixCoord storeOutputShape{static_cast<MatrixCoord::Index>(validRows),
                                 static_cast<MatrixCoord::Index>(validColumns)};

    layout::RowMajor layoutComputeInUb{
        actualSubblockShape, MakeCoord(static_cast<int64_t>(kTileN), 1L)};
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    copyGmToUbC(ubC, gmSubblockC, layoutComputeInUb, layoutSubblockC);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    // The short-tail workaround deliberately separates the physical
    // transfer from the logical Vector extent.  MTE2 above still copies
    // one complete 8-row half-fractal on each AIV subcore; the isolated
    // LogicalVectorRows route merely skips arithmetic for padded rows.
    const uint32_t physicalRows =
        static_cast<uint32_t>(actualSubblockShape.row());
    const bool logicalShortTail = LogicalVectorRows && validRows < physicalRows;
    const uint32_t rows =
        logicalShortTail ? static_cast<uint32_t>(validRows) : physicalRows;
    if (rows > 0U) {
      // Deinterleave all rows with four Vector instructions.  Each
      // repeat converts one 64-column slab; both the BF16 source
      // rows and compact FP32 destination rows advance by 16 data
      // blocks.
      constexpr uint32_t kFp32VectorSize = 64;
      const UnaryRepeatParams castInputParams(1, 1, kPairN / 8, kTileN / 16);
      SetMaskNorm();
      SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
      Cast<float, Element, false>(ubGate, ubC, RoundMode::CAST_NONE,
                                  (uint64_t)0, rows, castInputParams);
      if constexpr (kPairN > kFp32VectorSize)
        Cast<float, Element, false>(ubGate[kFp32VectorSize],
                                    ubC[kFp32VectorSize], RoundMode::CAST_NONE,
                                    (uint64_t)0, rows, castInputParams);
      Cast<float, Element, false>(ubUp, ubC[pairStride], RoundMode::CAST_NONE,
                                  (uint64_t)0, rows, castInputParams);
      if constexpr (kPairN > kFp32VectorSize)
        Cast<float, Element, false>(
            ubUp[kFp32VectorSize], ubC[pairStride + kFp32VectorSize],
            RoundMode::CAST_NONE, (uint64_t)0, rows, castInputParams);
      PipeBarrier<PIPE_V>();

      const uint32_t count = rows * kPairN;
      if constexpr (LogicalVectorRows) {
        // Issue explicit 64-FP32 repeats only for the sub-8-row
        // padded fragment. All full and naturally sized fragments
        // stay byte-for-byte on the production count path.
        if (logicalShortTail) {
          constexpr uint32_t kFp32VectorSize = 64;
          const uint32_t repeats = count / kFp32VectorSize;
          const UnaryRepeatParams unaryParams(1, 1, 8, 8);
          const BinaryRepeatParams binaryParams(1, 1, 1, 8, 8, 8);
          SetMaskNorm();
          SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
          Muls<float, false>(ubTmp, ubGate, -1.0f, (uint64_t)0, repeats,
                             unaryParams);
          PipeBarrier<PIPE_V>();
          Exp<float, false>(ubTmp, ubTmp, (uint64_t)0, repeats, unaryParams);
          PipeBarrier<PIPE_V>();
          Adds<float, false>(ubTmp, ubTmp, 1.0f, (uint64_t)0, repeats,
                             unaryParams);
          PipeBarrier<PIPE_V>();
          Div<float, false>(ubTmp, ubGate, ubTmp, (uint64_t)0, repeats,
                            binaryParams);
          PipeBarrier<PIPE_V>();
          Mul<float, false>(ubTmp, ubTmp, ubUp, (uint64_t)0, repeats,
                            binaryParams);
          PipeBarrier<PIPE_V>();
          Cast<Element, float, false>(ubD, ubTmp, RoundMode::CAST_RINT,
                                      (uint64_t)0, repeats,
                                      UnaryRepeatParams(1, 1, 4, 8));
        } else {
          Muls(ubTmp, ubGate, -1.0f, count);
          Exp(ubTmp, ubTmp, count);
          Adds(ubTmp, ubTmp, 1.0f, count);
          Div(ubGate, ubGate, ubTmp, count);
          Mul(ubGate, ubGate, ubUp, count);
          Cast(ubD, ubGate, RoundMode::CAST_RINT, count);
        }
      } else {
        Muls(ubTmp, ubGate, -1.0f, count);
        Exp(ubTmp, ubTmp, count);
        Adds(ubTmp, ubTmp, 1.0f, count);
        Div(ubGate, ubGate, ubTmp, count);
        Mul(ubGate, ubGate, ubUp, count);
        Cast(ubD, ubGate, RoundMode::CAST_RINT, count);
      }
    }
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);

    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    if (validRows > 0) {
      GlobalTensor<Element> gmD;
      gmD.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(params.ptrD));
      auto gmSubblockD = gmD[params.layoutD.GetOffset(outputOffset)];
      auto layoutSubblockD = params.layoutD.GetTileLayout(storeOutputShape);
      layout::RowMajor layoutStoreOutUb{
          storeOutputShape, MakeCoord(static_cast<int64_t>(kPairN), 1L)};
      copyUbToGmD(gmSubblockD, ubD, layoutSubblockD, layoutStoreOutUb);
    }
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
  }

 private:
  Params params;
  AscendC::LocalTensor<Element> ubC;
  AscendC::LocalTensor<float> ubGate;
  AscendC::LocalTensor<float> ubUp;
  AscendC::LocalTensor<float> ubTmp;
  AscendC::LocalTensor<Element> ubD;
  CopyGmToUbC copyGmToUbC;
  CopyUbToGmD copyUbToGmD;
};

template <class BlockMmad, class BlockScheduler,
          class BlockEpilogue_ = BlockSwiGluEpilogue<>>
class MatmulSwiGluRing {
 public:
  using ArchTag = typename BlockMmad::ArchTag;
  using L1TileShape = typename BlockMmad::L1TileShape;
  using ElementA = typename BlockMmad::ElementA;
  using LayoutA = typename BlockMmad::LayoutA;
  using ElementB = typename BlockMmad::ElementB;
  using LayoutB = typename BlockMmad::LayoutB;
  using ElementC = typename BlockMmad::ElementC;
  using BlockEpilogue = BlockEpilogue_;

  struct Params {
    GemmCoord problemShape;
    GM_ADDR ptrA;
    LayoutA layoutA;
    GM_ADDR ptrB;
    LayoutB layoutB;
    GM_ADDR ptrWorkspace;
    typename BlockEpilogue::Params epilogueParams;
  };

  struct Arguments {
    GemmCoord problemShape;
    GM_ADDR ptrA;
    GM_ADDR ptrB;
    GM_ADDR ptrWorkspace;
    GM_ADDR ptrD;
  };

  CATLASS_DEVICE
  static Params ToUnderlyingArguments(Arguments const& args, uint8_t*) {
    const uint32_t m = args.problemShape.m();
    const uint32_t n = args.problemShape.n();
    const uint32_t k = args.problemShape.k();
    return Params{
        args.problemShape,
        args.ptrA,
        LayoutA{m, k},
        args.ptrB,
        LayoutB::template MakeLayout<ElementB>(k, n),
        args.ptrWorkspace,
        typename BlockEpilogue::Params{args.ptrD, layout::RowMajor{m, n / 2}}};
  }

  CATLASS_DEVICE MatmulSwiGluRing() {}

  template <int32_t CORE_TYPE = g_coreType>
  CATLASS_DEVICE void operator()(Params const& params);

  template <>
  CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params) {
    BlockScheduler scheduler(params.problemShape,
                             MakeCoord(L1TileShape::M, L1TileShape::N));
    const uint32_t tasks = scheduler.GetCoreLoops();
    const uint32_t core = AscendC::GetBlockIdx();
    const uint32_t cores = AscendC::GetBlockNum();

    BlockMmad blockMmad(resource);
    AscendC::GlobalTensor<ElementA> gmA;
    AscendC::GlobalTensor<ElementB> gmB;
    AscendC::GlobalTensor<ElementC> gmWorkspace;
    gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(params.ptrA));
    gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(params.ptrB));
    gmWorkspace.SetGlobalBuffer(
        reinterpret_cast<__gm__ ElementC*>(params.ptrWorkspace));
    layout::RowMajor layoutWorkspaceTile{kTileM, kTileN};

    uint32_t localTask = 0;
    for (uint32_t task = core; task < tasks; task += cores, ++localTask) {
      const uint32_t stage = localTask % kWorkspaceStages;
      if (localTask >= kWorkspaceStages) {
        // Reuse slot localTask % 2 only after both AIV subcores have
        // finished reading its previous contents from GM.
        Arch::CrossCoreWaitFlag(workspaceFree);
      }
      GemmCoord coord = scheduler.GetBlockCoord(task);
      GemmCoord actual = scheduler.GetActualBlockShape(coord);

      GemmCoord nextCoord = coord;
      GemmCoord nextActual = actual;
      const bool hasNext = task + cores < tasks;
      if (hasNext) {
        nextCoord = scheduler.GetBlockCoord(task + cores);
        nextActual = scheduler.GetActualBlockShape(nextCoord);
      }

      MatrixCoord offA{coord.m() * L1TileShape::M, coord.k() * L1TileShape::K};
      constexpr uint32_t kWeightTaskStride = kPairN;
      MatrixCoord offB{coord.k() * L1TileShape::K,
                       coord.n() * kWeightTaskStride};
      MatrixCoord nextOffA{nextCoord.m() * L1TileShape::M,
                           nextCoord.k() * L1TileShape::K};
      MatrixCoord nextOffB{nextCoord.k() * L1TileShape::K,
                           nextCoord.n() * kWeightTaskStride};

      auto gmSlot = gmWorkspace[WorkspaceTileOffset(core, stage)];
      blockMmad(gmA[params.layoutA.GetOffset(offA)], params.layoutA,
                gmB[params.layoutB.GetOffset(offB)], params.layoutB, gmSlot,
                layoutWorkspaceTile, gmA[params.layoutA.GetOffset(nextOffA)],
                gmB[params.layoutB.GetOffset(nextOffB)], actual, nextActual,
                localTask == 0, hasNext);

      Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(workspaceReady);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
  }

  template <>
  CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params) {
    BlockScheduler scheduler(params.problemShape,
                             MakeCoord(L1TileShape::M, L1TileShape::N));
    const uint32_t tasks = scheduler.GetCoreLoops();
    const uint32_t core = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
    const uint32_t cores = AscendC::GetBlockNum();

    BlockEpilogue epilogue(resource, params.epilogueParams);
    AscendC::GlobalTensor<ElementC> gmWorkspace;
    gmWorkspace.SetGlobalBuffer(
        reinterpret_cast<__gm__ ElementC*>(params.ptrWorkspace));

    uint32_t localTask = 0;
    for (uint32_t task = core; task < tasks; task += cores, ++localTask) {
      GemmCoord coord = scheduler.GetBlockCoord(task);
      GemmCoord actual = scheduler.GetActualBlockShape(coord);
      if (actual.m() < C0_NUM_PER_FRACTAL) {
        // The AIC stored one complete 16-row fractal in the ring.
        // Let both AIV subcores consume eight rows so neither the
        // GM->UB tail transfer falls below its reliable block
        // count. The epilogue still bounds its final MTE3 store to
        // the logical row count.
        actual = GemmCoord{C0_NUM_PER_FRACTAL, actual.n(), actual.k()};
      }
      const uint32_t stage = localTask % kWorkspaceStages;
      auto gmSlot = gmWorkspace[WorkspaceTileOffset(core, stage)];

      Arch::CrossCoreWaitFlag(workspaceReady);
      epilogue(coord, actual, gmSlot);
      Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
      Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(workspaceFree);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
  }

 private:
  static constexpr Arch::FlagID kReadyFlag = 0;
  static constexpr Arch::FlagID kFreeFlag = 1;
  Arch::CrossCoreFlag workspaceReady{kReadyFlag};
  Arch::CrossCoreFlag workspaceFree{kFreeFlag};
  Arch::Resource<ArchTag> resource;
};

}  // namespace swiglu_impl

}  // namespace gr_ffn
