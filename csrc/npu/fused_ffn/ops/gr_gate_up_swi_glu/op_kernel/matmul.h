#pragma once
#ifndef K_MAX_SHAPE_DIM
  #define K_MAX_SHAPE_DIM 0
#endif

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

// BF16 primitives used by this package's two MMAD pipelines.
#include <cstdint>
#include <type_traits>
#include <kernel_operator.h>
#include "lib/matmul_intf.h"

#define CATLASS_DEVICE __forceinline__ __aicore__
#ifdef __CCE__
  #define CATLASS_HOST_DEVICE __forceinline__[host, aicore]
#else
  #define CATLASS_HOST_DEVICE
#endif
#define CATLASS_GLOBAL __global__ __aicore__

template <uint32_t ALIGN, typename T>
CATLASS_HOST_DEVICE constexpr T RoundUp(const T& val) {
  static_assert(ALIGN != 0, "ALIGN must not be 0");
  return (val + ALIGN - 1) / ALIGN * ALIGN;
}

template <uint32_t DIVISOR, typename T>
CATLASS_HOST_DEVICE constexpr T CeilDiv(const T dividend) {
  static_assert(DIVISOR != 0, "DIVISOR must not be 0");
  return (dividend + DIVISOR - 1) / DIVISOR;
}

template <class T, class U>
CATLASS_HOST_DEVICE constexpr auto CeilDiv(T const& dividend,
                                           U const& divisor) {
  return (dividend + divisor - 1) / divisor;
}

template <bool VALUE, class... Args>
constexpr bool DEPENDENT_BOOL_VALUE = VALUE;

template <class... Args>
constexpr bool DEPENDENT_FALSE = DEPENDENT_BOOL_VALUE<false, Args...>;

namespace Catlass {

constexpr uint32_t BYTE_PER_C0 = 32;
constexpr uint32_t BYTE_PER_C2 = 64;
constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
constexpr uint32_t BYTE_PER_FRACTAL = BYTE_PER_C0 * C0_NUM_PER_FRACTAL;

constexpr uint32_t BYTE_PER_BLK = 32;
constexpr uint32_t BLK_NUM_PER_VECTOR_FRACTAL = 8;
constexpr uint32_t BYTE_PER_VECTOR_FRACTAL =
    BYTE_PER_BLK * BLK_NUM_PER_VECTOR_FRACTAL;

constexpr uint32_t STRIDE_LIMIT = 65536;

}  // namespace Catlass

namespace Catlass::Arch {

struct AtlasA2 {
  static constexpr uint32_t UB_SIZE = 192 * 1024;
  static constexpr uint32_t L1_SIZE = 512 * 1024;
  static constexpr uint32_t L0A_SIZE = 64 * 1024;
  static constexpr uint32_t L0B_SIZE = 64 * 1024;
  static constexpr uint32_t L0C_SIZE = 128 * 1024;
};

}  // namespace Catlass::Arch

namespace Catlass {

template <int RANK_,                ///< Logical rank of coordinate
          class Index_ = uint32_t,  ///< Index type used for each dimension
          class LongIndex_ =
              int64_t  ///< Long index type used for linear offsets
          >
struct Coord {
 public:
  // Number of elements in Coord
  static const int RANK = RANK_;

  // Index typen used to store elements
  using Index = Index_;

  // Type used to represent linear offsets
  using LongIndex = LongIndex_;

  // Default ctor initializes uniformly
  CATLASS_HOST_DEVICE constexpr explicit Coord(Index value = Index(0)) {
    for (int i = 0; i < RANK; ++i) {
      idx[i] = value;
    }
  }

  // Constructs from an array of integers
  CATLASS_HOST_DEVICE constexpr Coord(Index const (&idx_)[RANK]) {
    for (int i = 0; i < RANK; ++i) {
      idx[i] = idx_[i];
    }
  }

  // Element-wise addition
  CATLASS_HOST_DEVICE
  Coord operator+(Coord const& b) const {
    Coord c;
    for (int i = 0; i < RANK; ++i) {
      c.idx[i] = idx[i] + b.idx[i];
    }
    return c;
  }

  // Element-wise subtraction
  CATLASS_HOST_DEVICE
  Coord operator-(Coord const& b) const {
    Coord c;
    for (int i = 0; i < RANK; i++) {
      c.idx[i] = idx[i] - b.idx[i];
    }
    return c;
  }

  // Subtract a scalar from each element
  CATLASS_HOST_DEVICE
  Coord operator-(Index const val) const {
    Coord c;
    for (int i = 0; i < RANK; ++i) {
      c.idx[i] = idx[i] - val;
    }
    return c;
  }

  // Element-wise multiply
  CATLASS_HOST_DEVICE
  Coord operator*(Coord const& b) const {
    Coord c;
    for (int i = 0; i < RANK; i++) {
      c.idx[i] = idx[i] * b.idx[i];
    }
    return c;
  }

  // Element-wise division
  CATLASS_HOST_DEVICE
  Coord operator/(Coord const& b) const {
    Coord c;
    for (int i = 0; i < RANK; i++) {
      c.idx[i] = idx[i] / b.idx[i];
    }
    return c;
  }

  // Member access operator
  CATLASS_HOST_DEVICE
  Index& operator[](int dim) { return idx[dim]; }

  // Member access operator
  CATLASS_HOST_DEVICE
  Index const& operator[](int dim) const { return idx[dim]; }

  // Access via index; may limit unrolling potential
  CATLASS_HOST_DEVICE
  Index& At(int dim) { return idx[dim]; }

  // Access via index; may limit unrolling potential
  CATLASS_HOST_DEVICE
  Index const& At(int dim) const { return idx[dim]; }

  template <int... Is>
  CATLASS_HOST_DEVICE auto GetCoordByAxis() const {
    Index idx_[sizeof...(Is)]{idx[Is]...};
    return Coord<sizeof...(Is), Index, LongIndex>{idx_};
  }

  CATLASS_HOST_DEVICE
  static Coord Min(Coord const& a, Coord const& b) {
    Coord res;
    for (int i = 0; i < RANK; ++i) {
      res[i] = a[i] < b[i] ? a[i] : b[i];
    }
    return res;
  }

 private:
  // Indices
  Index idx[RANK];
};

template <class T>
CATLASS_HOST_DEVICE constexpr Coord<2, T> MakeCoord(T dim0, T dim1) {
  T values[2] = {dim0, dim1};
  return Coord<2, T>(values);
}

template <class T>
CATLASS_HOST_DEVICE constexpr Coord<3, T> MakeCoord(T dim0, T dim1, T dim2) {
  T values[3] = {dim0, dim1, dim2};
  return Coord<3, T>(values);
}

template <class T>
CATLASS_HOST_DEVICE constexpr Coord<4, T> MakeCoord(T dim0, T dim1, T dim2,
                                                    T dim3) {
  T values[4] = {dim0, dim1, dim2, dim3};
  return Coord<4, T>(values);
}

}  // namespace Catlass

namespace Catlass {

struct MatrixCoord : public Coord<2, uint32_t> {
  using Index = uint32_t;

  using Base = Coord<2, Index>;

  using LongIndex = typename Base::LongIndex;

  static constexpr uint32_t ROW_INDEX = 0;

  static constexpr uint32_t COLUMN_INDEX = 1;

  CATLASS_HOST_DEVICE
  MatrixCoord() {}

  CATLASS_HOST_DEVICE
  MatrixCoord(Coord<2, Index> const& coord) : Base(coord) {}

  CATLASS_HOST_DEVICE
  MatrixCoord(Index row, Index column) : Base(MakeCoord(row, column)) {}

  CATLASS_HOST_DEVICE
  MatrixCoord(LongIndex row, LongIndex column)
      : Base(MakeCoord(Index(row), Index(column))) {}

  CATLASS_HOST_DEVICE
  Index const& row() const { return this->At(ROW_INDEX); }

  CATLASS_HOST_DEVICE
  Index& row() { return this->At(ROW_INDEX); }

  CATLASS_HOST_DEVICE
  Index const& column() const { return this->At(COLUMN_INDEX); }

  CATLASS_HOST_DEVICE
  Index& column() { return this->At(COLUMN_INDEX); }

  CATLASS_HOST_DEVICE
  MatrixCoord operator+(Base const& b) const {
    return MatrixCoord(Base::operator+(b));
  }
};

}  // namespace Catlass

namespace Catlass::layout {

struct RowMajor {
 public:
  static constexpr int RANK = 2;

  using Index = uint32_t;

  using LongIndex = int64_t;

  using Shape = Coord<RANK, Index>;

  using Stride = Coord<RANK, LongIndex>;

 public:
  CATLASS_HOST_DEVICE
  RowMajor(Index rows = 0, Index cols = 0)
      : shape_(MakeCoord(rows, cols)),
        stride_(MakeCoord(LongIndex(cols), LongIndex(1))) {}

  CATLASS_HOST_DEVICE
  RowMajor(Index rows, Index cols, LongIndex ldm)
      : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(ldm, LongIndex(1))) {}

  CATLASS_HOST_DEVICE
  RowMajor(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

  template <class Element>
  CATLASS_HOST_DEVICE static RowMajor MakeLayout(Index rows, Index cols) {
    return RowMajor(rows, cols);
  }

  CATLASS_HOST_DEVICE
  LongIndex GetOffset(MatrixCoord const& coord) const {
    return LongIndex(coord.row()) * stride_[0] + LongIndex(coord.column());
  }

  CATLASS_HOST_DEVICE
  RowMajor GetTileLayout(MatrixCoord const& tileShape) const {
    return RowMajor(tileShape, stride());
  }

  CATLASS_HOST_DEVICE
  typename Shape::Index shape(int idx) const { return shape_[idx]; }

  CATLASS_HOST_DEVICE
  Stride stride() const { return stride_; }

  CATLASS_HOST_DEVICE
  typename Stride::Index stride(int idx) const { return stride_[idx]; }

 private:
  //
  // Data members
  //

  Shape shape_;

  Stride stride_;
};

struct ColumnMajor {
 public:
  static constexpr int RANK = 2;

  using Index = uint32_t;

  using LongIndex = int64_t;

  using Shape = Coord<RANK, Index>;

  using Stride = Coord<RANK, LongIndex>;

 public:
  // Methods

  CATLASS_HOST_DEVICE
  ColumnMajor(Index rows = 0, Index cols = 0)
      : shape_(MakeCoord(rows, cols)),
        stride_(MakeCoord(LongIndex(1), LongIndex(rows))) {}

  CATLASS_HOST_DEVICE
  ColumnMajor(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

  template <class Element>
  CATLASS_HOST_DEVICE static ColumnMajor MakeLayout(Index rows, Index cols) {
    return ColumnMajor(rows, cols);
  }

  CATLASS_HOST_DEVICE
  LongIndex GetOffset(MatrixCoord const& coord) const {
    return LongIndex(coord.row()) + LongIndex(coord.column()) * stride_[1];
  }

  CATLASS_HOST_DEVICE
  ColumnMajor GetTileLayout(MatrixCoord const& tileShape) const {
    return ColumnMajor(tileShape, stride());
  }

  CATLASS_HOST_DEVICE
  typename Shape::Index shape(int idx) const { return shape_[idx]; }

  CATLASS_HOST_DEVICE
  Stride stride() const { return stride_; }

  CATLASS_HOST_DEVICE
  typename Stride::Index stride(int idx) const { return stride_[idx]; }

 private:
  //
  // Data members
  //

  Shape shape_;

  Stride stride_;
};

struct nZ {
 public:
  static constexpr int RANK = 4;

  using Index = uint32_t;

  using LongIndex = int64_t;

  static constexpr int ORG_SHAPE_RANK = 2;

  using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

  using Shape = Coord<RANK, Index>;

  using Stride = Coord<RANK, LongIndex>;

 public:
  // Methods

  CATLASS_HOST_DEVICE constexpr nZ(
      Index orgRows = 0,        /// Number of rows of origin matrices
      Index orgCols = 0,        /// Number of cols of origin matrices
      Index rowsInFractal = 0,  /// Number of rows inside the fractal
      Index rowsByFractal = 0,  /// number of rows by the fractal
      Index colsInFractal = 0,  /// number of cols inside the fractal
      Index colsByFractal = 0,  /// number of cols by the fractal
      LongIndex strideRowsInFractal =
          0,  /// number of elements between adjacent rows inside the fractal
      LongIndex strideRowsByFractal =
          0,  /// number of elements between adjacent fractal rows
      LongIndex strideColsInFractal =
          0,  /// number of elements between adjacent cols inside the fractal
      LongIndex strideColsByFractal =
          0)  /// number of elements between adjacent fractal cols
      : orgShape_(MakeCoord(orgRows, orgCols)),
        shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal,
                         colsByFractal)),
        stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal,
                          strideColsInFractal, strideColsByFractal)) {}

  CATLASS_HOST_DEVICE constexpr nZ(OrgShape orgShape, Shape shape,
                                   Stride stride)
      : orgShape_(orgShape), shape_(shape), stride_(stride) {}

  template <class Element>
  CATLASS_HOST_DEVICE constexpr static nZ MakeLayout(Index orgRows,
                                                     Index orgCols) {
    constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);
    Index rowsRound = RoundUp<ELE_NUM_PER_C0>(orgRows);
    Index colsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgCols);
    return nZ(orgRows, orgCols, ELE_NUM_PER_C0, rowsRound / ELE_NUM_PER_C0,
              C0_NUM_PER_FRACTAL, colsRound / C0_NUM_PER_FRACTAL, 1,
              colsRound * ELE_NUM_PER_C0, ELE_NUM_PER_C0, ELE_NUM_PER_FRACTAL);
  }

  CATLASS_HOST_DEVICE
  LongIndex GetOffset(MatrixCoord const& coord) const {
    return LongIndex(coord.row()) / shape_[0] * stride_[1] +
           LongIndex(coord.column()) / shape_[2] * stride_[3] +
           (LongIndex(coord.row()) % shape_[0]) * stride_[0] +
           (LongIndex(coord.column()) % shape_[2]) * stride_[2];
  }

  CATLASS_HOST_DEVICE
  nZ GetTileLayout(MatrixCoord const& tileOriShape) const {
    auto tileShape =
        MakeCoord(shape(0), CeilDiv(tileOriShape.row(), shape(0)), shape(2),
                  CeilDiv(tileOriShape.column(), shape(2)));
    return nZ(tileOriShape, tileShape, stride());
  }

  CATLASS_HOST_DEVICE
  typename OrgShape::Index orgShape(int idx) const { return orgShape_[idx]; }

  CATLASS_HOST_DEVICE
  typename Shape::Index shape(int idx) const { return shape_[idx]; }

  CATLASS_HOST_DEVICE
  Stride stride() const { return stride_; }

  CATLASS_HOST_DEVICE
  typename Stride::Index stride(int idx) const { return stride_[idx]; }

 private:
  OrgShape orgShape_;

  Shape shape_;

  Stride stride_;
};

struct zN {
 public:
  static constexpr int RANK = 4;

  using Index = uint32_t;

  using LongIndex = int64_t;

  static constexpr int ORG_SHAPE_RANK = 2;

  using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

  using Shape = Coord<RANK, Index>;

  using Stride = Coord<RANK, LongIndex>;

 public:
  // Methods

  CATLASS_HOST_DEVICE constexpr zN(
      Index orgRows = 0,        /// Number of rows of origin matrices
      Index orgCols = 0,        /// Number of cols of origin matrices
      Index rowsInFractal = 0,  /// Number of rows inside the fractal
      Index rowsByFractal = 0,  /// number of rows by the fractal
      Index colsInFractal = 0,  /// number of cols inside the fractal
      Index colsByFractal = 0,  /// number of cols by the fractal
      LongIndex strideRowsInFractal =
          0,  /// number of elements between adjacent rows inside the fractal
      LongIndex strideRowsByFractal =
          0,  /// number of elements between adjacent fractal rows
      LongIndex strideColsInFractal =
          0,  /// number of elements between adjacent cols inside the fractal
      LongIndex strideColsByFractal =
          0)  /// number of elements between adjacent fractal cols
      : orgShape_(MakeCoord(orgRows, orgCols)),
        shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal,
                         colsByFractal)),
        stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal,
                          strideColsInFractal, strideColsByFractal)) {}

  CATLASS_HOST_DEVICE constexpr zN(OrgShape orgShape, Shape shape,
                                   Stride stride)
      : orgShape_(orgShape), shape_(shape), stride_(stride) {}

  template <class Element>
  CATLASS_HOST_DEVICE constexpr static zN MakeLayout(Index orgRows,
                                                     Index orgCols) {
    constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);
    Index rowsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgRows);
    Index colsRound = RoundUp<ELE_NUM_PER_C0>(orgCols);
    return zN(orgRows, orgCols, C0_NUM_PER_FRACTAL,
              rowsRound / C0_NUM_PER_FRACTAL, ELE_NUM_PER_C0,
              colsRound / ELE_NUM_PER_C0, ELE_NUM_PER_C0, ELE_NUM_PER_FRACTAL,
              1, rowsRound * ELE_NUM_PER_C0);
  }

  CATLASS_HOST_DEVICE
  static zN MakeLayoutInL0C(MatrixCoord const& shape) {
    return zN(shape.row(), shape.column(), C0_NUM_PER_FRACTAL,
              CeilDiv<C0_NUM_PER_FRACTAL>(shape.row()), C0_NUM_PER_FRACTAL,
              CeilDiv<C0_NUM_PER_FRACTAL>(shape.column()), C0_NUM_PER_FRACTAL,
              C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL, 1,
              RoundUp<C0_NUM_PER_FRACTAL>(shape.row()) * C0_NUM_PER_FRACTAL);
  }

  CATLASS_HOST_DEVICE
  LongIndex GetOffset(MatrixCoord const& coord) const {
    return LongIndex(coord.row()) / shape_[0] * stride_[1] +
           LongIndex(coord.column()) / shape_[2] * stride_[3] +
           (LongIndex(coord.row()) % shape_[0]) * stride_[0] +
           (LongIndex(coord.column()) % shape_[2]) * stride_[2];
  }

  CATLASS_HOST_DEVICE
  zN GetTileLayout(MatrixCoord const& tileOriShape) const {
    auto tileShape =
        MakeCoord(shape(0), CeilDiv(tileOriShape.row(), shape(0)), shape(2),
                  CeilDiv(tileOriShape.column(), shape(2)));
    return zN(tileOriShape, tileShape, stride());
  }

  CATLASS_HOST_DEVICE
  typename OrgShape::Index orgShape(int idx) const { return orgShape_[idx]; }

  CATLASS_HOST_DEVICE
  typename Shape::Index shape(int idx) const { return shape_[idx]; }

  CATLASS_HOST_DEVICE
  Stride stride() const { return stride_; }

  CATLASS_HOST_DEVICE
  typename Stride::Index stride(int idx) const { return stride_[idx]; }

 private:
  OrgShape orgShape_;

  Shape shape_;

  Stride stride_;
};

struct zZ {
 public:
  static constexpr int RANK = 4;

  using Index = uint32_t;

  using LongIndex = int64_t;

  static constexpr int ORG_SHAPE_RANK = 2;

  using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

  using Shape = Coord<RANK, Index>;

  using Stride = Coord<RANK, LongIndex>;

 public:
  // Methods

  CATLASS_HOST_DEVICE constexpr zZ(
      Index orgRows = 0,        /// Number of rows of origin matrices
      Index orgCols = 0,        /// Number of cols of origin matrices
      Index rowsInFractal = 0,  /// Number of rows inside the fractal
      Index rowsByFractal = 0,  /// number of rows by the fractal
      Index colsInFractal = 0,  /// number of cols inside the fractal
      Index colsByFractal = 0,  /// number of cols by the fractal
      LongIndex strideRowsInFractal =
          0,  /// number of elements between adjacent rows inside the fractal
      LongIndex strideRowsByFractal =
          0,  /// number of elements between adjacent fractal rows
      LongIndex strideColsInFractal =
          0,  /// number of elements between adjacent cols inside the fractal
      LongIndex strideColsByFractal =
          0)  /// number of elements between adjacent fractal cols
      : orgShape_(MakeCoord(orgRows, orgCols)),
        shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal,
                         colsByFractal)),
        stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal,
                          strideColsInFractal, strideColsByFractal)) {}

  template <class Element>
  CATLASS_HOST_DEVICE constexpr static zZ MakeLayout(Index orgRows,
                                                     Index orgCols) {
    constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);
    Index rowsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgRows);
    Index colsRound = RoundUp<ELE_NUM_PER_C0>(orgCols);
    return zZ(orgRows, orgCols, C0_NUM_PER_FRACTAL,
              rowsRound / C0_NUM_PER_FRACTAL, ELE_NUM_PER_C0,
              colsRound / ELE_NUM_PER_C0, ELE_NUM_PER_C0,
              colsRound * C0_NUM_PER_FRACTAL, 1, ELE_NUM_PER_FRACTAL);
  }

  CATLASS_HOST_DEVICE
  LongIndex GetOffset(MatrixCoord const& coord) const {
    return LongIndex(coord.row()) / shape_[0] * stride_[1] +
           LongIndex(coord.column()) / shape_[2] * stride_[3];
  }

  CATLASS_HOST_DEVICE
  typename Shape::Index shape(int idx) const { return shape_[idx]; }

  CATLASS_HOST_DEVICE
  typename Stride::Index stride(int idx) const { return stride_[idx]; }

 private:
  OrgShape orgShape_;

  Shape shape_;

  Stride stride_;
};

}  // namespace Catlass::layout

namespace Catlass::Gemm {

template <class Element_, class Layout_,
          AscendC::TPosition POSITION_ = AscendC::TPosition::GM>
struct GemmType {
  using Element = Element_;
  using Layout = Layout_;
  static constexpr AscendC::TPosition POSITION = POSITION_;
};

}  // namespace Catlass::Gemm

namespace Catlass::Gemm::Tile {

template <class ArchTag, class GmType, class L1Type = void>
struct CopyGmToL1 {
  static_assert(DEPENDENT_FALSE<ArchTag>,
                "Unsupported copy gm to l1, can not find the specialization.");
};

template <class Element>
struct CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>> {
  using LayoutDst = layout::zN;
  using LayoutSrc = layout::RowMajor;

  static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

  // Methods

  CATLASS_DEVICE
  CopyGmToL1() {};

  CATLASS_DEVICE
  void operator()(AscendC::LocalTensor<Element> const& dstTensor,
                  AscendC::GlobalTensor<Element> const& srcTensor,
                  LayoutDst const& layoutDst, LayoutSrc const& layoutSrc) {
    AscendC::Nd2NzParams intriParams;

    intriParams.ndNum = 1;
    intriParams.dValue = layoutSrc.shape(1);
    intriParams.srcNdMatrixStride = 0;
    intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
    intriParams.dstNzMatrixStride = 0;

    if (layoutSrc.stride(0) < STRIDE_LIMIT) {
      intriParams.nValue = layoutSrc.shape(0);
      intriParams.srcDValue = layoutSrc.stride(0);
      intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
      AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    } else {
      intriParams.nValue = 1;
      intriParams.srcDValue = 0;
      intriParams.dstNzNStride = 0;
      for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
        AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0],
                          srcTensor[i * layoutSrc.stride(0)], intriParams);
      }
    }
  }
};

template <class Element>
struct CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::ColumnMajor>> {
  using LayoutDst = layout::nZ;
  using LayoutSrc = layout::ColumnMajor;

  static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

  // Methods

  CATLASS_DEVICE
  CopyGmToL1() {};

  CATLASS_DEVICE
  void operator()(AscendC::LocalTensor<Element> const& dstTensor,
                  AscendC::GlobalTensor<Element> const& srcTensor,
                  LayoutDst const& layoutDst, LayoutSrc const& layoutSrc) {
    AscendC::Nd2NzParams intriParams;

    intriParams.ndNum = 1;
    intriParams.dValue = layoutSrc.shape(0);
    intriParams.srcNdMatrixStride = 0;
    intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
    intriParams.dstNzMatrixStride = 0;

    if (layoutSrc.stride(1) < STRIDE_LIMIT) {
      intriParams.nValue = layoutSrc.shape(1);
      intriParams.srcDValue = layoutSrc.stride(1);
      intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
      AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    } else {
      intriParams.nValue = 1;
      intriParams.srcDValue = 0;
      intriParams.dstNzNStride = 0;
      for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
        AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0],
                          srcTensor[i * layoutSrc.stride(1)], intriParams);
      }
    }
  }
};

enum class ScaleGranularity { NO_QUANT = 0 };

template <class ArchTag, class ElementSrc, class ElementDst,
          ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT>
struct CopyL0CToGmQuantMode {
  static_assert(DEPENDENT_FALSE<ArchTag>,
                "Unsupported copy l0c to gm, can not find the specialization.");
};

template <>
struct CopyL0CToGmQuantMode<Catlass::Arch::AtlasA2, float, bfloat16_t,
                            ScaleGranularity::NO_QUANT> {
  static constexpr auto VALUE = QuantMode_t::F322BF16;
};

template <class ArchTag, class ElementAccumulator, class GmType,
          ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
          bool ReluEnable = false>
struct CopyL0CToGm {
  static_assert(DEPENDENT_FALSE<ArchTag>,
                "Unsupported copy l0c to gm, can not find the specialization.");
};

template <class ElementAccumulator_, class ElementDst_, bool ReluEnable_>
struct CopyL0CToGm<Catlass::Arch::AtlasA2, ElementAccumulator_,
                   Gemm::GemmType<ElementDst_, layout::RowMajor>,
                   ScaleGranularity::NO_QUANT, ReluEnable_> {
  using ArchTag = Catlass::Arch::AtlasA2;
  using ElementDst = ElementDst_;
  using ElementSrc = ElementAccumulator_;
  using LayoutSrc = Catlass::layout::zN;
  using LayoutDst = Catlass::layout::RowMajor;
  static constexpr auto quantPre =
      CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst,
                           ScaleGranularity::NO_QUANT>::VALUE;
  static constexpr auto reluEn = ReluEnable_;

  struct Params {};
  Params params;

  CATLASS_DEVICE
  CopyL0CToGm() = default;

  CATLASS_DEVICE
  void operator()(AscendC::GlobalTensor<ElementDst> const& dst,
                  AscendC::LocalTensor<ElementSrc> const& src,
                  LayoutDst const& dstLayout, LayoutSrc const& srcLayout,
                  uint8_t unitFlag = 0) {
    AscendC::FixpipeParamsV220 intriParams;

    // Fixpipe layout information
    intriParams.nSize = dstLayout.shape(1);
    intriParams.mSize = dstLayout.shape(0);
    intriParams.srcStride = srcLayout.stride(3) / srcLayout.stride(0);
    intriParams.dstStride = dstLayout.stride(0);

    // Fixpipe auxiliary arguments
    intriParams.quantPre = quantPre;
    intriParams.reluEn = reluEn;
    intriParams.unitFlag = unitFlag;

    // Call AscendC Fixpipe
    AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(
        dst, src, intriParams);
  }
};

template <class ArchTag, class L1Type, class L0Type = void>
struct CopyL1ToL0A {
  static_assert(DEPENDENT_FALSE<ArchTag>,
                "Unsupported copy l1 to l0, can not find the specialization.");
};

template <class ArchTag, class Element>
struct CopyL1ToL0A<
    ArchTag, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {
  using LayoutDst = layout::zZ;
  using LayoutSrc = layout::zN;

  static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
  static constexpr uint32_t ELE_NUM_PER_FRACTAL =
      BYTE_PER_FRACTAL / sizeof(Element);

  // Methods

  CATLASS_DEVICE
  CopyL1ToL0A() {};

  CATLASS_DEVICE
  void operator()(AscendC::LocalTensor<Element> const& dstTensor,
                  AscendC::LocalTensor<Element> const& srcTensor,
                  LayoutDst const& layoutDst, LayoutSrc const& layoutSrc) {
    AscendC::LoadData2DParams loadDataParams;

    loadDataParams.startIndex = 0;
    loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
    loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
    loadDataParams.sid = 0;
    loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
    loadDataParams.ifTranspose = false;
    loadDataParams.addrMode = 0;

    for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
      AscendC::LoadData(dstTensor[i * layoutDst.stride(1)],
                        srcTensor[i * layoutSrc.stride(1)], loadDataParams);
    }
  }
};

template <class ArchTag, class L1Type, class L0Type = void>
struct CopyL1ToL0B {
  static_assert(DEPENDENT_FALSE<ArchTag>,
                "Unsupported copy l1 to l0, can not find the specialization.");
};

template <class ArchTag, class Element>
struct CopyL1ToL0B<
    ArchTag, Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>> {
  using LayoutDst = layout::nZ;
  using LayoutSrc = layout::nZ;

  static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
  static constexpr uint32_t ELE_NUM_PER_FRACTAL =
      BYTE_PER_FRACTAL / sizeof(Element);

  // Methods

  CATLASS_DEVICE
  CopyL1ToL0B() {};

  CATLASS_DEVICE
  void operator()(AscendC::LocalTensor<Element> const& dstTensor,
                  AscendC::LocalTensor<Element> const& srcTensor,
                  LayoutDst const& layoutDst, LayoutSrc const& layoutSrc) {
    AscendC::LoadData2DParams loadDataParams;
    if (layoutSrc.shape(3) == layoutDst.shape(3)) {
      loadDataParams.startIndex = 0;
      loadDataParams.repeatTimes =
          static_cast<uint16_t>(layoutDst.shape(1) * layoutDst.shape(3));
      loadDataParams.srcStride = 1;
      loadDataParams.sid = 0;
      loadDataParams.dstGap = 0;
      loadDataParams.ifTranspose = false;
      loadDataParams.addrMode = 0;

      AscendC::LoadData(dstTensor, srcTensor, loadDataParams);
    } else {
      loadDataParams.startIndex = 0;
      loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
      loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
      loadDataParams.sid = 0;
      loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
      loadDataParams.ifTranspose = false;
      loadDataParams.addrMode = 0;

      for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
        AscendC::LoadData(dstTensor[i * layoutDst.stride(1)],
                          srcTensor[i * layoutSrc.stride(1)], loadDataParams);
      }
    }
  }
};

}  // namespace Catlass::Gemm::Tile

namespace Catlass::Gemm::helper {

template <class Element, class Layout>
struct L1AlignHelper {
  static_assert(DEPENDENT_FALSE<Element>,
                "Unsupported align helper, can not find the specialization.");
};

template <class Element>
struct L1AlignHelper<Element, layout::RowMajor> {
  static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
  static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
  static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
  static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

template <class Element>
struct L1AlignHelper<Element, layout::ColumnMajor> {
  static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
  static constexpr uint32_t M_ALIGNED = ELE_NUM_PER_C0;
  static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
  static constexpr uint32_t N_ALIGNED = C0_NUM_PER_FRACTAL;
};

template <class GmAType>
struct L1ATypeSelector {
  static_assert(
      DEPENDENT_FALSE<GmAType>,
      "Unsupported layout selector, can not find the specialization.");
};

template <class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::RowMajor>> {
  using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template <class GmBType>
struct L1BTypeSelector {
  static_assert(
      DEPENDENT_FALSE<GmBType>,
      "Unsupported layout selector, can not find the specialization.");
};

template <class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::ColumnMajor>> {
  using L1BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;
};

}  // namespace Catlass::Gemm::helper

namespace Catlass::Gemm::Tile {

// Both FFN matmuls use BF16 ND inputs/weights and FP32 accumulation.
// Only the fused front overrides the weight reader and ring-buffer store.
struct FfnTileCopy {
  using ArchTag = Arch::AtlasA2;
  using ElementA = bfloat16_t;
  using ElementB = bfloat16_t;
  using ElementAccumulator = float;
  using AType = Gemm::GemmType<ElementA, layout::RowMajor>;
  using BType = Gemm::GemmType<ElementB, layout::ColumnMajor>;
  using CType = Gemm::GemmType<bfloat16_t, layout::RowMajor>;

  using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, AType>;
  using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, BType>;
  using CopyL1ToL0A =
      Gemm::Tile::CopyL1ToL0A<ArchTag,
                              typename helper::L1ATypeSelector<AType>::L1AType>;
  using CopyL1ToL0B =
      Gemm::Tile::CopyL1ToL0B<ArchTag,
                              typename helper::L1BTypeSelector<BType>::L1BType>;
  using CopyL0CToGm =
      Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
};

struct TileMmad {
  using ElementA = bfloat16_t;
  using ElementB = bfloat16_t;
  using ElementAccumulator = float;

  // Methods

  CATLASS_DEVICE
  TileMmad() {}

  CATLASS_DEVICE
  void operator()(AscendC::LocalTensor<ElementAccumulator> const& l0CTensor,
                  AscendC::LocalTensor<ElementA> const& l0ATensor,
                  AscendC::LocalTensor<ElementB> const& l0BTensor, uint32_t m,
                  uint32_t n, uint32_t k, bool initC = true,
                  uint8_t unitFlag = 0) {
    AscendC::MmadParams mmadParams;
    mmadParams.m = m;
    mmadParams.n = n;
    mmadParams.k = k;
    mmadParams.unitFlag = unitFlag;
    mmadParams.cmatrixInitVal = initC;

    AscendC::Mmad(l0CTensor, l0ATensor, l0BTensor, mmadParams);

    const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
    if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) <
        PIPE_M_BARRIER_THRESHOLD) {
      AscendC::PipeBarrier<PIPE_M>();
    }
  }
};

}  // namespace Catlass::Gemm::Tile

namespace Catlass::Arch {

struct LocalTensorBufferBase {
 public:
  template <class Element = half>
  CATLASS_DEVICE AscendC::LocalTensor<Element> GetBufferByByte(
      const uint32_t offset) const {
    return tensor[offset].template ReinterpretCast<Element>();
  }

 protected:
  CATLASS_DEVICE
  LocalTensorBufferBase() = default;

  AscendC::LocalTensor<uint8_t> tensor;
};

template <class ArchTag, AscendC::TPosition Position>
struct LocalTensorBuffer {
  static_assert(
      DEPENDENT_FALSE<ArchTag>,
      "Unsupported local tensor buffer, can not find the specialization.");
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::A1>
    : LocalTensorBufferBase {
 public:
  static constexpr AscendC::TPosition Position = AscendC::TPosition::A1;

  CATLASS_DEVICE
  LocalTensorBuffer() {
    AscendC::TBuf<AscendC::TPosition::A1> tbufA1;
    GetTPipePtr()->InitBuffer(tbufA1, ArchTag::L1_SIZE);
    tensor = tbufA1.Get<uint8_t>();
  }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::A2>
    : LocalTensorBufferBase {
 public:
  static constexpr AscendC::TPosition Position = AscendC::TPosition::A2;

  CATLASS_DEVICE
  LocalTensorBuffer() {
    AscendC::TBuf<AscendC::TPosition::A2> tbufA2;
    GetTPipePtr()->InitBuffer(tbufA2, ArchTag::L0A_SIZE);
    tensor = tbufA2.Get<uint8_t>();
  }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::B2>
    : LocalTensorBufferBase {
 public:
  static constexpr AscendC::TPosition Position = AscendC::TPosition::B2;

  CATLASS_DEVICE
  LocalTensorBuffer() {
    AscendC::TBuf<AscendC::TPosition::B2> tbufB2;
    GetTPipePtr()->InitBuffer(tbufB2, ArchTag::L0B_SIZE);
    tensor = tbufB2.Get<uint8_t>();
  }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::CO1>
    : LocalTensorBufferBase {
 public:
  static constexpr AscendC::TPosition Position = AscendC::TPosition::CO1;

  CATLASS_DEVICE
  LocalTensorBuffer() {
    AscendC::TBuf<AscendC::TPosition::CO1> tbufCO1;
    GetTPipePtr()->InitBuffer(tbufCO1, ArchTag::L0C_SIZE);
    tensor = tbufCO1.Get<uint8_t>();
  }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::VECCALC>
    : LocalTensorBufferBase {
 public:
  static constexpr AscendC::TPosition Position = AscendC::TPosition::VECCALC;

  CATLASS_DEVICE
  LocalTensorBuffer() {
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbufVECCALC;
    GetTPipePtr()->InitBuffer(tbufVECCALC, ArchTag::UB_SIZE);
    tensor = tbufVECCALC.Get<uint8_t>();
  }
};

template <class ArchTag>
struct Resource {
 public:
  AscendC::TPipe pipe;

  LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> l1Buf;
  LocalTensorBuffer<ArchTag, AscendC::TPosition::A2> l0ABuf;
  LocalTensorBuffer<ArchTag, AscendC::TPosition::B2> l0BBuf;
  LocalTensorBuffer<ArchTag, AscendC::TPosition::CO1> l0CBuf;
  LocalTensorBuffer<ArchTag, AscendC::TPosition::VECCALC> ubBuf;

  CATLASS_DEVICE
  Resource() {
    // The initialization of AscendC::Tpipe will insert some synchronization
    // interfaces, which may conflict with the usage by users. Therefore, the
    // "destroy" interface is used for releasing.
    pipe.Destroy();
  }
};

}  // namespace Catlass::Arch

namespace Catlass {

template <uint32_t M_ = 1, uint32_t N_ = 1, uint32_t K_ = 1>
struct GemmShape {
  static constexpr uint32_t M = M_;
  static constexpr uint32_t N = N_;
  static constexpr uint32_t K = K_;
};

struct GemmCoord : public Coord<3, uint32_t> {
  using Index = uint32_t;

  using Base = Coord<3, Index>;

  static constexpr int M_INDEX = 0;

  static constexpr int N_INDEX = 1;

  static constexpr int K_INDEX = 2;

  CATLASS_HOST_DEVICE
  GemmCoord() {}

  CATLASS_HOST_DEVICE
  GemmCoord(Coord<3, Index> const& coord) : Base(coord) {}

  CATLASS_HOST_DEVICE
  GemmCoord(Index m, Index n, Index k) : Base(MakeCoord(m, n, k)) {}

  CATLASS_HOST_DEVICE
  Index const& m() const { return this->At(M_INDEX); }

  CATLASS_HOST_DEVICE
  Index& m() { return this->At(M_INDEX); }

  CATLASS_HOST_DEVICE
  Index const& n() const { return this->At(N_INDEX); }

  CATLASS_HOST_DEVICE
  Index& n() { return this->At(N_INDEX); }

  CATLASS_HOST_DEVICE
  Index const& k() const { return this->At(K_INDEX); }

  CATLASS_HOST_DEVICE
  Index& k() { return this->At(K_INDEX); }

  CATLASS_HOST_DEVICE
  auto GetCoordMN() const { return this->GetCoordByAxis<M_INDEX, N_INDEX>(); }
};

}  // namespace Catlass

namespace Catlass::Gemm::Block {

// Shared A2 Cube pipeline: two L1/L0 stages, natural K order, unit-flag
// MMAD/Fixpipe handoff. Shapes and the two FFN copy policies stay explicit.
template <class L1TileShape_, class L0TileShape_,
          class TileCopy_ = Gemm::Tile::FfnTileCopy>
struct BlockMmad {
 public:
  using ArchTag = Arch::AtlasA2;
  using L1TileShape = L1TileShape_;
  using L0TileShape = L0TileShape_;
  using ElementA = bfloat16_t;
  using LayoutA = layout::RowMajor;
  using ElementB = bfloat16_t;
  using LayoutB = layout::ColumnMajor;
  using ElementC = bfloat16_t;
  using LayoutC = layout::RowMajor;
  using TileMmad = Gemm::Tile::TileMmad;
  using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
  using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
  using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
  using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
  using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
  using ElementAccumulator = float;
  using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
  using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
  using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
  using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
  using LayoutCInL0 = layout::zN;

  using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
  using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

  static constexpr uint32_t STAGES = 2;
  static constexpr uint32_t L1A_SIZE =
      L1TileShape::M * L1TileShape::K * sizeof(ElementA);
  static constexpr uint32_t L1B_SIZE =
      L1TileShape::N * L1TileShape::K * sizeof(ElementB);
  static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
  static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
  static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
  static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
  static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;

  // Check LayoutC
  static_assert(std::is_same_v<LayoutC, layout::RowMajor>,
                "LayoutC only support RowMajor yet!");

  // Check L1TileShape
  static_assert((L1A_SIZE * STAGES + L1B_SIZE * STAGES) <= ArchTag::L1_SIZE,
                "L1TileShape exceeding the L1 space!");

  // Check L0TileShape
  static constexpr uint32_t L0A_TILE_SIZE =
      L0TileShape::M * L0TileShape::K * sizeof(ElementA);
  static constexpr uint32_t L0B_TILE_SIZE =
      L0TileShape::K * L0TileShape::N * sizeof(ElementB);
  static constexpr uint32_t L0C_TILE_SIZE =
      L0TileShape::M * L0TileShape::N * sizeof(ElementAccumulator);
  static_assert((L0A_TILE_SIZE * STAGES) <= L0A_SIZE,
                "L0TileShape exceeding the L0A space!");
  static_assert((L0B_TILE_SIZE * STAGES) <= L0B_SIZE,
                "L0TileShape exceeding the L0B space!");
  static_assert(L0C_TILE_SIZE <= L0C_SIZE,
                "L0TileShape exceeding the L0C space!");

  static_assert(L1TileShape::M == L0TileShape::M &&
                    L1TileShape::N == L0TileShape::N,
                "The situation where the basic blocks of L1 and L0 differ on "
                "the m and n axes is not supported yet");
  static_assert(L0TileShape::K <= L1TileShape::K,
                "L0TileShape::K cannot exceed L1TileShape::K");

  CATLASS_DEVICE
  BlockMmad(Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart = 0) {
    uint32_t l1AOffset = l1BufAddrStart;
    uint32_t l1BOffset = l1BufAddrStart + L1A_SIZE * STAGES;
    // Init buffers
    for (uint32_t i = 0; i < STAGES; i++) {
      l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(
          l1AOffset + L1A_SIZE * i);
      l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(
          l1BOffset + L1B_SIZE * i);
      l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(
          L0A_PINGPONG_BUF_SIZE * i);
      l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(
          L0B_PINGPONG_BUF_SIZE * i);

      l1AEventList[i] = i;
      l1BEventList[i] = i + STAGES;
      l0AEventList[i] = i;
      l0BEventList[i] = i + STAGES;
      AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
      AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
      AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
      AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
    }
    l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
  }

  CATLASS_DEVICE
  ~BlockMmad() {
    for (uint32_t i = 0; i < STAGES; i++) {
      AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
      AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
      AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
      AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
    }
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
  }

  CATLASS_DEVICE
  void operator()(
      AscendC::GlobalTensor<ElementA> const& gmBlockA, LayoutA const& layoutA,
      AscendC::GlobalTensor<ElementB> const& gmBlockB, LayoutB const& layoutB,
      AscendC::GlobalTensor<ElementC> const& gmBlockC, LayoutC const& layoutC,
      AscendC::GlobalTensor<ElementA> const& gmNextBlockA,
      AscendC::GlobalTensor<ElementB> const& gmNextBlockB,
      GemmCoord const& actualShape, GemmCoord const& actualShapeNext,
      bool isFirstBlock, bool hasNextBlock) {
    uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
    uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());

    auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(
        L1TileShape::M, L1TileShape::K);
    auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(
        L1TileShape::K, L1TileShape::N);
    auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mRound, nRound));

    uint32_t kTileCount = CeilDiv<L1TileShape::K>(actualShape.k());
    uint32_t kTileCountNext = CeilDiv<L1TileShape::K>(actualShapeNext.k());

    uint32_t startTileIdx = 0;

    uint32_t firstTileIdx = startTileIdx % kTileCount;
    uint32_t lastTileIdx = (startTileIdx + kTileCount - 1) % kTileCount;
    uint32_t kActual = (firstTileIdx < kTileCount - 1)
                           ? L1TileShape::K
                           : (actualShape.k() - firstTileIdx * L1TileShape::K);
    uint32_t firstTileIdxNext = startTileIdx % kTileCountNext;

    uint32_t mPartLoop = CeilDiv<L0TileShape::M>(mRound);
    uint32_t nPartLoop = CeilDiv<L0TileShape::N>(nRound);

    // k loop
    for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
      uint32_t shuffleKIdx = (startTileIdx + kLoopIdx) % kTileCount;
      // Load first matrix A tile in total kernel loop from GM to L1
      if (shuffleKIdx == firstTileIdx && isFirstBlock) {
        MatrixCoord gmTileAOffset{0, shuffleKIdx * L1TileShape::K};
        MatrixCoord gmTileBOffset{shuffleKIdx * L1TileShape::K, 0};
        auto gmTileA = gmBlockA[layoutA.GetOffset(gmTileAOffset)];
        auto gmTileB = gmBlockB[layoutB.GetOffset(gmTileBOffset)];
        // Load first matrix A tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
            l1AEventList[l1ListId]);
        auto layoutTileA =
            layoutA.GetTileLayout(MakeCoord(actualShape.m(), kActual));
        copyGmToL1A(l1ATensorList[l1ListId], gmTileA, layoutAInL1, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);

        // Load first matrix B tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
            l1BEventList[l1ListId]);
        auto layoutTileB =
            layoutB.GetTileLayout(MakeCoord(kActual, actualShape.n()));
        copyGmToL1B(l1BTensorList[l1ListId], gmTileB, layoutBInL1, layoutTileB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
      }

      uint32_t l1ListIdNext = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;
      uint32_t kActualNext{0};

      // preload next tile from GM to L1
      if (shuffleKIdx != lastTileIdx) {
        uint32_t shuffleKIdxNext = (startTileIdx + kLoopIdx + 1) % kTileCount;
        // Get L1 tensor for next stage
        auto l1ATensor = l1ATensorList[l1ListIdNext];
        auto l1BTensor = l1BTensorList[l1ListIdNext];
        // Get GM tensor for next stage
        kActualNext =
            (shuffleKIdxNext < kTileCount - 1)
                ? L1TileShape::K
                : (actualShape.k() - shuffleKIdxNext * L1TileShape::K);
        MatrixCoord gmTileAOffset{0, shuffleKIdxNext * L1TileShape::K};
        MatrixCoord gmTileBOffset{shuffleKIdxNext * L1TileShape::K, 0};
        auto gmTileA = gmBlockA[layoutA.GetOffset(gmTileAOffset)];
        auto gmTileB = gmBlockB[layoutB.GetOffset(gmTileBOffset)];

        // load next matrix A tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
            l1AEventList[l1ListIdNext]);
        auto layoutTileA =
            layoutA.GetTileLayout(MakeCoord(actualShape.m(), kActualNext));
        copyGmToL1A(l1ATensor, gmTileA, layoutAInL1, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(
            l1AEventList[l1ListIdNext]);

        // load next matrix B tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
            l1BEventList[l1ListIdNext]);
        auto layoutTileB =
            layoutB.GetTileLayout(MakeCoord(kActualNext, actualShape.n()));
        copyGmToL1B(l1BTensor, gmTileB, layoutBInL1, layoutTileB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(
            l1BEventList[l1ListIdNext]);
      }
      if (shuffleKIdx == lastTileIdx && hasNextBlock) {
        // Get L1 tensor for next stage
        auto l1ATensor = l1ATensorList[l1ListIdNext];
        auto l1BTensor = l1BTensorList[l1ListIdNext];
        // Get GM tensor for next stage
        kActualNext =
            (firstTileIdxNext < kTileCountNext - 1)
                ? L1TileShape::K
                : (actualShapeNext.k() - firstTileIdxNext * L1TileShape::K);
        MatrixCoord gmTileAOffset{0, firstTileIdxNext * L1TileShape::K};
        MatrixCoord gmTileBOffset{firstTileIdxNext * L1TileShape::K, 0};
        auto gmTileA = gmNextBlockA[layoutA.GetOffset(gmTileAOffset)];
        auto gmTileB = gmNextBlockB[layoutB.GetOffset(gmTileBOffset)];
        // load next matrix A tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
            l1AEventList[l1ListIdNext]);
        auto layoutTileA =
            layoutA.GetTileLayout(MakeCoord(actualShapeNext.m(), kActualNext));
        copyGmToL1A(l1ATensor, gmTileA, layoutAInL1, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(
            l1AEventList[l1ListIdNext]);

        // load next matrix B tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
            l1BEventList[l1ListIdNext]);
        auto layoutTileB =
            layoutB.GetTileLayout(MakeCoord(kActualNext, actualShapeNext.n()));
        copyGmToL1B(l1BTensor, gmTileB, layoutBInL1, layoutTileB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(
            l1BEventList[l1ListIdNext]);
      }

      // Get L1 tensor for current stage
      auto l1ATensor = l1ATensorList[l1ListId];
      auto l1BTensor = l1BTensorList[l1ListId];

      // Get the loop nums on L0
      uint32_t kPartLoop = CeilDiv<L0TileShape::K>(kActual);

      uint32_t l0ABufId = 0;
      uint32_t l0BBufId = 0;

      for (int mPartIdx = 0; mPartIdx < mPartLoop; mPartIdx++) {
        uint32_t mPartActual = (mPartIdx < mPartLoop - 1)
                                   ? L0TileShape::M
                                   : (mRound - mPartIdx * L0TileShape::M);

        for (int kPartIdx = 0; kPartIdx < kPartLoop; kPartIdx++) {
          uint32_t kPartActual = (kPartIdx < kPartLoop - 1)
                                     ? L0TileShape::K
                                     : (kActual - kPartIdx * L0TileShape::K);

          // Locate the current tile on L0A
          auto l0ATile = l0ATensorList[l0ABufId];
          LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(
              mPartActual, kPartActual);
          // Locate the current tile of matrix A on L1
          MatrixCoord l1AOffset{mPartIdx * L0TileShape::M,
                                kPartIdx * L0TileShape::K};
          auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1AOffset)];

          AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ABufId]);
          if ((mPartIdx == 0) && (kPartIdx == 0)) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(
                l1AEventList[l1ListId]);
          }

          copyL1ToL0A(l0ATile, l1ATile, layoutAInL0, layoutAInL1);

          if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1)) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
                l1AEventList[l1ListId]);
          }

          for (int nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
            uint32_t nPartActual = (nPartIdx < nPartLoop - 1)
                                       ? L0TileShape::N
                                       : (nRound - nPartIdx * L0TileShape::N);

            // Locate the current tile on L0B
            auto l0BTile = l0BTensorList[l0BBufId];
            LayoutBInL0 layoutBInL0 =
                LayoutBInL0::template MakeLayout<ElementB>(kPartActual,
                                                           nPartActual);
            // Locate the current tile of matrix B on L1
            MatrixCoord l1BOffset{kPartIdx * L0TileShape::K,
                                  nPartIdx * L0TileShape::N};
            auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BOffset)];

            // Wait for mmad finished
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(
                l0BEventList[l0BBufId]);
            // If the current tile is the first one on the k&n axis, wait for
            // loading matrix B from GM to L1
            if ((kPartIdx == 0) && (nPartIdx == 0)) {
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(
                  l1BEventList[l1ListId]);
            }

            // Load current tile from L1 to L0B
            copyL1ToL0B(l0BTile, l1BTile, layoutBInL0, layoutBInL1);

            // If the current tile is the last one on the k&n axis, notify to
            // load matrix B from GM to L1
            if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
              AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
                  l1BEventList[l1ListId]);
            }
            // Notify to do mmad
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

            // Locate the current tile on L0C
            MatrixCoord l0COffset{mPartIdx * L0TileShape::M,
                                  nPartIdx * L0TileShape::N};
            auto l0CTile = l0CTensor[layoutInL0C.GetOffset(l0COffset)];

            // Compute the matrix multiplication on L0A and L0B and write the
            // result to the accumulator Wait for loading L0B
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

            // If the current tile is the first tile on the k axis, the
            // accumulator needs to be reset to 0
            bool initC = ((kLoopIdx == 0) && (kPartIdx == 0));
            // If the unit flag is enabled, the unit flag is set according to
            // the calculation progress
            uint8_t unitFlag = 0b00;
            if ((kLoopIdx == kTileCount - 1) && (mPartIdx == mPartLoop - 1) &&
                (kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
              unitFlag = 0b11;
            } else {
              unitFlag = 0b10;
            }
            // Perform calculation operations
            tileMmad(l0CTile, l0ATile, l0BTile, mPartActual, nPartActual,
                     kPartActual, initC, unitFlag);

            // Notify to move the next L0B tile
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(
                l0BEventList[l0BBufId]);

            l0BBufId = (l0BBufId + 1 < STAGES) ? (l0BBufId + 1) : 0;
          }
          AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ABufId]);
          l0ABufId = (l0ABufId + 1 < STAGES) ? (l0ABufId + 1) : 0;
        }
      }

      l1ListId = l1ListIdNext;
      kActual = kActualNext;
    }

    // copy block out
    LayoutC layoutBlock = layoutC.GetTileLayout(actualShape.GetCoordMN());

    copyL0CToGm(gmBlockC, l0CTensor, layoutBlock, layoutInL0C, 0b11);
  }

 protected:
  AscendC::LocalTensor<ElementA> l1ATensorList[STAGES];
  AscendC::LocalTensor<ElementB> l1BTensorList[STAGES];
  AscendC::LocalTensor<ElementA> l0ATensorList[STAGES];
  AscendC::LocalTensor<ElementB> l0BTensorList[STAGES];
  AscendC::LocalTensor<ElementAccumulator> l0CTensor;

  int32_t l1AEventList[STAGES];
  int32_t l1BEventList[STAGES];
  int32_t l0AEventList[STAGES];
  int32_t l0BEventList[STAGES];

  uint32_t l1ListId{0};

  TileMmad tileMmad;
  CopyGmToL1A copyGmToL1A;
  CopyGmToL1B copyGmToL1B;
  CopyL1ToL0A copyL1ToL0A;
  CopyL1ToL0B copyL1ToL0B;
  CopyL0CToGm copyL0CToGm;
};

template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct GemmIdentityBlockSwizzle {
  GemmCoord problemShape;
  MatrixCoord tileMN;
  MatrixCoord loopsMN;

  CATLASS_DEVICE
  GemmIdentityBlockSwizzle(GemmCoord const& problemShape_,
                           MatrixCoord const& tileMN_)
      : problemShape(problemShape_), tileMN(tileMN_) {
    loopsMN = CeilDiv(MatrixCoord(problemShape.GetCoordMN()), tileMN);
  }

  CATLASS_DEVICE
  uint32_t GetCoreLoops() const { return loopsMN.row() * loopsMN.column(); }

  CATLASS_DEVICE
  GemmCoord GetBlockCoord(uint32_t taskIdx) {
    uint32_t innerIdx = taskIdx % GetCoreLoops();
    if constexpr (SwizzleDirection == 0) {  // Zn
      uint32_t tileBlockLoop = CeilDiv(loopsMN.row(), SwizzleOffset);
      uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMN.column());
      uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMN.column());

      uint32_t nRow = SwizzleOffset;
      if (tileBlockIdx == tileBlockLoop - 1) {
        nRow = loopsMN.row() - SwizzleOffset * tileBlockIdx;
      }
      uint32_t mIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nRow;
      uint32_t nIdx = inTileBlockIdx / nRow;
      if (tileBlockIdx % 2 == 1) {
        nIdx = loopsMN.column() - nIdx - 1;
      }
      return GemmCoord{mIdx, nIdx, 0};
    } else if constexpr (SwizzleDirection == 1) {  // Nz
      uint32_t tileBlockLoop = CeilDiv(loopsMN.column(), SwizzleOffset);
      uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMN.row());
      uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMN.row());

      uint32_t nCol = SwizzleOffset;
      if (tileBlockIdx == tileBlockLoop - 1) {
        nCol = loopsMN.column() - SwizzleOffset * tileBlockIdx;
      }
      uint32_t mIdx = inTileBlockIdx / nCol;
      uint32_t nIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCol;
      if (tileBlockIdx % 2 == 1) {
        mIdx = loopsMN.row() - mIdx - 1;
      }
      return GemmCoord{mIdx, nIdx, 0};
    }
  }

  CATLASS_DEVICE
  GemmCoord GetActualBlockShape(GemmCoord blockCoord) {
    uint32_t mActual = (blockCoord.m() == (loopsMN.row() - 1))
                           ? (problemShape.m() - blockCoord.m() * tileMN.row())
                           : tileMN.row();
    uint32_t nActual =
        (blockCoord.n() == (loopsMN.column() - 1))
            ? (problemShape.n() - blockCoord.n() * tileMN.column())
            : tileMN.column();
    uint32_t kActual = problemShape.k();
    return GemmCoord{mActual, nActual, kActual};
  }
};

}  // namespace Catlass::Gemm::Block
