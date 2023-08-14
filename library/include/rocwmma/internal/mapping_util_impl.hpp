/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef ROCWMMA_MAPPING_UTIL_IMPL_HPP
#define ROCWMMA_MAPPING_UTIL_IMPL_HPP

#if !defined(__HIPCC_RTC__)
#include <hip/hip_runtime.h>
#endif

#include "constants.hpp"
#include "mapping_util.hpp"
#include "types.hpp"
#include "utils.hpp"

namespace rocwmma
{

    namespace detail
    {
        // Workgroup configuration
        // Assumption: major thread dimension is X
        // Notation (x, y) = (row, col)
        ROCWMMA_DEVICE inline uint32_t laneId(sycl::id<1>& id)
        {
            // threadIdx.x % AMDGCN_WAVE_SIZE;
            return id[0] & (Constants::AMDGCN_WAVE_SIZE - 1u);
        }

        ROCWMMA_DEVICE constexpr inline Coord2d waveCount(Coord2d const& threadCount)
        {
            // waveCount.x = threadCount.x / AMDGCN_WAVE_SIZE
            // waveCount.y = threadCount.y
            return make_coord2d(get<0>(threadCount) >> Log2<Constants::AMDGCN_WAVE_SIZE>::value,
                                get<1>(threadCount));
        }

        ROCWMMA_DEVICE constexpr inline Coord2d threadCount(Coord2d const& waveCount)
        {
            // threadCount.x = waveCount.x * AMDGCN_WAVE_SIZE
            // threadCount.y = waveCount.y
            return make_coord2d(get<0>(waveCount) << Log2<Constants::AMDGCN_WAVE_SIZE>::value,
                                get<1>(waveCount));
        }

        /// WaveSpace

        template <uint32_t TBlockX, uint32_t TBlockY>
        ROCWMMA_DEVICE inline uint32_t WaveSpace<TBlockX, TBlockY>::localLaneId(sycl::id<1>& id)
        {
            return laneId(id);
        }

        template <uint32_t TBlockX, uint32_t TBlockY>
        ROCWMMA_DEVICE constexpr inline auto WaveSpace<TBlockX, TBlockY>::localWaveCoord(sycl::id<2>& id)
            -> WaveCoordT
        {
            return waveCount(make_coord2d(static_cast<uint32_t>(id[0]),
                                          static_cast<uint32_t>(id[1])));
        }

        template <uint32_t TBlockX, uint32_t TBlockY>
        ROCWMMA_DEVICE inline auto WaveSpace<TBlockX, TBlockY>::globalWaveCoord(sycl::sub_group& sub_group) -> WaveCoordT
        {
            auto BlockDim = sub_group.get_local_range();
            auto BlockId = sub_group.get_group_id();
            auto id = sub_group.get_local_id();
            return waveCount(make_coord2d(BlockId[0] * TBlockX + id[0],
                                          BlockId[1] * TBlockY + id[1]));
        }

        template <>
        ROCWMMA_DEVICE inline auto WaveSpace<0, 0>::globalWaveCoord(sycl::sub_group& sub_group) -> WaveCoordT
        {
            auto BlockDim = sub_group.get_local_range();
            auto BlockId = sub_group.get_group_id();
            auto id = sub_group.get_local_id();
            return waveCount(make_coord2d(BlockId[0] * BlockDim[0] + id[0],
                                          BlockId[1] * BlockDim[1] + id[1]));
        }

        // template <uint32_t TBlockX, uint32_t TBlockY>
        // ROCWMMA_DEVICE constexpr inline auto WaveSpace<TBlockX, TBlockY>::workgroupCoord(sycl::sub_group& sub_group)
        //     -> WorkgroupCoordT
        // {
        //     auto BlockId = sub_group.get_group_id();
        //     return make_coord2d(static_cast<uint32_t>(BlockId[0]),
        //                         static_cast<uint32_t>(BlockId[1]));
        // }

        template <uint32_t TBlockX, uint32_t TBlockY>
        template <bool IsConst /* = (TBlockX > 0u && TBlockY > 0u) */,
                  typename std::enable_if_t<IsConst>* /* = nullptr */>
        ROCWMMA_DEVICE constexpr inline auto WaveSpace<TBlockX, TBlockY>::workgroupDim(sycl::sub_group& sub_group)
            -> WorkgroupDimT
        {
            return waveCount(make_coord2d(TBlockX, TBlockY));
        }

        template <uint32_t TBlockX, uint32_t TBlockY>
        template <bool IsConst /* = (TBlockX > 0u && TBlockY > 0u) */,
                  typename std::enable_if_t<!IsConst>* /* = nullptr */>
        ROCWMMA_DEVICE inline auto WaveSpace<TBlockX, TBlockY>::workgroupDim(sycl::sub_group& sub_group) -> WorkgroupDimT
        {
            auto BlockDim = sub_group.get_local_range();
            return waveCount(make_coord2d(BlockDim[0], BlockDim[1]));
        }

        /// MatrixSpace
        template <uint32_t BlockHeight, uint32_t BlockWidth>
        ROCWMMA_DEVICE inline auto
            MatrixSpace<BlockHeight, BlockWidth>::fromBlockCoord(BlockCoordT const& blockCoord)
                -> MatrixCoordT
        {
            // Map block to matrix coordinate space.
            return make_coord2d(get<0>(blockCoord) * BlockHeight, // ROW
                                get<1>(blockCoord) * BlockWidth); // COL
        }

        /// DataSpace
        template <sycl::ext::oneapi::experimental::matrix::layout DataLayout>
        ROCWMMA_DEVICE constexpr inline auto
            DataSpace<DataLayout>::leadingDim(MatrixSizeT const& matrixSize)
        {
            return get<MinorIndex>(matrixSize);
        }

        template <sycl::ext::oneapi::experimental::matrix::layout DataLayout>
        ROCWMMA_DEVICE constexpr inline auto
            DataSpace<DataLayout>::fromMatrixCoord(MatrixCoordT const& matrixCoord,
                                                        uint32_t            leadingDim)
        {
            // 1D data element offset transform
            return get<MajorIndex>(matrixCoord) * leadingDim + get<MinorIndex>(matrixCoord);
        }

    } // namespace detail

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT,
        sycl::ext::oneapi::experimental::matrix::layout DataLayout>
    ROCWMMA_DEVICE inline uint32_t MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::laneId(sycl::id<1>& id)
    {
        return WaveSpace::localLaneId(id);
    }

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT,
        sycl::ext::oneapi::experimental::matrix::layout DataLayout>
    ROCWMMA_DEVICE inline auto MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::waveCoord(sycl::id<2>& id)
        -> WaveCoordT
    {
        return WaveSpace::localWaveCoord(id);
    }

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT,
        sycl::ext::oneapi::experimental::matrix::layout DataLayout>
    ROCWMMA_DEVICE inline auto MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::blockCoord(sycl::sub_group& sub_group)
        -> BlockCoordT
    {
        // Map each wave 1 : 1 to global block grid
        return WaveSpace::globalWaveCoord(sub_group);
    }

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT,
        sycl::ext::oneapi::experimental::matrix::layout DataLayout>
    ROCWMMA_DEVICE inline auto
        MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::matrixCoord(sycl::id<2>& id) -> MatrixCoordT
    {
        return MatrixSpace::fromBlockCoord(blockCoord(id));
    }

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT,
        sycl::ext::oneapi::experimental::matrix::layout DataLayout>
    ROCWMMA_DEVICE inline DataT const*
        MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::dataCoord(DataT const* baseAddr,
                                                                           uint32_t     ldm)
    {
        return baseAddr + DataSpace::fromMatrixCoord(matrixCoord(), ldm);
    }

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT,
        sycl::ext::oneapi::experimental::matrix::layout DataLayout>
    ROCWMMA_DEVICE inline DataT*
        MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::dataCoord(DataT*   baseAddr,
                                                                           uint32_t ldm)
    {
        return baseAddr + DataSpace::fromMatrixCoord(matrixCoord(), ldm);
    }

    /// Current workgroup perspective

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT,
        sycl::ext::oneapi::experimental::matrix::layout DataLayout>
    ROCWMMA_DEVICE inline auto
        MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::workgroupDim(sycl::sub_group& sub_group) -> WorkgroupDimT
    {
        return WaveSpace::workgroupDim(sub_group);
    }

    /// Coordinate override helpers

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT,
        sycl::ext::oneapi::experimental::matrix::layout DataLayout>
    ROCWMMA_DEVICE inline auto
        MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::blockCoordM(uint32_t m)
            -> BlockCoordT
    {
        auto coord    = blockCoord();
        get<0>(coord) = m;
        return coord;
    }

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT,
        sycl::ext::oneapi::experimental::matrix::layout DataLayout>
    ROCWMMA_DEVICE inline auto
        MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::blockCoordN(uint32_t n)
            -> BlockCoordT
    {
        auto coord    = blockCoord();
        get<1>(coord) = n;
        return coord;
    }

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT,
        sycl::ext::oneapi::experimental::matrix::layout DataLayout>
    ROCWMMA_DEVICE inline auto
        MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::matrixCoordM(uint32_t m)
            -> MatrixCoordT
    {
        auto coord    = matrixCoord();
        get<0>(coord) = m;
        return coord;
    }

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT,
        sycl::ext::oneapi::experimental::matrix::layout DataLayout>
    ROCWMMA_DEVICE inline auto
        MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::matrixCoordN(uint32_t n)
            -> MatrixCoordT
    {
        auto coord    = matrixCoord();
        get<1>(coord) = n;
        return coord;
    }

    /// Conversion helpers

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT,
        sycl::ext::oneapi::experimental::matrix::layout DataLayout>
    ROCWMMA_DEVICE inline auto MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::matrixCoord(
        BlockCoordT const& blockCoord) -> MatrixCoordT
    {
        return MatrixSpace::fromBlockCoord(std::forward<BlockCoordT const>(blockCoord));
    }

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT,
        sycl::ext::oneapi::experimental::matrix::layout DataLayout>
    ROCWMMA_DEVICE inline uint32_t
        MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::dataOffset(
            MatrixCoordT const& matrixCoord, uint32_t ldm)
    {
        return DataSpace::fromMatrixCoord(std::forward<MatrixCoordT const>(matrixCoord), ldm);
    }

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT,
        sycl::ext::oneapi::experimental::matrix::layout DataLayout>
    ROCWMMA_DEVICE inline DataT const*
        MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::dataCoord(
            DataT const* baseAddr, MatrixCoordT const& matrixCoord, uint32_t ldm)
    {
        return baseAddr
               + DataSpace::fromMatrixCoord(std::forward<MatrixCoordT const>(matrixCoord), ldm);
    }

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT,
        sycl::ext::oneapi::experimental::matrix::layout DataLayout>
    ROCWMMA_DEVICE inline DataT* MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::dataCoord(
        DataT* baseAddr, MatrixCoordT const& matrixCoord, uint32_t ldm)
    {
        return baseAddr
               + DataSpace::fromMatrixCoord(std::forward<MatrixCoordT const>(matrixCoord), ldm);
    }

} // namespace rocwmma

#endif // ROCWMMA_MAPPING_UTIL_IMPL_HPP
