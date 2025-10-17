#include "gaussian.hpp"

#include <array>
#include <cmath>
#include <cassert>
#include <immintrin.h>
#include <span>


namespace
{
    constexpr std::size_t MAX_RADIUS = 15;
    using Weights = std::array<float, MAX_RADIUS + 1>;
    using VectorWeights = std::array<__m256, MAX_RADIUS + 1>;
    [[nodiscard]] Weights calculate_weights(std::int32_t radius)
    {
        static constexpr float MAX_X = 1.333f; // Why?
        static constexpr float pi = 3.14159265358979323846f;

        Weights weights{};
        for (std::int32_t i = 0; i <= radius; ++i)
        {
            double x = static_cast<float>(i) * MAX_X / static_cast<float>(radius);
            weights[i] = static_cast<float>(std::exp(-x * x * pi));
        }

        return weights;
    }
    [[nodiscard]] float calc_weight_sum(const Weights& weights, std::int32_t radius)
    {
        assert(weights.size() == radius + 1);

        float sum = weights.front();
        for (auto&& it = std::begin(weights) + 1; it != std::end(weights); ++it)
        {
            sum += 2.0f * *it;
        }

        return sum;
    }
    [[nodiscard]] gaussian::ScratchImage make_scratch_image(const gaussian::Image& image)
    {
        gaussian::ScratchImage scratch{};
        scratch.red.resize(image.width * image.height);
        scratch.green.resize(image.width * image.height);
        scratch.blue.resize(image.width * image.height);
        scratch.width = image.width;
        scratch.height = image.height;
        scratch.maxval = image.maxval;

        return scratch;
    }
    [[nodiscard]] std::int64_t required_block_tasks(std::int32_t total,
                                                    std::int32_t numRows)
    {
        return (static_cast<std::int64_t>(total) + numRows - 1) / numRows;
    }
    [[nodiscard]] __m256 cast_uint8_to_ps(const uint8_t* pValue)
    {
        // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadl_epi64&ig_expand=4044
        __m128i b8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pValue));        // 8x uint8
        // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cvtepu8_epi32&ig_expand=4044,1866
        __m128i u16 = _mm_cvtepu8_epi16(b8);                                            // 8x uint8 to 8x int32
        // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cvtepu16_epi32&ig_expand=1854,1771
        __m256i i32 = _mm256_cvtepu16_epi32(u16);
        // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cvtepi32_ps&ig_expand=4044,1866,1664
        return _mm256_cvtepi32_ps(i32);                                                 // 8x int32 to 8x float32
    }
    void calculate_middle_tail(const std::uint8_t* pSrcRow,
                                float* pDstRow,
                                const Weights& weights,
                                float centerWeight,
                                float weightSum,
                                std::int32_t radius,
                                std::int32_t x,
                                std::int32_t middleEndExclusive)
    {
        while(x < middleEndExclusive)
        {
            float result = centerWeight * static_cast<float>(pSrcRow[x]);
            for (std::int32_t i = 1; i <= radius; ++i)
            {
                float weight = weights[i];
                result += (weight * static_cast<float>(pSrcRow[x - i])) + (weight * static_cast<float>(pSrcRow[x + i]));
            }
            pDstRow[x] = result / weightSum;

            ++x;
        }
    }
    void calculate_middle(const std::uint8_t* pSrcRow,
                          float* pDstRow,
                          const Weights& weights,
                          float centerWeight,
                          float weightSum,
                          std::int32_t radius,
                          std::int32_t width,
                          __m256 normalizationFactor,
                          const std::array<__m256, MAX_RADIUS + 1>& vWeights)
    {
        assert(width > radius);
        __m256 vCenterWeight = vWeights.front();
        std::int32_t middleEndExclusive = width - radius;
        std::int32_t x = radius;
        static constexpr std::int32_t numLanes = 16;
        std::int32_t simdEnd = middleEndExclusive - numLanes;
        while(x <= simdEnd)
        {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mul_ps&ig_expand=4044,1866,1664,4687
            __m256 result1 = _mm256_mul_ps(vCenterWeight, cast_uint8_to_ps(pSrcRow + x + 0));
            __m256 result2 = _mm256_mul_ps(vCenterWeight, cast_uint8_to_ps(pSrcRow + x + 8));
            for (std::int32_t i = 1; i <= radius; ++i)
            {
                __m256 left1 = cast_uint8_to_ps(pSrcRow + x - i + 0);
                __m256 right1 = cast_uint8_to_ps(pSrcRow + x + i + 0);
                __m256 left2 = cast_uint8_to_ps(pSrcRow + x - i + 8);
                __m256 right2 = cast_uint8_to_ps(pSrcRow + x + i + 8);
#ifdef __FMA__
                // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fmadd_ps&ig_expand=4044,1866,1664,4687,3107
                result1 = _mm256_fmadd_ps(_mm256_add_ps(left1, right1), vWeights[i], result1);
                result2 = _mm256_fmadd_ps(_mm256_add_ps(left2, right2), vWeights[i], result2);
#else
                result1 = _mm256_add_ps(result1, _mm256_mul_ps(_mm256_add_ps(left1, right1), vWeights[i]));
                result2 = _mm256_add_ps(result2, _mm256_mul_ps(_mm256_add_ps(left2, right2), vWeights[i]));
#endif
            }
#ifdef __FMA__
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mul_ps&ig_expand=4044,1866,1664,4687,3107,4687
            result1 = _mm256_mul_ps(result1, normalizationFactor);
            result2 = _mm256_mul_ps(result2, normalizationFactor);
#else
            __m256 denominator = _mm256_set1_ps(weightSum);
            result1 = _mm256_div_ps(result1, denominator);
            result2 = _mm256_div_ps(result2, denominator);
#endif
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu_ps&ig_expand=4044,1866,1664,4687,3107,4687,6482,6543
            _mm256_storeu_ps(pDstRow + x + 0, result1);
            _mm256_storeu_ps(pDstRow + x + 8, result2);

            x += numLanes;
        }

        calculate_middle_tail(pSrcRow, pDstRow, weights, centerWeight, weightSum, radius, x, middleEndExclusive);
    }
    void calculate_row_border(const std::uint8_t* pSrcRow,
                                std::int32_t srcIndex,
                                float* pDstRow,
                                const Weights& weights,
                                float centerWeight,
                                std::int32_t radius,
                                std::int32_t rightEnd)
    {
        float result = centerWeight * static_cast<float>(pSrcRow[srcIndex]);
        float sumW = centerWeight;
        for (std::int32_t i = 1; i <= radius; ++i)
        {
            std::int32_t xLeft = srcIndex - i;
            std::int32_t xRight = srcIndex + i;
            if (xLeft >= 0)
            {
                result += weights[i] * static_cast<float>(pSrcRow[xLeft]);
                sumW += weights[i];
            }
            if (xRight < rightEnd)
            {
                result += weights[i] * static_cast<float>(pSrcRow[xRight]);
                sumW += weights[i];
            }
        }
        pDstRow[srcIndex] = result / sumW;
    }
    void calculate_left_border(const std::uint8_t* pSrcRow,
                                float* pDstRow,
                                const Weights& weights,
                                float centerWeight,
                                std::int32_t radius,
                                std::int32_t rightEnd)
    {
        assert(radius < rightEnd);
        for (std::int32_t x = 0; x < radius; ++x)
        {
            calculate_row_border(pSrcRow, x, pDstRow, weights, centerWeight, radius, rightEnd);
        }
    }
    void calculate_right_border(const std::uint8_t* pSrcRow,
                                float* pDstRow,
                                const Weights& weights,
                                float centerWeight,
                                std::int32_t radius,
                                std::int32_t rightEnd)
    {
        assert(radius < rightEnd);
        std::int32_t xStart = std::max(0, rightEnd - radius);
        for (std::int32_t x = xStart; x < rightEnd; ++x)
        {
            calculate_row_border(pSrcRow, x, pDstRow, weights, centerWeight, radius, rightEnd);
        }
    }
[[nodiscard]] auto make_horizontal_pass_task(Latch& latch,
                                            const VectorWeights& vWeights,
                                            const Weights& weights,
                                            float weightSum,
                                            std::int32_t radius,
                                            std::int32_t width,
                                            std::span<const std::uint8_t> src,
                                            std::span<float> dst,
                                            std::size_t startRow,
                                            std::size_t endRow)
    {
        return [=, &latch]()
        {
            __m256 normalizationFactor = _mm256_set1_ps(1.0f / weightSum);
            float centerWeight = weights[0];
            std::int32_t stride = width;

            for (std::size_t row = startRow; row < endRow; ++row )
            {
                std::size_t index = row * stride;
                const std::uint8_t* pSrcRow = std::addressof(src[index]);
                float* pDstRow = std::addressof(dst[index]);

                calculate_left_border(pSrcRow, pDstRow, weights, centerWeight, radius, width);

                calculate_middle(
                    pSrcRow,
                    pDstRow,
                    weights,
                    centerWeight,
                    weightSum,
                    radius,
                    width,
                    normalizationFactor,
                    vWeights);

                calculate_right_border(pSrcRow, pDstRow, weights, centerWeight, radius, width);
            }

            latch.count_down();
        };
    }
    void horizontal_pass(ThreadPool& tp,
                        Latch& latch,
                        const VectorWeights vWeights,
                        const Weights& weights,
                        float weightSum,
                        std::int32_t radius,
                        std::int32_t width,
                        std::int32_t height,
                        std::span<const std::uint8_t> src,
                        std::span<float> dst)
    {
        assert(weights.size() == radius + 1);

        static constexpr int rowsPerTask = 64;  // should take this as parameter but whatever i wont tweak it..
        for (std::size_t y = 0; y < static_cast<std::size_t>(height); y += rowsPerTask)
        {
            std::size_t startRow = y;
            std::size_t endRow = std::min(static_cast<std::size_t>(height), startRow + rowsPerTask);
            tp.add_task(
                make_horizontal_pass_task(
                    latch,
                    vWeights,
                    weights,
                    weightSum,
                    radius,
                    width,
                    src,
                    dst,
                    startRow,
                    endRow));
        }
    }
    void to_uint8(ThreadPool& tp, std::span<const float> src, std::span<uint8_t> dst)
    {
        // This has to be done because the original code just assign floats to uint8 mid calculations lmfao
        assert(src.size() == dst.size());
        static constexpr std::size_t numLanes = 8;
        assert(src.size() > numLanes);

        auto numTasks = static_cast<std::int64_t>(tp.thread_count());
        std::int64_t taskSize = (static_cast<std::int64_t>(src.size()) + numTasks - 1) / numTasks;

        Latch latch{ numTasks };
        for (std::int64_t task = 0; task < numTasks; ++task)
        {
            std::int64_t start = task * taskSize;
            std::int64_t end = std::min(static_cast<std::int64_t>(src.size()), start + taskSize);

            tp.add_task([&latch, src, dst, start, end]()
            {
                __m256 min = _mm256_set1_ps(0.0f);
                __m256 max = _mm256_set1_ps(255.0f);

                static constexpr std::int64_t numLanes = 8;
                std::int64_t i = start;
                while(i + numLanes <= end)
                {
                    __m256 v = _mm256_loadu_ps(std::addressof(src[i]));
                    // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_max_ps&ig_expand=4044,1866,1664,4687,3107,4687,6482,6543,4360
                    v = _mm256_max_ps(min, _mm256_min_ps(v, max));
                    // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cvttps_epi32&ig_expand=4044,1866,1664,4687,3107,4687,6482,6543,4360,2414
                    __m256i i32 = _mm256_cvttps_epi32(v);
                    // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_castsi256_si128&ig_expand=4044,1866,1664,4687,3107,4687,6482,6543,4360,2414,690
                    __m128i lo = _mm256_castsi256_si128(i32);
                    // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_extractf128_si256&ig_expand=4044,1866,1664,4687,3107,4687,6482,6543,4360,2414,690,2939
                    __m128i hi = _mm256_extractf128_si256(i32, 1);
                    // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_packus_epi32&ig_expand=4044,1866,1664,4687,3107,4687,6482,6543,4360,2414,690,2939,4880
                    __m128i u16 = _mm_packus_epi32(lo, hi);
                    // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_packus_epi16&ig_expand=4044,1866,1664,4687,3107,4687,6482,6543,4360,2414,690,2939,4880,4871
                    __m128i u8  = _mm_packus_epi16(u16, u16);
                    _mm_storel_epi64(reinterpret_cast<__m128i*>(std::addressof(dst[i])), u8);

                    i += numLanes;
                }

                while(i < end)
                {
                    dst[i] = static_cast<uint8_t>(src[i]);
                    ++i;
                }

                latch.count_down();
            });
        }

        latch.wait();
    }
    void transpose8x8(const float* pSrc,
                      std::int32_t srcStride,
                      float* pDst,
                      std::int32_t dstStride)
    {
        // https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2
        __m256 r0 = _mm256_loadu_ps(pSrc + 0 * srcStride);
        __m256 r1 = _mm256_loadu_ps(pSrc + 1 * srcStride);
        __m256 r2 = _mm256_loadu_ps(pSrc + 2 * srcStride);
        __m256 r3 = _mm256_loadu_ps(pSrc + 3 * srcStride);
        __m256 r4 = _mm256_loadu_ps(pSrc + 4 * srcStride);
        __m256 r5 = _mm256_loadu_ps(pSrc + 5 * srcStride);
        __m256 r6 = _mm256_loadu_ps(pSrc + 6 * srcStride);
        __m256 r7 = _mm256_loadu_ps(pSrc + 7 * srcStride);

        __m256 t0 = _mm256_unpacklo_ps(r0, r1);
        __m256 t1 = _mm256_unpackhi_ps(r0, r1);
        __m256 t2 = _mm256_unpacklo_ps(r2, r3);
        __m256 t3 = _mm256_unpackhi_ps(r2, r3);
        __m256 t4 = _mm256_unpacklo_ps(r4, r5);
        __m256 t5 = _mm256_unpackhi_ps(r4, r5);
        __m256 t6 = _mm256_unpacklo_ps(r6, r7);
        __m256 t7 = _mm256_unpackhi_ps(r6, r7);

        __m256 u0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1,0,1,0));
        __m256 u1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3,2,3,2));
        __m256 u2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1,0,1,0));
        __m256 u3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3,2,3,2));
        __m256 u4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1,0,1,0));
        __m256 u5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3,2,3,2));
        __m256 u6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1,0,1,0));
        __m256 u7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3,2,3,2));

        __m256 v0 = _mm256_permute2f128_ps(u0, u4, 0x20);
        __m256 v1 = _mm256_permute2f128_ps(u1, u5, 0x20);
        __m256 v2 = _mm256_permute2f128_ps(u2, u6, 0x20);
        __m256 v3 = _mm256_permute2f128_ps(u3, u7, 0x20);
        __m256 v4 = _mm256_permute2f128_ps(u0, u4, 0x31);
        __m256 v5 = _mm256_permute2f128_ps(u1, u5, 0x31);
        __m256 v6 = _mm256_permute2f128_ps(u2, u6, 0x31);
        __m256 v7 = _mm256_permute2f128_ps(u3, u7, 0x31);

        _mm256_storeu_ps(pDst + 0 * dstStride, v0);
        _mm256_storeu_ps(pDst + 1 * dstStride, v1);
        _mm256_storeu_ps(pDst + 2 * dstStride, v2);
        _mm256_storeu_ps(pDst + 3 * dstStride, v3);
        _mm256_storeu_ps(pDst + 4 * dstStride, v4);
        _mm256_storeu_ps(pDst + 5 * dstStride, v5);
        _mm256_storeu_ps(pDst + 6 * dstStride, v6);
        _mm256_storeu_ps(pDst + 7 * dstStride, v7);
    }
    void transpose_block_scalar(std::span<const float> src,
                                    std::int32_t srcStride,
                                    std::span<float> dst,
                                    std::int32_t dstStride,
                                    std::int32_t column,
                                    std::int32_t row,
                                    std::int32_t columnEnd,
                                    std::int32_t rowEnd)
    {
        for (std::int32_t y = row; y < rowEnd; ++y)
        {
            std::int32_t srcOffset = y * srcStride;
            const float* pSrc = std::addressof(src[srcOffset + column]);
            std::int32_t dstOffset = column * dstStride;
            float* pDst = std::addressof(dst[dstOffset + y]);
            for (std::int32_t x = column; x < columnEnd; ++x)
            {
                *pDst = *pSrc;
                ++pSrc;
                pDst += dstStride;
            }
        }
    }
    void transpose_block(std::span<const float> src,
                        std::int32_t srcStride,
                        std::span<float> dst,
                        std::int32_t dstStride,
                        std::int32_t columnStart,
                        std::int32_t rowStart,
                        std::int32_t columnEnd,
                        std::int32_t rowEnd)
    {
        std::int32_t row = rowStart;
        while(row + 8 <= rowEnd)
        {
            std::int32_t column = columnStart;
            while(column + 8 <= columnEnd)
            {
                std::int32_t srcOffset = row * srcStride;
                std::int32_t dstOffset = column * dstStride;

                const float* pSrc = std::addressof(src[srcOffset + column]);
                float* pDst = std::addressof(dst[dstOffset + row]);
                transpose8x8(pSrc, srcStride, pDst, dstStride);

                column += 8;
            }

            if (column < columnEnd)
            {
                // Right edge
                transpose_block_scalar(src, srcStride, dst, dstStride, column, row, columnEnd, row + 8);
            }

            row += 8;
        }
        if (row < rowEnd)
        {
            // Bottom edge
            transpose_block_scalar(src, srcStride, dst, dstStride, columnStart, row, columnEnd, rowEnd);
        }
    }
    void transpose(ThreadPool& pool,
                    Latch& latch,
                    std::span<const float> src,
                    std::int32_t srcStride,
                    std::span<float> dst,
                    std::int32_t dstStride,
                    std::int32_t width,
                    std::int32_t height,
                    std::int32_t blockSize)
    {
        std::int32_t tasks = 0;
        for (std::int32_t rowStart = 0; rowStart < height; rowStart += blockSize)
        {
            std::int32_t rowEnd = std::min(rowStart + blockSize, height);
            for (std::int32_t columnStart = 0; columnStart < width; columnStart += blockSize)
            {
                std::int32_t columnEnd = std::min(columnStart + blockSize, width);
                pool.add_task([=, &latch]()
                {
                    transpose_block(src, srcStride, dst, dstStride, columnStart, rowStart, columnEnd, rowEnd);
                    latch.count_down();
                });
                ++tasks;
            }
        }
    }
    [[nodiscard]] std::int64_t num_transpose_tasks(std::int32_t width, std::int32_t height, std::int32_t numBlocks)
    {
        auto ceil = [](std::int32_t value, std::int32_t denominator)
        {
            return (value + denominator - 1) / denominator;
        };
        return ceil(width, numBlocks) * ceil(height, numBlocks);
    }
    void interleave_results(ThreadPool& tp, const gaussian::ScratchImage& results, std::span<std::uint8_t> outInterleaved)
    {
        auto numTasks = static_cast<std::int64_t>(tp.thread_count());
        std::int64_t rowsPerTask = (static_cast<std::int64_t>(results.height) + numTasks - 1) / numTasks;

        Latch latch{ numTasks };
        for (std::int64_t task = 0; task < numTasks; ++task)
        {
            std::int64_t start = task * rowsPerTask;
            std::int64_t end = std::min(static_cast<std::int64_t>(results.height), start + rowsPerTask);
            tp.add_task([&latch, &results, outInterleaved, start, end]()
            {
                for (std::int64_t row = start; row < end; ++row)
                {
                    std::int64_t firstSrcElement = row * results.width;
                    std::int64_t firstDstElement = firstSrcElement * 3;
                    for (int x = 0; x < results.width; ++x)
                    {
                        outInterleaved[firstDstElement + 3 * x + 0] = static_cast<std::uint8_t>(results.red[firstSrcElement + x]);
                        outInterleaved[firstDstElement + 3 * x + 1] = static_cast<std::uint8_t>(results.green[firstSrcElement + x]);
                        outInterleaved[firstDstElement + 3 * x + 2] = static_cast<std::uint8_t>(results.blue[firstSrcElement + x]);
                    }
                }
                latch.count_down();
            });
        }

        latch.wait();
    }
}   // namespace
namespace gaussian
{
    std::vector<std::uint8_t> add_blur(ThreadPool& tp, Image& image, std::int32_t radius)
    {
        assert(image.red.size() == image.green.size());
        assert(image.green.size() == image.blue.size());

        Weights w = calculate_weights(radius);
        float weightSum = calc_weight_sum(w, radius);
        alignas(32) static std::array<__m256, MAX_RADIUS + 1> vWeights{};
        for (std::size_t i = 0; i < w.size(); ++i)
        {
            vWeights[i] = _mm256_set1_ps(w[i]);
        }

        ScratchImage scratch = make_scratch_image(image);

        // Horizontal pass per color channel
        static constexpr std::int32_t numChannels = 3;
        static constexpr std::int32_t rowsPerTask = 64;
        {
            std::int64_t numTasks = required_block_tasks(image.height, rowsPerTask) * numChannels;
            Latch latch{ numTasks };
            horizontal_pass(tp, latch, vWeights, w, weightSum, radius, image.width, image.height, image.red, scratch.red);
            horizontal_pass(tp, latch, vWeights, w, weightSum, radius, image.width, image.height, image.green, scratch.green);
            horizontal_pass(tp, latch, vWeights, w, weightSum, radius, image.width, image.height, image.blue, scratch.blue);

            latch.wait();
        }

        static constexpr std::int32_t blockSize = 64;
        ScratchImage transposed = make_scratch_image(image);
        {
            std::int32_t srcStride = image.width;
            std::int32_t dstStride = image.height;
            Latch latch(num_transpose_tasks(image.width, image.height, blockSize) * numChannels);

            transpose(tp, latch, scratch.red, srcStride, transposed.red, dstStride, image.width, image.height, blockSize);
            transpose(tp, latch, scratch.green, srcStride, transposed.green, dstStride, image.width, image.height, blockSize);
            transpose(tp, latch, scratch.blue, srcStride, transposed.blue, dstStride, image.width, image.height, blockSize);

            latch.wait();
        }

        to_uint8(tp, transposed.red, image.red);
        to_uint8(tp, transposed.green, image.green);
        to_uint8(tp, transposed.blue, image.blue);

        // Vertical pass per color channel
        {
            std::int64_t numTasks = required_block_tasks(image.width, rowsPerTask) * numChannels;
            Latch latch{ numTasks };
            horizontal_pass(tp, latch, vWeights, w, weightSum, radius, image.height, image.width, image.red, scratch.red);
            horizontal_pass(tp, latch, vWeights, w, weightSum, radius, image.height, image.width, image.green, scratch.green);
            horizontal_pass(tp, latch, vWeights, w, weightSum, radius, image.height, image.width, image.blue, scratch.blue);

            latch.wait();
        }

        {
            std::int32_t srcStride = image.height;
            std::int32_t dstStride = image.width;
            Latch latch(num_transpose_tasks(image.width, image.height, blockSize) * numChannels);

            transpose(tp, latch, scratch.red, srcStride, transposed.red, dstStride, image.height, image.width, blockSize);
            transpose(tp, latch, scratch.green, srcStride, transposed.green, dstStride, image.height, image.width, blockSize);
            transpose(tp, latch, scratch.blue, srcStride, transposed.blue, dstStride, image.height, image.width, blockSize);

            latch.wait();
        }

        std::vector<std::uint8_t> interleavedResult(image.width * image.height * numChannels);
        interleave_results(tp, transposed, interleavedResult);

        return interleavedResult;
    }
}    // namespace gaussian