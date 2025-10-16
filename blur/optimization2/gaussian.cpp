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
    [[nodsicard]] __m256 cast_uint8_to_ps(const uint8_t* pValue)
    {
        // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadl_epi64&ig_expand=4044
        __m128i lo8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pValue));        // This might be UB.. I dunno.
        // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cvtepu8_epi32&ig_expand=4044,1866
        __m256i i32 = _mm256_cvtepu8_epi32(lo8);                                        // 8x uint8 to 8x int32
        // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cvtepi32_ps&ig_expand=4044,1866,1664
        return _mm256_cvtepi32_ps(i32);                                                 // 8x int32 to 8x float32
    }
    void calculate_middle_tail(
        const std::uint8_t* pSrcRow,
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
    void calculate_middle(
        const std::uint8_t* pSrcRow,
        float* pDstRow,
        const Weights& weights,
        float centerWeight,
        float weightSum,
        std::int32_t radius,
        std::int32_t width,
        __m256 normalizationFactor,
        __m256 centerWeightAvx
        )
    {
        assert(width > radius);
        std::int32_t middleEndExclusive = width - radius;
        std::int32_t x = radius;
        static constexpr std::int32_t numLanes = 8;
        std::int32_t simdEnd = middleEndExclusive - numLanes;
        while(x <= simdEnd)
        {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mul_ps&ig_expand=4044,1866,1664,4687
            __m256 result = _mm256_mul_ps(centerWeightAvx, cast_uint8_to_ps(pSrcRow + x));
            for (std::int32_t i = 1; i <= radius; ++i)
            {
                __m256 weight = _mm256_set1_ps(weights[i]);
                __m256 left = cast_uint8_to_ps(pSrcRow + x - i);
                __m256 right = cast_uint8_to_ps(pSrcRow + x + i);
#ifdef __FMA__
                // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fmadd_ps&ig_expand=4044,1866,1664,4687,3107
                result = _mm256_fmadd_ps(left, weight, result);
                result = _mm256_fmadd_ps(right, weight, result);
#else
                result = _mm256_add_ps(_mm256_mul_ps(left, vw[i]), result);
                result = _mm256_add_ps(_mm256_mul_ps(right, vw[i]), result);
#endif
            }
#ifdef __FMA__
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mul_ps&ig_expand=4044,1866,1664,4687,3107,4687
            result = _mm256_mul_ps(result, normalizationFactor);
#else
            __m256 denominator = _mm256_set1_ps(weightSum);
            result = _mm256_div_ps(result, denominator);
#endif
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu_ps&ig_expand=4044,1866,1664,4687,3107,4687,6482,6543
            _mm256_storeu_ps(pDstRow + x, result);

            x += numLanes;
        }

        calculate_middle_tail(pSrcRow, pDstRow, weights, centerWeight, weightSum, radius, x, middleEndExclusive);
    }
    void calculate_row_border(
        const std::uint8_t* pSrcRow,
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
    void calculate_left_border(
        const std::uint8_t* pSrcRow,
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
    void calculate_right_border(
        const std::uint8_t* pSrcRow,
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
[[nodiscard]] auto make_horizontal_pass_task(
    Latch& latch,
    const Weights& weights,
    float weightSum,
    float centerWeight,
    __m256 centerWeightAvx,
    __m256 normalizationFactor,
    std::int32_t radius,
    std::int32_t width,
    const std::uint8_t* pSrcRow,
    float* pDstRow)
    {
        return [&latch, &weights, weightSum, centerWeight, centerWeightAvx, normalizationFactor, radius,  width, pSrcRow, pDstRow]()
        {
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
                centerWeightAvx);

            calculate_right_border(pSrcRow, pDstRow, weights, centerWeight, radius, width);

            latch.count_down();
        };
    }
    void horizontal_pass(ThreadPool& tp,
                        Latch& latch,
                        const Weights& weights,
                        float weightSum,
                        std::int32_t radius,
                        std::int32_t width,
                        std::int32_t height,
                        std::span<const std::uint8_t> src,
                        std::span<float> dst)
    {
        assert(weights.size() == radius + 1);

        std::int32_t srcStride = width;
        std::int32_t dstStride = width;

        __m256 normalizationFactor = _mm256_set1_ps(1.0f / weightSum);
        float centerWeight = weights.front();
        __m256 centerWeightAvx = _mm256_set1_ps(centerWeight);

        for (std::size_t y = 0; y < static_cast<std::size_t>(height); ++y)
        {
            std::size_t sourceIndex = y * srcStride;
            const std::uint8_t* pSrcRow = std::addressof(src[sourceIndex]);
            std::size_t dstIndex = y * dstStride;
            float* pDstRow = std::addressof(dst[dstIndex]);

            tp.add_task(
                make_horizontal_pass_task(
                    latch,
                    weights,
                    weightSum,
                    centerWeight,
                    centerWeightAvx,
                    normalizationFactor,
                    radius,
                    width,
                    pSrcRow,
                    pDstRow));
        }
    }
    void to_uint8(std::span<const float> src, std::span<uint8_t> dst)
    {
            // This has to be done because the original code just assigns float to uint8 mid calculations lmfao
            assert(src.size() == dst.size());

            __m256 vmin = _mm256_set1_ps(0.0f);
            __m256 vmax = _mm256_set1_ps(255.0f);

            std::size_t n = src.size();
            std::size_t i = 0;
            while(i + 8 <= n)
            {
                __m256 v = _mm256_loadu_ps(std::addressof(src[i]));
                // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_max_ps&ig_expand=4044,1866,1664,4687,3107,4687,6482,6543,4360
                v = _mm256_max_ps(vmin, _mm256_min_ps(v, vmax));
                // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cvttps_epi32&ig_expand=4044,1866,1664,4687,3107,4687,6482,6543,4360,2414
                __m256i i32 = _mm256_cvttps_epi32(v);               // truncate
                // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_castsi256_si128&ig_expand=4044,1866,1664,4687,3107,4687,6482,6543,4360,2414,690
                __m128i low = _mm256_castsi256_si128(i32);
                // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_extractf128_si256&ig_expand=4044,1866,1664,4687,3107,4687,6482,6543,4360,2414,690,2939
                __m128i high = _mm256_extractf128_si256(i32, 1);
                // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_packus_epi32&ig_expand=4044,1866,1664,4687,3107,4687,6482,6543,4360,2414,690,2939,4880
                __m128i u16 = _mm_packus_epi32(low, high);             // 8x uint16
                // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_packus_epi16&ig_expand=4044,1866,1664,4687,3107,4687,6482,6543,4360,2414,690,2939,4880,4871
                __m128i u8  = _mm_packus_epi16(u16, u16);           // 16x uint8, use low 8
                _mm_storel_epi64(reinterpret_cast<__m128i*>(std::addressof(dst[i])), u8);

                i += 8;
            }

            while(i < n)
            {
                float v = src[i];
                if (v < 0.f)
                {
                    v = 0.f;
                }
                else if (v > 255.f)
                {
                    v = 255.f;
                }
                dst[i] = static_cast<uint8_t>(static_cast<int>(v));  // trunc

                ++i;
            }
        }

    void calculate_column_middle_tail(const std::uint8_t* pSrcCol,
                                        float* pDstCol,
                                        const Weights& weights,
                                        float centerWeight,
                                        float weightSum,
                                        std::int32_t radius,
                                        std::int32_t column,
                                        std::int32_t middleEndExclusive,
                                        std::int32_t stride)
    {
        while (column < middleEndExclusive)
        {
            float result = centerWeight * static_cast<float>(pSrcCol[column * stride]);
            for (std::int32_t i = 1; i <= radius; ++i) {
                float w = weights[i];
                result += w * static_cast<float>(pSrcCol[(column - i) * stride])
                        + w * static_cast<float>(pSrcCol[(column + i) * stride]);
            }
            pDstCol[column * stride] = result / weightSum;
            ++column;
        }
    }
    void calculate_column_middle(
        const std::uint8_t* pSrcCol,
        float* pDstCol,
        const Weights& weights,
        float centerWeight,
        float weightSum,
        std::int32_t radius,
        std::int32_t height,
        std::int32_t stride,
        __m256 normalizationFactor,
        __m256 centerWeightAvx
        )
    {
        assert(height > radius);
        std::int32_t middleEndExclusive = height - radius;
        std::int32_t column = radius;
        static constexpr std::int32_t numLanes = 8;
        std::int32_t simdEnd = middleEndExclusive - numLanes;
        while(column <= simdEnd)
        {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mul_ps&ig_expand=4044,1866,1664,4687
            __m256 result = _mm256_mul_ps(centerWeightAvx, cast_uint8_to_ps(pSrcCol + column));
            for (std::int32_t i = 1; i <= radius; ++i)
            {
                __m256 weight = _mm256_set1_ps(weights[i]);
                __m256 top = cast_uint8_to_ps(pSrcCol + column - i);
                __m256 bot = cast_uint8_to_ps(pSrcCol + column + i);
#ifdef __FMA__
                // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fmadd_ps&ig_expand=4044,1866,1664,4687,3107
                result = _mm256_fmadd_ps(top, weight, result);
                result = _mm256_fmadd_ps(bot, weight, result);
#else
                result = _mm256_add_ps(_mm256_mul_ps(top, vw[i]), result);
                result = _mm256_add_ps(_mm256_mul_ps(bot, vw[i]), result);
#endif
            }
#ifdef __FMA__
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mul_ps&ig_expand=4044,1866,1664,4687,3107,4687
            result = _mm256_mul_ps(result, normalizationFactor);
#else
            __m256 denominator = _mm256_set1_ps(weightSum);
            result = _mm256_div_ps(result, denominator);
#endif
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu_ps&ig_expand=4044,1866,1664,4687,3107,4687,6482,6543
            _mm256_storeu_ps(pDstCol + column, result);

            column += numLanes;
        }

        calculate_middle_tail(pSrcCol, pDstCol, weights, centerWeight, weightSum, radius, column, middleEndExclusive);
    }
    void calculate_column_border(
        const std::uint8_t* pSrcCol,
        std::int32_t stride,
        float* pDstCol,
        const Weights& weights,
        float centerWeight,
        std::int32_t radius,
        std::int32_t height,
        std::int32_t column)
    {
        float result = centerWeight * static_cast<float>(pSrcCol[column * stride]);
        float sumW = centerWeight;
        for (std::int32_t i = 1; i <= radius; ++i)
        {
            std::int32_t colTop = column - i;
            std::int32_t colBottom = column + i;
            if (colTop >= 0)
            {
                result += weights[i] * static_cast<float>(pSrcCol[colTop * stride]);
                sumW += weights[i];
            }
            if (colBottom < height)
            {
                result += weights[i] * static_cast<float>(pSrcCol[colBottom * stride]);
                sumW += weights[i];
            }
        }
        pDstCol[column * stride] = result / sumW;
    }
    void calculate_bottom_border(
        const std::uint8_t* pSrcCol,
        float* pDstCol,
        const Weights& weights,
        float centerWeight,
        std::int32_t radius,
        std::int32_t stride,
        std::int32_t height)
    {
        assert(radius < height);
        std::int32_t start = height - radius;
        for (std::int32_t y = start; y < height; ++y)
        {
            calculate_column_border(pSrcCol, stride, pDstCol, weights, centerWeight, radius, height, y);
        }
    }
    void calculate_top_border(
        const std::uint8_t* pSrcCol,
        float* pDstCol,
        const Weights& weights,
        float centerWeight,
        std::int32_t radius,
        std::int32_t stride,
        std::int32_t height)
    {
        assert(radius < height);
        for (std::int32_t y = 0; y < radius; ++y)
        {
            calculate_column_border(pSrcCol, stride, pDstCol, weights, centerWeight, radius, height, y);
        }
    }
    void vertical_pass(ThreadPool& tp,
                        Latch& latch,
                        const Weights& weights,
                        float weightSum,
                        std::int32_t radius,
                        std::int32_t width,
                        std::int32_t height,
                        std::span<const std::uint8_t> src,
                        std::span<float> dst)
    {
        __m256 normalizationFactor = _mm256_set1_ps(1.0f / weightSum);
        float centerWeight = weights.front();
        __m256 centerWeightAvx = _mm256_set1_ps(centerWeight);

        for (std::size_t x = 0; x < static_cast<std::size_t>(width); ++x)
        {
            const std::uint8_t* pSrcCol = std::addressof(src[x]);
            float* pDstCol = std::addressof(dst[x]);

            calculate_bottom_border(pSrcCol, pDstCol, weights, centerWeight, radius, width, height);

            calculate_column_middle(pSrcCol, pDstCol, weights, centerWeight, weightSum, radius, height, width, normalizationFactor, centerWeightAvx);

            calculate_top_border(pSrcCol, pDstCol, weights, centerWeight, radius, width, height);
        }
    }
}   // namespace
namespace gaussian
{
    Image& add_blur(ThreadPool& tp, Image& image, std::int32_t radius)
    {
        Weights w = calculate_weights(radius);
        float weightSum = calc_weight_sum(w, radius);

        ScratchImage scratch = make_scratch_image(image);

        // Horizontal pass per color channel
        std::int64_t numTasks = image.height * 3; // One task for each row for each color channel
        Latch latch{ numTasks };
        horizontal_pass(tp, latch, w, weightSum, radius, image.width, image.height, image.red, scratch.red);
        horizontal_pass(tp, latch, w, weightSum, radius, image.width, image.height, image.green, scratch.green);
        horizontal_pass(tp, latch, w, weightSum, radius, image.width, image.height, image.blue, scratch.blue);
        latch.wait();

        to_uint8(scratch.red, image.red);
        to_uint8(scratch.green, image.green);
        to_uint8(scratch.blue, image.blue);

        // Vertical pass per color channel
        vertical_pass(tp, latch, w, weightSum, radius, image.width, image.height, image.red, scratch.red);
        vertical_pass(tp, latch, w, weightSum, radius, image.width, image.height, image.green, scratch.green);
        vertical_pass(tp, latch, w, weightSum, radius, image.width, image.height, image.blue, scratch.blue);

        to_uint8(scratch.red, image.red);
        to_uint8(scratch.green, image.green);
        to_uint8(scratch.blue, image.blue);

        return image;
    }
}    // namespace gaussian