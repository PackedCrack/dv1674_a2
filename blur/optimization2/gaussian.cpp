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
    [[nodiscard]] gaussian::BlurredImage make_blurred_image(const gaussian::Image& image)
    {
        gaussian::BlurredImage blurred{};
        blurred.red.resize(image.width * image.height);
        blurred.green.resize(image.width * image.height);
        blurred.blue.resize(image.width * image.height);
        blurred.width = image.width;
        blurred.height = image.height;
        blurred.maxval = image.maxval;

        return blurred;
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
        const std::uint8_t* pSrc,
        float* pDst,
        const Weights& weights,
        float centerWeight,
        float weightSum,
        std::int32_t radius,
        std::int32_t x,
        std::int32_t middleEndExclusive)
    {
        while(x < middleEndExclusive)
        {
            float result = centerWeight * static_cast<float>(pSrc[x]);
            for (std::int32_t i = 1; i <= radius; ++i)
            {
                float weight = weights[i];
                result += (weight * static_cast<float>(pSrc[x - i])) + (weight * static_cast<float>(pSrc[x + i]));
            }
            pDst[x] = result / weightSum;

            ++x;
        }
    }
    void calculate_middle(
        const std::uint8_t* pSrc,
        float* pDst,
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
        std::int32_t simdEnd = middleEndExclusive - 8;
        while(x <= simdEnd)
        {
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mul_ps&ig_expand=4044,1866,1664,4687
            __m256 result = _mm256_mul_ps(centerWeightAvx, cast_uint8_to_ps(pSrc + x));
            for (std::int32_t i = 1; i <= radius; ++i)
            {
                __m256 weight = _mm256_set1_ps(weights[i]);
#ifdef __FMA__
                // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fmadd_ps&ig_expand=4044,1866,1664,4687,3107
                result = _mm256_fmadd_ps(cast_uint8_to_ps(pSrc + x - i), weight, result);
                result = _mm256_fmadd_ps(cast_uint8_to_ps(pSrc + x + i), weight, result);
#else
#error
#endif
            }
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mul_ps&ig_expand=4044,1866,1664,4687,3107,4687
            result = _mm256_mul_ps(result, normalizationFactor);
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu_ps&ig_expand=4044,1866,1664,4687,3107,4687,6482,6543
            _mm256_storeu_ps(pDst + x, result);

            x += 8;
        }

        calculate_middle_tail(pSrc, pDst, weights, centerWeight, weightSum, radius, x, middleEndExclusive);
    }
    void calculate_border(
        const std::uint8_t* pSrc,
        std::int32_t srcIndex,
        float* pDst,
        const Weights& weights,
        float centerWeight,
        std::int32_t radius,
        std::int32_t rightEnd)
    {
        float result = centerWeight * static_cast<float>(pSrc[srcIndex]);
        float sumW = centerWeight;
        for (std::int32_t i = 1; i <= radius; ++i)
        {
            std::int32_t xLeft = srcIndex - i;
            std::int32_t xRight = srcIndex + i;
            if (xLeft >= 0)
            {
                result += weights[i] * static_cast<float>(pSrc[xLeft]);
                sumW += weights[i];
            }
            if (xRight < rightEnd)
            {
                result += weights[i] * static_cast<float>(pSrc[xRight]);
                sumW += weights[i];
            }
        }
        pDst[srcIndex] = result / sumW;
    }
    void calculate_left_border(
        const std::uint8_t* pSrc,
        float* pDst,
        const Weights& weights,
        float centerWeight,
        std::int32_t radius,
        std::int32_t rightBorder)
    {
        assert(radius < rightBorder);
        for (std::int32_t x = 0; x < radius; ++x)
        {
            calculate_border(pSrc, x, pDst, weights, centerWeight, radius, rightBorder);
        }
    }
    void calculate_right_border(
        const std::uint8_t* pSrc,
        float* pDst,
        const Weights& weights,
        float centerWeight,
        std::int32_t radius,
        std::int32_t rightBorder)
    {
        assert(radius < rightBorder);
        std::int32_t xStart = std::max(0, rightBorder - radius);
        for (std::int32_t x = xStart; x < rightBorder; ++x)
        {
            calculate_border(pSrc, x, pDst, weights, centerWeight, radius, rightBorder);
        }
    }
    void horizontal_pass(
                        const Weights& weights,
                        float weightSum,
                        std::int32_t radius,
                        std::int32_t width,
                        std::int32_t height,
                        const std::uint8_t* pSrc,
                        //std::span<const std::uint8_t> src,
                        float* pDst)
    {
        assert(weights.size() == radius + 1);

        std::int32_t samplesPerRow = width;
        std::int32_t srcStride = samplesPerRow;
        std::int32_t dstStride = samplesPerRow;

        __m256 normalizationFactor = _mm256_set1_ps(1.0f / weightSum);
        float centerWeight = weights.front();
        __m256 centerWeightAvx = _mm256_set1_ps(centerWeight);

        for (std::size_t y = 0; y < static_cast<std::size_t>(height); ++y)
        {
            std::size_t sourceIndex = y * srcStride;
            const std::uint8_t* pSrcPixel = pSrc + sourceIndex;
            std::size_t dstIndex = y * dstStride;
            float* pDstPixel = pDst + dstIndex;

            calculate_left_border(pSrcPixel, pDstPixel, weights, centerWeight, radius, width);

            calculate_middle(
                pSrcPixel,
                pDstPixel,
                weights,
                centerWeight,
                weightSum,
                radius,
                width,
                normalizationFactor,
                centerWeightAvx);

            calculate_right_border(pSrcPixel, pDstPixel, weights, centerWeight, radius, width);
        }
    }
}   // namespace
namespace gaussian
{
    BlurredImage calculate(ThreadPool& tp, const Image& image, std::int32_t radius)
    {
        Weights w = calculate_weights(radius);
        float weightSum = calc_weight_sum(w, radius);

        BlurredImage result = make_blurred_image(image);
        // Horizontal pass per color channel
        horizontal_pass(w, weightSum, radius, image.width, image.height, image.red.data(), result.red.data());
        horizontal_pass(w, weightSum, radius, image.width, image.height, image.green.data(), result.green.data());
        horizontal_pass(w, weightSum, radius, image.width, image.height, image.blue.data(), result.blue.data());

        // Transpose


        // Vertical pass per color channel

        std::array<uint8_t, 20> test{};
        for (auto i = 0; i < test.size(); ++i)
        {
            test[i] = static_cast<uint8_t>(result.red[i]);
        }
        std::uint8_t r = static_cast<std::uint8_t>(result.red.front());
        int a = 10;
    }
}    // namespace gaussian