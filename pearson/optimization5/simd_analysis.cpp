#include "simd_analysis.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <immintrin.h>
#include <cassert>

namespace
{
    std::size_t to_index(std::size_t xi, std::size_t yi, std::size_t numVectors)
    {
        assert(xi < yi);
        assert(yi < numVectors);

        // Maps 2D coordinate to 1D
        // https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
        return ((2 * numVectors - 3 - xi) * xi) / 2 + yi - 1;
    }
    template<typename register_t, typename vector_t>
    [[nodiscard]] consteval std::size_t elements_per_register()
    {
        return sizeof(register_t) / sizeof(vector_t);
    }
    template<typename register_t, typename vector_t>
    [[nodiscard]] consteval std::size_t elements_per_iteration()
    {
        return elements_per_register<register_t, vector_t>() * 2;
    }
    double get_register_sum(__m256d reg)
    {
        // register == [a, b, c, d]

        // https://web.archive.org/web/20250324225414/https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-9/mm256-castpd256-pd128.html
        __m128d low  = _mm256_castpd256_pd128(reg); // registerLow = [a, b]
        // https://web.archive.org/web/20250427071932/https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-10/mm256-extractf128-ps.html
        __m128d high = _mm256_extractf128_pd(reg, 1);   // registerHigh = [c, d]

        // https://intel-intrinsics.dpldocs.info/v1.11.0/inteli.emmintrin._mm_add_pd.html
        low = _mm_add_pd(low, high);    // [a+c, b+d]

        // https://intel-intrinsics.dpldocs.info/inteli.emmintrin._mm_unpackhi_pd.html
        __m128d copy = _mm_unpackhi_pd(low, low);   // [b+d, b+d]

        // https://intel-intrinsics.dpldocs.info/inteli.emmintrin._mm_add_sd.html
        low = _mm_add_sd(low, copy);    // [a+c+b+d, *]

        // https://intel-intrinsics.dpldocs.info/inteli.emmintrin._mm_cvtsd_f64.html
        return _mm_cvtsd_f64(low);
    }
    void calc_sumx_and_sumxx_avx2(double* pX, std::int64_t numElements, double* pOutSumX, double* pOutSumXX)
    {
        // _mm256_load_pd requires alignment on 32 bytes
        assert(reinterpret_cast<std::intptr_t>(pX) % 32 == 0);

        __m256d sumX = _mm256_setzero_pd();
        __m256d sumXX = _mm256_setzero_pd();

        static constexpr std::int64_t n = elements_per_register<__m256d, double>();
        for (int64_t offset = 0; offset < numElements; offset += n)
        {
            __m256d x = _mm256_load_pd(pX + offset);
            sumX = _mm256_add_pd(sumX, x);
#ifdef __FMA__
            sumXX = _mm256_fmadd_pd(x, x, sumXX);
#else
            sumXX = _mm256_add_pd(sumXX, _mm256_mul_pd(x, x));
#endif
        }
        *pOutSumX   = get_register_sum(sumX);
        *pOutSumXX = get_register_sum(sumXX);
    }
    [[nodiscard]] double calc_sumxy_avx2(const double* pX, const double* pY, std::int64_t numElements)
    {
        // _mm256_load_pd requires alignment on 32 bytes
        assert(reinterpret_cast<std::intptr_t>(pX) % 32 == 0);
        assert(reinterpret_cast<std::intptr_t>(pY) % 32 == 0);
        assert(numElements % 4 == 0);

        __m256d sum = _mm256_setzero_pd();
        static constexpr std::int64_t n = elements_per_register<__m256d, double>();
        for (int64_t offset = 0; offset < numElements; offset += n)
        {
            __m256d x = _mm256_load_pd(pX + offset);
            __m256d y = _mm256_load_pd(pY + offset);
#ifdef __FMA__
            sum = _mm256_fmadd_pd(x, y, sum);
#else
            sum = _mm256_add_pd(sum, _mm256_mul_pd(x, y));
#endif
        }

        return get_register_sum(sum);
    }
    [[nodiscard]] auto make_task_precompute(
        Latch& latch,
        std::size_t rangeStart,
        std::size_t rangeEnd,
        double* pData,
        std::int64_t stride,
        std::vector<double>& SUMX,
        std::vector<double>& SUMXX)
    {
        return [rangeStart, rangeEnd, &SUMX, &SUMXX, pData, stride, &latch]
        {
            for(std::size_t i = rangeStart; i < rangeEnd ; ++i)
            {
                double sumX;
                double sumXX;
                double* pVector = pData + i * stride;
                calc_sumx_and_sumxx_avx2(pVector, stride, std::addressof(sumX), std::addressof(sumXX));
                SUMX[i] = sumX;
                SUMXX[i] = sumXX;
            }
            latch.count_down();
        };
    }
    [[nodiscard]] auto make_task_main_compute(
        Latch& latch,
        std::size_t numVectors,
        std::size_t rangeStart,
        std::size_t rangeEnd,
        double* pData,
        std::int64_t stride,
        std::vector<double>& SUMX,
        std::vector<double>& SUMXX,
        std::vector<double>& results)
    {
        return [rangeStart, rangeEnd, pData, stride, numVectors, &latch, &results, &SUMX, &SUMXX]
        {
            auto dStride = static_cast<double>(stride);
            for(size_t xi = rangeStart; xi < rangeEnd; ++xi)
            {
                double* pX = pData + xi * stride;
                double sumX = SUMX[xi];
                double sumXX = SUMXX[xi];
                for(size_t yi = xi + 1; yi < numVectors; ++yi)
                {
                    double* pY = pData + yi * stride;
                    double sumXY = calc_sumxy_avx2(pX, pY, stride);

                    double sumY = SUMX[yi];
                    double sumYY = SUMXX[yi];
                    double numerator = (dStride * sumXY) - (sumX * sumY);
                    double denominator = ((dStride * sumXX) - (sumX * sumX)) * ((dStride * sumYY) - (sumY * sumY));

                    results[to_index(xi, yi, numVectors)] = numerator / std::sqrt(denominator);
                }
            }

            latch.count_down();
        };
    }
}   // namespace
namespace analysis
{
    std::vector<double> correlation_coefficients(ThreadPool& tp, double* pData, std::size_t size, std::int64_t stride)
    {
        assert(stride == 128 || stride == 256 || stride == 512 || stride == 1024);
        assert(size % stride == 0);

        std::size_t numVectors = size / static_cast<std::size_t>(stride);
        std::vector<double> SUMX(numVectors);
        std::vector<double> SUMXX(numVectors);

        // Pre compute pass
        {
            // Fork-join precompute per-vector sums
            std::size_t vectorsPerRange = 32;
            auto numTasks = static_cast<std::int64_t>(numVectors / vectorsPerRange);
            Latch latch{ numTasks };
            for(std::size_t rangeStart = 0 ; rangeStart < numVectors; rangeStart += vectorsPerRange)
            {
                std::size_t rangeEnd = std::min(numVectors, rangeStart + vectorsPerRange);
                tp.add_task(make_task_precompute(latch, rangeStart, rangeEnd, pData, stride, SUMX, SUMXX));
            }

            // Wait for precompute
            latch.wait();
        }


        const size_t numResults = numVectors * (numVectors - 1) / 2;
        std::vector<double> results(numResults);

        std::size_t vectorsPerRange = 16;
        auto numTasks = static_cast<std::int64_t>(numVectors / vectorsPerRange);
        Latch latch{ numTasks };
        for(size_t rangeStart = 0; rangeStart < numVectors; rangeStart += vectorsPerRange)
        {
            size_t rangeEnd = std::min(numVectors, rangeStart + vectorsPerRange);
            tp.add_task(make_task_main_compute(latch, numVectors, rangeStart, rangeEnd, pData, stride, SUMX, SUMXX, results));
        }

        // Wait for main compute
        latch.wait();

        return results;
    }
}   // namespace analysis
