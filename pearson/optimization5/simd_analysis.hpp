#pragma once

#include "ThreadPool.hpp"
#include <cstdint>

namespace analysis
{
    std::vector<double> correlation_coefficients(ThreadPool& tp, double* pData, std::size_t size, std::int64_t stride);
}   // namespace analysis
