#pragma once

#include <vector>
#include <cstdint>

namespace analysis
{
    std::vector<double> correlation_coefficients(double* pData, std::size_t size, std::int64_t stride);
}   // namespace analysis
