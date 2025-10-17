/*
Author: David Holmqvist <daae19@student.bth.se>
*/

#include <array>
#include "filters.hpp"
#include "matrix.hpp"
#include "ppm.hpp"
#include <cmath>
#include <cstdint>

namespace Filter
{

    namespace Gauss
    {
        void get_weights(int n, double *weights_out)
        {
            for (auto i{0}; i <= n; i++)
            {
                double x{static_cast<double>(i) * max_x / n};
                weights_out[i] = exp(-x * x * pi);
            }
        }
    }

    Matrix blur(Matrix m, const int radius)
    {
        Matrix scratch{PPM::max_dimension};
        auto dst{m};

        for (auto x{0}; x < dst.get_x_size(); x++)
        {
            for (auto y{0}; y < dst.get_y_size(); y++)
            {
                double w[Gauss::max_radius]{};
                Gauss::get_weights(radius, w);

                // unsigned char Matrix::r(unsigned x, unsigned y) const
                // {
                //     return R[y * x_size + x];
                // }

                auto r{w[0] * dst.r(x, y)}, g{w[0] * dst.g(x, y)}, b{w[0] * dst.b(x, y)}, n{w[0]};

                for (auto wi{1}; wi <= radius; wi++)
                {
                    auto wc{w[wi]};
                    auto x2{x - wi};
                    if (x2 >= 0)
                    {
                        r += wc * dst.r(x2, y);
                        g += wc * dst.g(x2, y);
                        b += wc * dst.b(x2, y);
                        n += wc;
                    }
                    x2 = x + wi;
                    if (x2 < dst.get_x_size())
                    {
                        r += wc * dst.r(x2, y);
                        g += wc * dst.g(x2, y);
                        b += wc * dst.b(x2, y);
                        n += wc;
                    }
                }
                scratch.r(x, y) = r / n;
                scratch.g(x, y) = g / n;
                scratch.b(x, y) = b / n;
            }
        }

        uint8_t* pR = (uint8_t*)scratch.R;
        uint8_t* pG = (uint8_t*)scratch.G;
        uint8_t* pB = (uint8_t*)scratch.B;
        std::array<uint8_t, 105> test2{};
        for (auto i = 0; i < test2.size(); ++i)
        {
            test2[i] = static_cast<uint8_t>(pR[i]);
        }
        std::array<uint8_t, 105> test3{};
        for (auto i = 0; i < test3.size(); ++i)
        {
            test3[i] = static_cast<uint8_t>(pG[i]);
        }
        std::array<uint8_t, 105> test4{};
        for (auto i = 0; i < test4.size(); ++i)
        {
            test4[i] = static_cast<uint8_t>(pB[i]);
        }

        for (auto x{0}; x < dst.get_x_size(); x++)
        {
            for (auto y{0}; y < dst.get_y_size(); y++)
            {
                double w[Gauss::max_radius]{};
                Gauss::get_weights(radius, w);

                auto r{w[0] * scratch.r(x, y)}, g{w[0] * scratch.g(x, y)}, b{w[0] * scratch.b(x, y)}, n{w[0]};

                for (auto wi{1}; wi <= radius; wi++)
                {
                    auto wc{w[wi]};
                    auto y2{y - wi};
                    if (y2 >= 0)
                    {
                        r += wc * scratch.r(x, y2);
                        g += wc * scratch.g(x, y2);
                        b += wc * scratch.b(x, y2);
                        n += wc;
                    }
                    y2 = y + wi;
                    if (y2 < dst.get_y_size())
                    {
                        r += wc * scratch.r(x, y2);
                        g += wc * scratch.g(x, y2);
                        b += wc * scratch.b(x, y2);
                        n += wc;
                    }
                }
                dst.r(x, y) = r / n;
                dst.g(x, y) = g / n;
                dst.b(x, y) = b / n;
            }
        }

        std::array<uint8_t, 20> test{};
        for (auto i = 0; i < test.size(); ++i)
        {
            test[i] = static_cast<uint8_t>(dst.R[i]);
        }

        return dst;
    }

}
