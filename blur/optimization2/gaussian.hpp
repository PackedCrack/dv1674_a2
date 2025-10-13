#pragma once

#include <cstdint>
#include <vector>

namespace gaussian
{
    template<class allocated_t, std::size_t aligned_as>
    struct aligned_allocator
    {
        using value_type = allocated_t;

        aligned_allocator() noexcept = default;
        template<class U>
        explicit aligned_allocator(const aligned_allocator<U, aligned_as>&) noexcept
        {}
        [[nodiscard]] allocated_t* allocate(std::size_t n)
        {
            return static_cast<allocated_t*>(::operator new(n * sizeof(allocated_t), std::align_val_t(aligned_as)));
        }
        void deallocate(allocated_t* p, std::size_t n) noexcept
        {
            ::operator delete(p, n * sizeof(allocated_t), std::align_val_t(aligned_as));
        }
        template<class U>
        struct rebind
        {
            using other = aligned_allocator<U, aligned_as>;
        };
        template<class allocated_t1, std::size_t alignment, class allocated_t2>
        friend bool operator==(const aligned_allocator<allocated_t1, alignment>&, const aligned_allocator<allocated_t2, alignment>&)
        {
            return true;
        }
        template<class allocated_t1, std::size_t alignment, class allocated_t2>
        friend bool operator!=(const aligned_allocator<allocated_t1, alignment>&, const aligned_allocator<allocated_t2, alignment>&)
        {
            return false;
        }
    };
    struct Image
    {
        using value_type = std::uint8_t;

        std::vector<value_type, aligned_allocator<value_type, 32>> red;
        std::vector<value_type, aligned_allocator<value_type, 32>> green;
        std::vector<value_type, aligned_allocator<value_type, 32>> blue;
        std::int32_t width;
        std::int32_t height;
        std::int32_t maxval;
    };
    void calculate(Image& image, std::int32_t radius);
}    // namespace gaussian