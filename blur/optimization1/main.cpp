

#include "matrix.hpp"
#include "ppm.hpp"
#include "filters.hpp"

#include <cstdlib>
#include <cstdint>
#include <vector>
#include <charconv>
#include <cassert>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>


namespace
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
    class File
    {
    public:
        explicit File(const char* path)
            : m_pBegin{ nullptr }
            , m_FileDescriptor{ open(path, O_RDONLY) }
            , m_FileStats{}
        {
            if (valid())
            {
                if (fstat(m_FileDescriptor, std::addressof(m_FileStats)))
                {
                    std::printf("\nfstat failed");
                    close(m_FileDescriptor);
                    std::exit(1);
                }
#ifdef POSIX_FADV_SEQUENTIAL
                posix_fadvise(m_FileDescriptor, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
                m_pBegin = mmap(nullptr, size(), PROT_READ, MAP_PRIVATE, m_FileDescriptor, 0);
                if (m_pBegin == MAP_FAILED)
                {
                    std::printf("\nmmap failed");
                    std::exit(1);
                }
            }
        }
        ~File()
        {
            if (valid())
            {
                close(m_FileDescriptor);
            }
            if (m_pBegin != MAP_FAILED)
            {
                munmap(m_pBegin, size());
            }
        }
        File(const File&) = delete;
        File& operator=(const File&) = delete;
        [[nodiscard]] bool valid() const
        {
            return m_FileDescriptor >= 0;
        }
        [[nodiscard]] std::uint64_t size() const
        {
            return static_cast<std::uint64_t>(m_FileStats.st_size);
        }
        [[nodiscard]] const char* begin() const
        {
            return static_cast<char*>(m_pBegin);
        }
        [[nodiscard]] const char* end() const
        {
            return static_cast<char*>(m_pBegin) + size();
        }
    private:
        void* m_pBegin = nullptr;
        std::int32_t m_FileDescriptor = -1;
        struct stat m_FileStats;
    };
    struct Image
    {
        using element_t = double;

        std::vector<element_t, aligned_allocator<element_t, 32>> red;
        std::vector<element_t, aligned_allocator<element_t, 32>> green;
        std::vector<element_t, aligned_allocator<element_t, 32>> blue;
        std::int64_t stride;
        std::int32_t width;
        std::int32_t height;
        std::int32_t maxval;
    };

    [[nodiscard]] const char* skip_magic(const char* pFileContent)
    {
        assert(*pFileContent == 'P');
        while (*pFileContent != '\n')
        {
            ++pFileContent;
        }
        ++pFileContent;

        return pFileContent;
    }
    [[nodiscard]] const char* skip_comments(const char* pFileContent)
    {
        if (*pFileContent == '#')
        {
            while (*pFileContent != '\n')
            {
                ++pFileContent;
            }
            ++pFileContent;
        }

        return pFileContent;
    }
    [[nodiscard]] const char* extract_value(const char* pFileContent, std::int32_t* pOutValue)
    {
        const char* pEnd = pFileContent;
        while(*pEnd != '\n' && *pEnd != ' ')
        {
            ++pEnd;
        }
        ++pEnd;

        auto [ptr, ec] = std::from_chars(pFileContent, pEnd, *pOutValue);
        assert(ec == std::errc{});

        return pEnd;
    }
    [[nodiscard]] const char* extract_width(const char* pFileContent, std::int32_t* pOutWidth)
    {
        return extract_value(pFileContent, pOutWidth);
    }
    [[nodiscard]] const char* extract_height(const char* pFileContent, std::int32_t* pOutHeight)
    {
        return extract_value(pFileContent, pOutHeight);
    }
    [[nodiscard]] const char* extract_maxval(const char* pFileContent, std::int32_t* pOutMaxVal)
    {
        return extract_value(pFileContent, pOutMaxVal);
    }
    [[nodiscard]] Image load(const char* path)
    {
        File f{ path };
        const char* pFileContent = f.begin();
        pFileContent = skip_magic(pFileContent);
        pFileContent = skip_comments(pFileContent);

        Image image{};
        pFileContent = extract_width(pFileContent, std::addressof(image.width));
        pFileContent = extract_height(pFileContent, std::addressof(image.height));
        pFileContent = extract_maxval(pFileContent, std::addressof(image.maxval));

        int a = 10;


    }
} // namespace
int main(int argc, char const** argv)
{
    //if (argc != 4) {
    //    std::cerr << "Usage: " << argv[0] << " [radius] [infile] [outfile]" << std::endl;
    //    std::exit(1);
    //}
    Image image = load(argv[2]);

    return 0;

    PPM::Reader reader {};
    PPM::Writer writer {};

    auto m { reader(argv[2]) };
    auto radius { static_cast<unsigned>(std::stoul(argv[1])) };

    auto blurred { Filter::blur(m, radius) };
    writer(blurred, argv[3]);

    return 0;
}
