#include "simd_analysis.hpp"
#include "ThreadPool.hpp"

#include <cstdint>
#include <charconv>
#include <cassert>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <vector>
#include <span>
#include <fstream>
#include <cstring>


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
    struct Data
    {
        using element_t = double;

        std::vector<element_t, aligned_allocator<element_t, 32>> buffer;
        std::int64_t stride;
    };
    [[nodiscard]] const char* extract_dimension(const char* pFileBegin, const char* pFileEnd, std::int64_t* outDimensions)
    {
        // Read until newline boundary.
        const char* pDimensionEnd = pFileBegin;
        while (pDimensionEnd < pFileEnd && *pDimensionEnd != '\n')
            ++pDimensionEnd;

        auto result = std::from_chars(pFileBegin, pDimensionEnd, *outDimensions);
        assert(result.ec == std::errc{});

        return pDimensionEnd;
    }
    [[nodiscard]] const char* skip_whitespace(const char* p, const char* end)
    {
        while (p < end && static_cast<unsigned char>(*p) <= ' ')
        {
            ++p;
        }
        return p;
    }
    [[nodiscard]] Data load_data(char** argv)
    {
        char* filePath = argv[1];
        File f{ filePath };
        if (!f.valid())
        {
            std::printf("\nFailed to open: %s", filePath);
            std::exit(1);
        }
        assert(f.size() > 0);

        const char* pFileBegin = f.begin();
        const char* pFileEnd = f.end();

        Data data{};
        const char* pDimensionEnd = extract_dimension(pFileBegin, pFileEnd, std::addressof(data.stride));
        data.buffer.resize(static_cast<std::size_t>(data.stride) * static_cast<std::size_t>(data.stride));

        const char* pData = pDimensionEnd;
        for (auto&& value : data.buffer)
        {
            pData = skip_whitespace(pData, pFileEnd);
            if (pData >= pFileEnd)
            {
                break;
            }

            auto result = std::from_chars(pData, pFileEnd, value, std::chars_format::general);
            assert(result.ec == std::errc{});
            pData = result.ptr;
        }

        return data;
    }
    void save_results(char** argv, std::span<const double> results)
    {
        // The File RAII wrapper wasn't written with writing in mind so its easier to just do this raw..
        std::int32_t fileDescriptor = open(argv[2], O_CREAT | O_TRUNC | O_RDWR, 0644);
        assert(fileDescriptor >= 0);

        static constexpr std::size_t maxLineSize = 128;
        std::size_t maxFilesize = results.size() * maxLineSize;
        ftruncate(fileDescriptor, static_cast<off_t>(maxFilesize));

        void* pMap = mmap(nullptr, maxFilesize, PROT_READ | PROT_WRITE, MAP_SHARED, fileDescriptor, 0);
        assert(pMap != MAP_FAILED);

        char* pWrite = static_cast<char*>(pMap);
        char* const pEnd  = pWrite + maxFilesize;

        for (auto&& result : results)
        {
            // Write number. Leave room for '\n'.
            auto [ptr, ec] = std::to_chars(pWrite, pEnd - 1, result);
            assert(ec == std::errc{});
            *ptr = '\n';
            ++ptr;
            pWrite = ptr;
        }

        munmap(pMap, maxFilesize);

        auto realSize = static_cast<std::ptrdiff_t>(pWrite - static_cast<char*>(pMap));
        ftruncate(fileDescriptor, static_cast<off_t>(realSize));
        close(fileDescriptor);
    }
}   // namespace
int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::printf("Usage: ./program [input] [output] [thread_count]");
        return 1;
    }

    Data data = load_data(argv);

    std::int32_t threadCount = std::stoi(argv[3]);
    ThreadPool tp{ threadCount };
    std::vector<double> results =
        analysis::correlation_coefficients(tp, data.buffer.data(), data.buffer.size(), data.stride);

    save_results(argv, results);

    return 0;
}
