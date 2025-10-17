#include "ThreadPool.hpp"
#include "gaussian.hpp"

#include <string>
#include <cstdio>
#include <charconv>
#include <cassert>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <span>


namespace
{
    std::string s_PpmHeader{};  // hack because i'm to lazy to write output logic for the PPM header.
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
    [[nodiscard]] auto make_task_store_rgb(gaussian::Image& image, std::int32_t taskStart, std::int32_t taskEnd, Latch& latch, const char* pFirstPixel)
    {
        return [
            &latch,
            &image,
            taskStart,
            taskEnd,
            pFirstPixel]()
        {
            static constexpr std::int32_t bytesPerPixel = 3;
            std::int32_t bytesPerRow = image.width * bytesPerPixel;
            for (std::int32_t i = taskStart; i < taskEnd; ++i)
            {
                const std::uint8_t* pSample = reinterpret_cast<const std::uint8_t*>(pFirstPixel) + i * bytesPerRow;
                std::size_t offset = i * image.width;
                for (std::int32_t x = 0; x < image.width; ++x)
                {
                    std::size_t index = offset + x;
                    image.red[index] = pSample[0];
                    image.green[index] = pSample[1];
                    image.blue[index] = pSample[2];
                    pSample += bytesPerPixel;
                }
            }
            latch.count_down();
        };
    }
    void store_non_interleaved_rgb_data(ThreadPool& tp, gaussian::Image& image, const char* pFirstPixel)
    {
        std::int32_t rowsPerTask = 64;
        std::int32_t numTasks = (image.height + rowsPerTask - 1) / rowsPerTask;
        Latch latch{ numTasks };
        for (std::int32_t i = 0; i < numTasks; ++i)
        {
            std::int32_t taskStart = i * rowsPerTask;
            std::int32_t taskEnd = std::min(image.height, taskStart + rowsPerTask);
            tp.add_task(make_task_store_rgb(image, taskStart, taskEnd, latch, pFirstPixel));
        }
        latch.wait();
    }
    [[nodiscard]] gaussian::Image load(ThreadPool& tp, const char* path)
    {
        File f{ path };
        const char* pFileContent = f.begin();
        pFileContent = skip_magic(pFileContent);
        pFileContent = skip_comments(pFileContent);

        gaussian::Image image{};
        pFileContent = extract_width(pFileContent, std::addressof(image.width));
        pFileContent = extract_height(pFileContent, std::addressof(image.height));
        pFileContent = extract_maxval(pFileContent, std::addressof(image.maxval));
        assert(image.maxval == 255);

        std::uintptr_t size = pFileContent - f.begin();
        s_PpmHeader = std::string{ f.begin(), size };

        image.red.resize(image.width * image.height);
        image.blue.resize(image.width * image.height);
        image.green.resize(image.width * image.height);

        store_non_interleaved_rgb_data(tp, image, pFileContent);

        return image;
    }
    void save_results(const char* filepath, std::span<const std::uint8_t> results)
    {
        std::int32_t fileDescriptor = open(filepath, O_CREAT | O_TRUNC | O_RDWR, 0644);
        assert(fileDescriptor >= 0);

        std::size_t filesize = s_PpmHeader.size() + results.size();
        ftruncate(fileDescriptor, static_cast<off_t>(filesize));

        void* pMap = mmap(nullptr, filesize, PROT_READ | PROT_WRITE, MAP_SHARED, fileDescriptor, 0);
        assert(pMap != MAP_FAILED);

        char* pWrite = static_cast<char*>(pMap);
        std::memcpy(pWrite, s_PpmHeader.data(), s_PpmHeader.size());
        pWrite += s_PpmHeader.size();
        std::memcpy(pWrite, results.data(), results.size());

        munmap(pMap, filesize);
        close(fileDescriptor);
    }
} // namespace
int main(int argc, char const** argv)
{
    std::int32_t numThreads = std::stoi(argv[4]);
    ThreadPool tp{ numThreads };
    gaussian::Image image = load(tp, argv[2]);

    std::int32_t radius = std::stoi(argv[1]);
    if (radius != 15)
    {
        std::printf("\nRadius must be 15 not %i", radius);
        return 1;
    }

    std::vector<std::uint8_t> result = gaussian::add_blur(tp, image, radius);
    save_results(argv[3], result);

    return 0;
}
