#pragma once

#include <cstdint>
#include <atomic>
#include <queue>
#include <functional>
#include <cassert>
#include <concepts>
#include <type_traits>
#include <limits>
#include <pthread.h>


class Mutex
{
public:
    Mutex();
    ~Mutex();
    Mutex(const Mutex& other) = delete;
    Mutex(Mutex&& other) = delete;
    Mutex& operator=(const Mutex& other) = delete;
    Mutex& operator=(Mutex&& other) = delete;
public:
    bool lock();
    bool unlock();
    [[nodiscard]] pthread_mutex_t* handle();
private:
    pthread_mutex_t m_Mutex;
};
class ConditionVariable
{
public:
    ConditionVariable();
    ~ConditionVariable();
    ConditionVariable(const ConditionVariable& other) = delete;
    ConditionVariable(ConditionVariable&& other) = delete;
    ConditionVariable& operator=(const ConditionVariable& other) = delete;
    ConditionVariable& operator=(ConditionVariable&& other) = delete;
public:
    bool broadcast();
    bool signal();
    [[nodiscard]] bool wait(Mutex& mutex);
private:
    pthread_cond_t  m_ConditionVariable;
};
// https://en.cppreference.com/w/cpp/thread/lock_guard.html
// reimplementing this because apparently pthreads lmfao
class LockGuard
{
public:
    explicit LockGuard(Mutex& mutex);
    ~LockGuard();
    LockGuard(const LockGuard& other) = delete;
    LockGuard(LockGuard&& other) = delete;
    LockGuard& operator=(const LockGuard& other) = delete;
    LockGuard& operator=(LockGuard&& other) = delete;
private:
    Mutex* m_pMutex;
};
// https://en.cppreference.com/w/cpp/thread/latch.html
// reimplementing this because apparently pthreads lmfao
class Latch
{
public:
    Latch(std::int64_t size);
    ~Latch() = default;
    Latch(const Latch& other) = delete;
    Latch(Latch&& other) = delete;
    Latch& operator=(const Latch& other) = delete;
    Latch& operator=(Latch&& other) = delete;
public:
    void count_down();
    void wait();
private:
    // assume cache line 64 bytes
    // https://en.wikipedia.org/wiki/False_sharing
    alignas(64) std::atomic<std::int64_t> m_Count;    // I can't find anythign online about pthreads supporting atomics.. So i'm using the standard lbirary for this...
    Mutex m_Mutex;
    ConditionVariable  m_CV;
};
template<typename callable_t, typename arg_t>
requires std::invocable<std::decay_t<callable_t>&, std::decay_t<arg_t>&>
class Thread
{
    using StartAddress = std::decay_t<callable_t>;
    using Arguments = std::decay_t<arg_t>;
public:
    Thread(callable_t&& callable, arg_t&& arg)
        : m_Thread{}
        , PFN_start_address{ std::forward<callable_t>(callable) }
        , m_Args{ std::forward<arg_t>(arg) }
        , m_Executing{ false }
    {
        if (pthread_create(std::addressof(m_Thread), nullptr, std::addressof(Thread::trampoline), this) == 0)
        {
            m_Executing = true;
        }
    }
    ~Thread()
    {
        if (m_Executing)
        {
            pthread_join(m_Thread, nullptr);
        }
    }
    Thread(const Thread& other) = delete;
    // We use unstable this pointers - so no moving
    Thread(Thread&& other) = delete;
    Thread& operator=(const Thread& other) = delete;
    Thread& operator=(Thread&& other) = delete;
private:
    static void* trampoline(void* pThis) noexcept
    {
        auto* pSelf = static_cast<Thread*>(pThis);
        std::invoke(pSelf->PFN_start_address, pSelf->m_Args);

        return nullptr;
    }
private:
    pthread_t m_Thread;
    StartAddress PFN_start_address;
    Arguments m_Args;
    bool m_Executing;
};
class ThreadPool
{
    static void* peon(ThreadPool* pSelf);
    using Peon = Thread<decltype(&ThreadPool::peon), ThreadPool*>;
public:
    ThreadPool(std::int32_t numThreads);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    // We use unstable this pointers - so no moving
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;
public:
    void add_task(std::function<void()> task);
    [[nodiscard]] std::size_t thread_count() const;
private:
    Mutex m_Mutex;
    ConditionVariable  m_CV;
    std::queue<std::function<void()>> m_TaskQ;
    bool m_Exit;
    std::deque<Peon> m_Threads; // store this is linked list so Thread can remain non movable for simplicity
};