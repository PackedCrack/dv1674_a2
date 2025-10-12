#include "ThreadPool.hpp"

///////////////////////////////////
///        Mutex           ///
///////////////////////////////////
Mutex::Mutex()
    : m_Mutex(PTHREAD_MUTEX_INITIALIZER)
{}
Mutex::~Mutex()
{
    pthread_mutex_destroy(std::addressof(m_Mutex));
}
bool Mutex::lock()
{
    return pthread_mutex_lock(std::addressof(m_Mutex)) == 0;
}
bool Mutex::unlock()
{
    return pthread_mutex_unlock(std::addressof(m_Mutex)) == 0;
}
pthread_mutex_t* Mutex::handle()
{
    return std::addressof(m_Mutex);
}
///////////////////////////////////
///        ConditionVariable           ///
///////////////////////////////////
ConditionVariable::ConditionVariable()
    : m_ConditionVariable(PTHREAD_COND_INITIALIZER)
{}
ConditionVariable::~ConditionVariable()
{
    pthread_cond_destroy(std::addressof(m_ConditionVariable));
}
bool ConditionVariable::broadcast()
{
    return pthread_cond_broadcast(std::addressof(m_ConditionVariable)) == 0;
}
bool ConditionVariable::signal()
{
    return pthread_cond_signal(std::addressof(m_ConditionVariable)) == 0;
}
bool ConditionVariable::wait(Mutex& mutex)
{
    // Spurious wakeups from the pthread_cond_timedwait() or pthread_cond_wait() functions may occur.
    // https://linux.die.net/man/3/pthread_cond_wait
    return pthread_cond_wait(std::addressof(m_ConditionVariable), mutex.handle()) == 0;
}
///////////////////////////////////
///        LockGuard           ///
///////////////////////////////////
LockGuard::LockGuard(Mutex& mutex)
    : m_pMutex{ std::addressof(mutex) }
{
    m_pMutex->lock();
}
LockGuard::~LockGuard()
{
    m_pMutex->unlock();
}
///////////////////////////////////
///           Latch           ///
///////////////////////////////////
Latch::Latch(std::int64_t size)
    : m_Count{ size }
    , m_Mutex{}
    , m_CV{}
{}
void Latch::count_down()
{
    // https://en.cppreference.com/w/cpp/atomic/memory_order.html
    // https://en.cppreference.com/w/cpp/atomic/atomic_fetch_sub.html
    if (std::atomic_fetch_sub_explicit(std::addressof(m_Count), 1, std::memory_order_acq_rel) == 1)
    {
        LockGuard lock{ m_Mutex };
        m_CV.broadcast();
    }
}
void Latch::wait()
{
    if (std::atomic_load_explicit(std::addressof(m_Count), std::memory_order_acquire) == 0)
    {
        return;
    }

    LockGuard lock{ m_Mutex };
    while (std::atomic_load_explicit(std::addressof(m_Count), std::memory_order_acquire) != 0)
    {
        [[maybe_unused]] bool success = m_CV.wait(m_Mutex);
    }
}
///////////////////////////////////
///           ThreadPool        ///
///////////////////////////////////
void* ThreadPool::peon(ThreadPool* pSelf)
{
    while(true)
    {
        std::function<void()> task{};
        {
            LockGuard lock{ pSelf->m_Mutex };
            while(pSelf->m_TaskQ.empty() && !pSelf->m_Exit)
            {
                [[maybe_unused]] bool success = pSelf->m_CV.wait(pSelf->m_Mutex);
            }
            if(pSelf->m_TaskQ.empty() && pSelf->m_Exit)
            {
                break;
            }
            task = std::move(pSelf->m_TaskQ.front()); pSelf->m_TaskQ.pop();
        }
        task();
    }

    return nullptr;
}
ThreadPool::ThreadPool(std::int32_t numThreads)
    : m_Mutex{}
    , m_CV{}
    , m_TaskQ{}
    , m_Exit{ false }
    , m_Threads{}
{
    assert(numThreads > 0);

    for(std::int32_t i = 0; i < numThreads; ++i)
    {
        m_Threads.emplace_back(ThreadPool::peon, this);
    }
}
ThreadPool::~ThreadPool()
{
    {
        LockGuard lock{ m_Mutex };
        m_Exit = true;
        m_CV.broadcast();
    }

    m_Threads.clear();
}
void ThreadPool::add_task(std::function<void()> task)
{
    LockGuard lock{ m_Mutex };
    m_TaskQ.push(std::move(task));
    m_CV.signal();
}
std::size_t ThreadPool::thread_count() const
{
    return m_Threads.size();
}