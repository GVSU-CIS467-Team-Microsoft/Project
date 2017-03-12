#include "../include/imebra/memoryPool.h"

#include "../implementation/memoryImpl.h"

namespace imebra
{

void MemoryPool::flush()
{
    implementation::memoryPoolGetter::getMemoryPoolGetter().getMemoryPoolLocal().flush();
}

size_t MemoryPool::getUnusedMemorySize()
{
    return implementation::memoryPoolGetter::getMemoryPoolGetter().getMemoryPoolLocal().getUnusedMemorySize();
}

void MemoryPool::setMemoryPoolSize(size_t minMemoryBlockSize, size_t maxMemoryPoolSize)
{
    implementation::memoryPoolGetter::getMemoryPoolGetter().getMemoryPoolLocal().setMinMaxMemory(minMemoryBlockSize, maxMemoryPoolSize);
}

}
