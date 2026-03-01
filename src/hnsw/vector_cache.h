#pragma once
#include "hnswlib.h"
#include "../utils/settings.hpp"
#include <vector>
#include <mutex>
#include <atomic>
#include <cstring>
#include <limits>
#include <cstdlib>
#include <string>

namespace hnswlib {

class VectorCache {
public:

    // Helper to calculate required cache bits based on element count and percentage
    static size_t calculateCacheBits(size_t element_count, size_t cache_percent = settings::VECTOR_CACHE_PERCENTAGE) {
        if (element_count == 0 || cache_percent == 0) return 0;
        
        size_t target_elements = (element_count * cache_percent) / 100;
        
        // Calculate bits needed: 2^bits >= target_elements
        size_t cache_bits = 0;
        while ((1ULL << cache_bits) < target_elements) {
            cache_bits++;
        }
        
        // Enforce minimum bits
        if (cache_bits < settings::VECTOR_CACHE_MIN_BITS) {
            cache_bits = settings::VECTOR_CACHE_MIN_BITS;
        }

        return cache_bits;
    }

private:
    size_t cacheBits_ = 0;
    size_t cacheSize_ = 0;
    size_t cacheMask_ = 0;
    size_t vectorCacheDataSize_ = 0;
    size_t data_size_ = 0;
    uint8_t* vectorCache_ = nullptr;
    std::atomic<uint8_t>* slotLife_ = nullptr;
    
    static constexpr idInt INVALID_ID = static_cast<idInt>(-1);

public:
    VectorCache() = default;
    
    // Constructor with initialization
    VectorCache(size_t data_size, size_t cache_bits) {
        init(data_size, cache_bits);
    }
    
    ~VectorCache() {
        if (vectorCache_) {
            delete[] vectorCache_;
            vectorCache_ = nullptr;
        }
        if (slotLife_) {
            delete[] slotLife_;
            slotLife_ = nullptr;
        }
    }
    
    void init(size_t data_size, size_t cache_bits) {
        if (vectorCache_) {
            delete[] vectorCache_;
            vectorCache_ = nullptr;
        }
        if (slotLife_) {
            delete[] slotLife_;
            slotLife_ = nullptr;
        }

        if (cache_bits == 0) {
            cacheBits_ = 0;
            cacheSize_ = 0;
            cacheMask_ = 0;
            data_size_ = 0;
            vectorCacheDataSize_ = 0;
            return;
        }

        data_size_ = data_size;
        cacheBits_ = cache_bits;
        cacheSize_ = 1 << cacheBits_;
        cacheMask_ = cacheSize_ - 1;
        vectorCacheDataSize_ = data_size_ + sizeof(idInt);
        
        vectorCache_ = new uint8_t[cacheSize_ * vectorCacheDataSize_];
        slotLife_ = new std::atomic<uint8_t>[cacheSize_];
        
        // Initialize all entries to INVALID_ID
        for (size_t i = 0; i < cacheSize_; i++) {
            idInt* id_ptr = reinterpret_cast<idInt*>(vectorCache_ + i * vectorCacheDataSize_);
            *id_ptr = INVALID_ID;
            slotLife_[i].store(0, std::memory_order_relaxed);
        }
    }
    
    size_t getCacheIndex(idInt internal_id) const {
        return internal_id & cacheMask_;
    }

    // Not thread-safe. Caller must hold the appropriate cache stripe lock.
    const uint8_t* getPointer(idInt internal_id) {
        if (!vectorCache_ || !slotLife_) return nullptr;

        size_t index = getCacheIndex(internal_id);
        uint8_t* entry = vectorCache_ + index * vectorCacheDataSize_;
        idInt* stored_id = reinterpret_cast<idInt*>(entry);
        if (*stored_id != internal_id) {
            return nullptr;
        }

        slotLife_[index].store(1, std::memory_order_relaxed);
        return entry + sizeof(idInt);
    }

    // Not thread-safe. Caller must hold the appropriate cache stripe lock.
    const uint8_t* getPointer(idInt internal_id) const {
        return const_cast<VectorCache*>(this)->getPointer(internal_id);
    }

    // Not thread-safe. Caller must hold the appropriate cache stripe lock.
    void insert(idInt internal_id, const uint8_t* data) {
        if (!vectorCache_ || !slotLife_) return;
        
        size_t index = getCacheIndex(internal_id);
        uint8_t* entry = vectorCache_ + index * vectorCacheDataSize_;

        idInt* stored_id = reinterpret_cast<idInt*>(entry);

        // Same id: refresh value and restore life.
        if (*stored_id == internal_id) {
            memcpy(entry + sizeof(idInt), data, data_size_);
            slotLife_[index].store(1, std::memory_order_relaxed);
            return;
        }

        // Different id: one-life policy.
        // life == 1 -> give one life (flip to 0, do not replace now)
        // life == 0 -> replace slot and set life back to 1
        uint8_t life = slotLife_[index].load(std::memory_order_relaxed);
        if (life != 0) {
            slotLife_[index].store(0, std::memory_order_relaxed);
            return;
        }

        *stored_id = internal_id;
        memcpy(entry + sizeof(idInt), data, data_size_);
        slotLife_[index].store(1, std::memory_order_relaxed);
    }

    // Not thread-safe. Caller must hold the appropriate cache stripe lock.
    void update(idInt internal_id, const uint8_t* data) {
        if(!vectorCache_ || !slotLife_) return;

        size_t index = getCacheIndex(internal_id);
        uint8_t* entry = vectorCache_ + index * vectorCacheDataSize_;
        idInt* stored_id = reinterpret_cast<idInt*>(entry);

        if(*stored_id != internal_id) {
            return;
        }

        *stored_id = internal_id;
        memcpy(entry + sizeof(idInt), data, data_size_);
        slotLife_[index].store(1, std::memory_order_relaxed);
    }
    
    size_t getCacheBits() const { return cacheBits_; }
    size_t getCacheSize() const { return cacheSize_; }
    void setCacheBits(size_t bits) { cacheBits_ = bits; }
    
    size_t getMemoryUsage() const {
        if (!vectorCache_) return 0;
        return cacheSize_ * vectorCacheDataSize_;
    }
};

} // namespace hnswlib
