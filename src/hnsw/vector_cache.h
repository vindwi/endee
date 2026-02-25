#pragma once
#include "hnswlib.h"
#include "../utils/settings.hpp"
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <cstring>
#include <array>
#include <limits>
#include <cstdlib>
#include <string>
#include <utility>

namespace hnswlib {

class VectorCache {
public:
    class CacheReadHandle {
    public:
        CacheReadHandle() = default;
        CacheReadHandle(const CacheReadHandle&) = delete;
        CacheReadHandle& operator=(const CacheReadHandle&) = delete;
        CacheReadHandle(CacheReadHandle&&) = default;
        CacheReadHandle& operator=(CacheReadHandle&&) = default;

        const uint8_t* data() const { return data_; }
        size_t size() const { return size_; }
        explicit operator bool() const { return data_ != nullptr; }

    private:
        friend class VectorCache;

        CacheReadHandle(std::shared_lock<std::shared_mutex>&& lock,
                        const uint8_t* data,
                        size_t size)
            : lock_(std::move(lock)), data_(data), size_(size) {}

        std::shared_lock<std::shared_mutex> lock_;
        const uint8_t* data_ = nullptr;
        size_t size_ = 0;
    };

    inline static size_t VECTOR_CACHE_PERCENTAGE = settings::VECTOR_CACHE_PERCENTAGE;

    inline static size_t VECTOR_CACHE_MIN_BITS = settings::VECTOR_CACHE_MIN_BITS;
    // Helper to calculate required cache bits based on element count and percentage
    static size_t calculateCacheBits(size_t element_count, size_t cache_percent = VECTOR_CACHE_PERCENTAGE) {
        if (element_count == 0 || cache_percent == 0) return 0;
        
        size_t target_elements = (element_count * cache_percent) / 100;
        
        // Calculate bits needed: 2^bits >= target_elements
        size_t bits = 0;
        while ((1ULL << bits) < target_elements) {
            bits++;
        }
        
        // Enforce minimum bits
        if (bits < VECTOR_CACHE_MIN_BITS) {
            bits = VECTOR_CACHE_MIN_BITS;
        }

        return bits;
    }

private:
    size_t cacheBits_ = 0;
    size_t cacheSize_ = 0;
    size_t cacheMask_ = 0;
    size_t vectorCacheDataSize_ = 0;
    size_t vectorCacheEntrySize_ = 0;
    size_t data_size_ = 0;
    uint8_t* vectorCache_ = nullptr;
    
    static constexpr size_t CACHE_STRIPE_BITS = 10; // 1024 stripes
    static constexpr size_t CACHE_STRIPE_COUNT = 1 << CACHE_STRIPE_BITS;
    static constexpr size_t CACHE_STRIPE_MASK = CACHE_STRIPE_COUNT - 1;
    mutable std::array<std::shared_mutex, CACHE_STRIPE_COUNT> vectorCacheStripeMutexes_;
    
    static constexpr idInt INVALID_ID = static_cast<idInt>(-1);

    uint8_t* getEntryPtr(size_t index) const {
        return vectorCache_ + index * vectorCacheEntrySize_;
    }

    idInt* getEntryIdPtr(uint8_t* entry) const {
        return reinterpret_cast<idInt*>(entry);
    }

    uint8_t* getEntryLifePtr(uint8_t* entry) const {
        return entry + sizeof(idInt);
    }

    uint8_t* getEntryDataPtr(uint8_t* entry) const {
        return entry + sizeof(idInt) + sizeof(uint8_t);
    }

    std::shared_mutex& getCacheStripeMutex(size_t cache_index) const {
        size_t stripe_id = cache_index & CACHE_STRIPE_MASK;
        return vectorCacheStripeMutexes_[stripe_id];
    }

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
    }
    
    void init(size_t data_size, size_t cache_bits) {
        if (vectorCache_) {
            delete[] vectorCache_;
            vectorCache_ = nullptr;
        }

        if (cache_bits == 0) {
            cacheBits_ = 0;
            cacheSize_ = 0;
            cacheMask_ = 0;
            data_size_ = 0;
            vectorCacheDataSize_ = 0;
            vectorCacheEntrySize_ = 0;
            return;
        }

        data_size_ = data_size;
        cacheBits_ = cache_bits;
        cacheSize_ = 1 << cacheBits_;
        cacheMask_ = cacheSize_ - 1;
        vectorCacheDataSize_ = data_size_ + sizeof(idInt);
        vectorCacheEntrySize_ = sizeof(idInt) + sizeof(uint8_t) + data_size_;
        
        vectorCache_ = new uint8_t[cacheSize_ * vectorCacheEntrySize_];
        
        // Initialize all entries to INVALID_ID with one-life flag cleared
        for (size_t i = 0; i < cacheSize_; i++) {
            uint8_t* entry = getEntryPtr(i);
            idInt* id_ptr = getEntryIdPtr(entry);
            uint8_t* life_ptr = getEntryLifePtr(entry);
            *id_ptr = INVALID_ID;
            *life_ptr = 0;
        }
    }
    
    bool get(idInt internal_id, uint8_t* buffer) const {
        if (!vectorCache_) return false;
        
        size_t index = internal_id & cacheMask_;
        uint8_t* entry = getEntryPtr(index);
        
        std::shared_lock<std::shared_mutex> lock(getCacheStripeMutex(index));
        
        idInt* stored_id = getEntryIdPtr(entry);
        if (*stored_id == internal_id) {
            memcpy(buffer, getEntryDataPtr(entry), data_size_);
            return true;
        }
        return false;
    }

    bool getReadHandle(idInt internal_id, CacheReadHandle& out_handle) const {
        out_handle = CacheReadHandle();
        if (!vectorCache_) return false;

        size_t index = internal_id & cacheMask_;
        uint8_t* entry = getEntryPtr(index);

        std::shared_lock<std::shared_mutex> lock(getCacheStripeMutex(index));

        idInt* stored_id = getEntryIdPtr(entry);
        if (*stored_id == internal_id) {
            out_handle = CacheReadHandle(std::move(lock), getEntryDataPtr(entry), data_size_);
            return true;
        }

        return false;
    }
    
    void insert(idInt internal_id, const uint8_t* data) {
        if (!vectorCache_) return;
        
        size_t index = internal_id & cacheMask_;
        uint8_t* entry = getEntryPtr(index);
        
        std::unique_lock<std::shared_mutex> lock(getCacheStripeMutex(index), std::try_to_lock);
        if (!lock.owns_lock()) {
            return;
        }
        
        idInt* stored_id = getEntryIdPtr(entry);
        uint8_t* life_ptr = getEntryLifePtr(entry);
        uint8_t* entry_data = getEntryDataPtr(entry);

        // same id: refresh payload and restore one life
        if (*stored_id == internal_id) {
            memcpy(entry_data, data, data_size_);
            *life_ptr = 1;
            return;
        }

        // empty slot: insert and give one life
        if (*stored_id == INVALID_ID) {
            *stored_id = internal_id;
            memcpy(entry_data, data, data_size_);
            *life_ptr = 1;
            return;
        }

        // conflict: one life to existing occupant, then replacement on next conflict
        if (*life_ptr != 0) {
            *life_ptr = 0;
            return;
        }

        *stored_id = internal_id;
        memcpy(entry_data, data, data_size_);
        *life_ptr = 1;
    }
    
    size_t getCacheBits() const { return cacheBits_; }
    size_t getCacheSize() const { return cacheSize_; }
    void setCacheBits(size_t bits) { cacheBits_ = bits; }
    
    size_t getMemoryUsage() const {
        if (!vectorCache_) return 0;
        return cacheSize_ * vectorCacheEntrySize_;
    }
};

} // namespace hnswlib
