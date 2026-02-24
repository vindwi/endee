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

namespace hnswlib {

class VectorCache {
public:
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
    size_t data_size_ = 0;
    uint8_t* vectorCache_ = nullptr;
    
    static constexpr size_t CACHE_STRIPE_BITS = 8; // 256 stripes
    static constexpr size_t CACHE_STRIPE_COUNT = 1 << CACHE_STRIPE_BITS;
    static constexpr size_t CACHE_STRIPE_MASK = CACHE_STRIPE_COUNT - 1;
    mutable std::array<std::shared_mutex, CACHE_STRIPE_COUNT> vectorCacheStripeMutexes_;
    
    static constexpr idInt INVALID_ID = static_cast<idInt>(-1);

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
            return;
        }

        data_size_ = data_size;
        cacheBits_ = cache_bits;
        cacheSize_ = 1 << cacheBits_;
        cacheMask_ = cacheSize_ - 1;
        vectorCacheDataSize_ = data_size_ + sizeof(idInt);
        
        vectorCache_ = new uint8_t[cacheSize_ * vectorCacheDataSize_];
        
        // Initialize all entries to INVALID_ID
        for (size_t i = 0; i < cacheSize_; i++) {
            idInt* id_ptr = reinterpret_cast<idInt*>(vectorCache_ + i * vectorCacheDataSize_);
            *id_ptr = INVALID_ID;
        }
    }
    
    bool get(idInt internal_id, uint8_t* buffer) const {
        if (!vectorCache_) return false;
        
        size_t index = internal_id & cacheMask_;
        uint8_t* entry = vectorCache_ + index * vectorCacheDataSize_;
        
        std::shared_lock<std::shared_mutex> lock(getCacheStripeMutex(index));
        
        idInt* stored_id = reinterpret_cast<idInt*>(entry);
        if (*stored_id == internal_id) {
            memcpy(buffer, entry + sizeof(idInt), data_size_);
            return true;
        }
        return false;
    }
    
    void insert(idInt internal_id, const uint8_t* data) {
        if (!vectorCache_) return;
        
        size_t index = internal_id & cacheMask_;
        uint8_t* entry = vectorCache_ + index * vectorCacheDataSize_;
        
        std::unique_lock<std::shared_mutex> lock(getCacheStripeMutex(index));
        
        idInt* stored_id = reinterpret_cast<idInt*>(entry);
        *stored_id = internal_id;
        memcpy(entry + sizeof(idInt), data, data_size_);
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
