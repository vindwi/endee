#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <memory>
#include "common.hpp"
#include "int8.hpp"
#include "../hnsw/hnswlib.h"

namespace ndd {
    namespace quant {
        namespace int8e {

            constexpr float INT8_SCALE = 127.0f;
            constexpr float INV_SQRT2 = 0.7071067811865475f;

            constexpr size_t get_sign_word_count(size_t dimension) {
                return (dimension + 63) / 64;
            }

            constexpr size_t get_sign_storage_size(size_t dimension) {
                return get_sign_word_count(dimension) * sizeof(uint64_t);
            }

            // Apply a deterministic pairwise orthogonal rotation to spread energy.
            inline void rotate_pairwise_inplace(std::vector<float>& values) {
                const size_t dim = values.size();
                const size_t paired = (dim / 2) * 2;
                for(size_t i = 0; i < paired; i += 2) {
                    const float x = values[i];
                    const float y = values[i + 1];
                    values[i] = (x + y) * INV_SQRT2;
                    values[i + 1] = (x - y) * INV_SQRT2;
                }
            }

            constexpr size_t get_storage_size(size_t dimension) {
                return dimension * sizeof(int8_t) + get_sign_storage_size(dimension)
                       + sizeof(float);
            }

            inline const uint64_t* extract_sign_words(const uint8_t* buffer, size_t dimension) {
                return reinterpret_cast<const uint64_t*>(buffer + dimension * sizeof(int8_t));
            }

            inline uint64_t* extract_sign_words(uint8_t* buffer, size_t dimension) {
                return reinterpret_cast<uint64_t*>(buffer + dimension * sizeof(int8_t));
            }

            inline float extract_scale(const uint8_t* buffer, size_t dimension) {
                const size_t sign_bytes = get_sign_storage_size(dimension);
                return *reinterpret_cast<const float*>(buffer + dimension * sizeof(int8_t)
                                                       + sign_bytes);
            }

            // Quantize with nearest rounding and store sign(residual) bits,
            // where residual = real - rounded:
            // 0 for negative residual, 1 for non-negative residual.
            inline std::vector<uint8_t>
            quantize_vector_fp32_to_int8_buffer(const std::vector<float>& input) {
                if(input.empty()) {
                    return std::vector<uint8_t>();
                }

                std::vector<float> rotated = input;
                rotate_pairwise_inplace(rotated);

                const size_t dimension = rotated.size();
                std::vector<uint8_t> base = int8::quantize_vector_fp32_to_int8_buffer(rotated);
                std::vector<uint8_t> buffer(get_storage_size(dimension));

                int8_t* data_ptr = reinterpret_cast<int8_t*>(buffer.data());
                const int8_t* base_data = reinterpret_cast<const int8_t*>(base.data());
                std::copy(base_data, base_data + dimension, data_ptr);

                uint64_t* sign_words = extract_sign_words(buffer.data(), dimension);
                const size_t sign_word_count = get_sign_word_count(dimension);
                for(size_t w = 0; w < sign_word_count; ++w) {
                    sign_words[w] = 0ULL;
                }

                const float scale = int8::extract_scale(base.data(), dimension);
                const float inv_scale = 1.0f / scale;
                for(size_t i = 0; i < dimension; ++i) {
                    const float scaled_real = rotated[i] * inv_scale;
                    const float residual = scaled_real - static_cast<float>(data_ptr[i]);
                    if(residual >= 0.0f) {
                        const size_t word = i >> 6;
                        const size_t bit = i & 63;
                        sign_words[word] |= (1ULL << bit);
                    }
                }

                float* scale_ptr =
                        reinterpret_cast<float*>(buffer.data() + (dimension * sizeof(int8_t))
                                                 + get_sign_storage_size(dimension));
                *scale_ptr = scale;

                return buffer;
            }

            inline std::vector<uint8_t>
            quantize_vector_fp32_to_int8_buffer_auto(const std::vector<float>& input) {
                if(input.empty()) {
                    return std::vector<uint8_t>();
                }

                std::vector<float> rotated = input;
                rotate_pairwise_inplace(rotated);

                const size_t dimension = rotated.size();
                std::vector<uint8_t> base = int8::quantize_vector_fp32_to_int8_buffer_auto(rotated);
                std::vector<uint8_t> buffer(get_storage_size(dimension));

                int8_t* data_ptr = reinterpret_cast<int8_t*>(buffer.data());
                const int8_t* base_data = reinterpret_cast<const int8_t*>(base.data());
                std::copy(base_data, base_data + dimension, data_ptr);

                uint64_t* sign_words = extract_sign_words(buffer.data(), dimension);
                const size_t sign_word_count = get_sign_word_count(dimension);
                for(size_t w = 0; w < sign_word_count; ++w) {
                    sign_words[w] = 0ULL;
                }

                const float scale = int8::extract_scale(base.data(), dimension);
                const float inv_scale = 1.0f / scale;
                for(size_t i = 0; i < dimension; ++i) {
                    const float scaled_real = rotated[i] * inv_scale;
                    const float residual = scaled_real - static_cast<float>(data_ptr[i]);
                    if(residual >= 0.0f) {
                        const size_t word = i >> 6;
                        const size_t bit = i & 63;
                        sign_words[word] |= (1ULL << bit);
                    }
                }

                float* scale_ptr =
                        reinterpret_cast<float*>(buffer.data() + (dimension * sizeof(int8_t))
                                                 + get_sign_storage_size(dimension));
                *scale_ptr = scale;

                return buffer;
            }

            inline std::vector<float> dequantize_int8_buffer_to_fp32(const uint8_t* buffer,
                                                                     size_t dimension) {
                std::vector<uint8_t> base(int8::get_storage_size(dimension));
                std::copy(buffer, buffer + dimension * sizeof(int8_t), base.data());

                const float scale = extract_scale(buffer, dimension);
                float* base_scale_ptr =
                        reinterpret_cast<float*>(base.data() + dimension * sizeof(int8_t));
                *base_scale_ptr = scale;

                std::vector<float> out = int8::dequantize_int8_buffer_to_fp32(base.data(), dimension);
                const uint64_t* sign_words = extract_sign_words(buffer, dimension);
                for(size_t i = 0; i < dimension; ++i) {
                    const size_t word = i >> 6;
                    const size_t bit = i & 63;
                    const float residual_center = ((sign_words[word] >> bit) & 1ULL) != 0ULL
                                                          ? 0.25f
                                                          : -0.25f;
                    out[i] += residual_center * scale;
                }
                // This rotation is self-inverse, so applying it restores original orientation.
                rotate_pairwise_inplace(out);
                return out;
            }

            inline std::vector<uint8_t> quantize(const std::vector<float>& input) {
                return quantize_vector_fp32_to_int8_buffer_auto(input);
            }

            inline std::vector<float> dequantize(const uint8_t* in, size_t dim) {
                return dequantize_int8_buffer_to_fp32(in, dim);
            }

            static float L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
                const int8_t* pVect1 = (const int8_t*)pVect1v;
                const int8_t* pVect2 = (const int8_t*)pVect2v;
                const auto* params = static_cast<const hnswlib::DistParams*>(qty_ptr);
                const size_t qty = params->dim;

                const float scale1 = extract_scale((const uint8_t*)pVect1, qty);
                const float scale2 = extract_scale((const uint8_t*)pVect2, qty);
                const uint64_t* bits1 = extract_sign_words((const uint8_t*)pVect1, qty);
                const uint64_t* bits2 = extract_sign_words((const uint8_t*)pVect2, qty);

                int64_t dot = 0;
                int64_t sum_x2 = 0;
                int64_t sum_y2 = 0;
                int64_t sum_xi = 0;
                int64_t sum_yi = 0;

                size_t i = 0;
#if defined(USE_AVX512)
                __m512i dot_acc = _mm512_setzero_si512();
                __m512i x2_acc = _mm512_setzero_si512();
                __m512i y2_acc = _mm512_setzero_si512();
                __m512i sx_acc = _mm512_setzero_si512();
                __m512i sy_acc = _mm512_setzero_si512();
                const __m512i ones16 = _mm512_set1_epi16(1);

                for(; i + 64 <= qty; i += 64) {
                    const __m256i a0 = _mm256_loadu_si256((const __m256i*)(pVect1 + i));
                    const __m256i b0 = _mm256_loadu_si256((const __m256i*)(pVect2 + i));
                    const __m256i a1 = _mm256_loadu_si256((const __m256i*)(pVect1 + i + 32));
                    const __m256i b1 = _mm256_loadu_si256((const __m256i*)(pVect2 + i + 32));

                    const __m512i a0_16 = _mm512_cvtepi8_epi16(a0);
                    const __m512i b0_16 = _mm512_cvtepi8_epi16(b0);
                    const __m512i a1_16 = _mm512_cvtepi8_epi16(a1);
                    const __m512i b1_16 = _mm512_cvtepi8_epi16(b1);

                    dot_acc = _mm512_add_epi32(dot_acc, _mm512_madd_epi16(a0_16, b0_16));
                    dot_acc = _mm512_add_epi32(dot_acc, _mm512_madd_epi16(a1_16, b1_16));

                    x2_acc = _mm512_add_epi32(x2_acc, _mm512_madd_epi16(a0_16, a0_16));
                    x2_acc = _mm512_add_epi32(x2_acc, _mm512_madd_epi16(a1_16, a1_16));

                    y2_acc = _mm512_add_epi32(y2_acc, _mm512_madd_epi16(b0_16, b0_16));
                    y2_acc = _mm512_add_epi32(y2_acc, _mm512_madd_epi16(b1_16, b1_16));

                    sx_acc = _mm512_add_epi32(sx_acc, _mm512_madd_epi16(a0_16, ones16));
                    sx_acc = _mm512_add_epi32(sx_acc, _mm512_madd_epi16(a1_16, ones16));

                    sy_acc = _mm512_add_epi32(sy_acc, _mm512_madd_epi16(b0_16, ones16));
                    sy_acc = _mm512_add_epi32(sy_acc, _mm512_madd_epi16(b1_16, ones16));
                }

                dot += static_cast<int64_t>(_mm512_reduce_add_epi32(dot_acc));
                sum_x2 += static_cast<int64_t>(_mm512_reduce_add_epi32(x2_acc));
                sum_y2 += static_cast<int64_t>(_mm512_reduce_add_epi32(y2_acc));
                sum_xi += static_cast<int64_t>(_mm512_reduce_add_epi32(sx_acc));
                sum_yi += static_cast<int64_t>(_mm512_reduce_add_epi32(sy_acc));
#elif defined(USE_AVX2)
                __m256i dot_acc = _mm256_setzero_si256();
                __m256i x2_acc = _mm256_setzero_si256();
                __m256i y2_acc = _mm256_setzero_si256();
                __m256i sx_acc = _mm256_setzero_si256();
                __m256i sy_acc = _mm256_setzero_si256();
                const __m256i ones16 = _mm256_set1_epi16(1);

                for(; i + 32 <= qty; i += 32) {
                    const __m128i a0 = _mm_loadu_si128((const __m128i*)(pVect1 + i));
                    const __m128i b0 = _mm_loadu_si128((const __m128i*)(pVect2 + i));
                    const __m128i a1 = _mm_loadu_si128((const __m128i*)(pVect1 + i + 16));
                    const __m128i b1 = _mm_loadu_si128((const __m128i*)(pVect2 + i + 16));

                    const __m256i a0_16 = _mm256_cvtepi8_epi16(a0);
                    const __m256i b0_16 = _mm256_cvtepi8_epi16(b0);
                    const __m256i a1_16 = _mm256_cvtepi8_epi16(a1);
                    const __m256i b1_16 = _mm256_cvtepi8_epi16(b1);

                    dot_acc = _mm256_add_epi32(dot_acc, _mm256_madd_epi16(a0_16, b0_16));
                    dot_acc = _mm256_add_epi32(dot_acc, _mm256_madd_epi16(a1_16, b1_16));

                    x2_acc = _mm256_add_epi32(x2_acc, _mm256_madd_epi16(a0_16, a0_16));
                    x2_acc = _mm256_add_epi32(x2_acc, _mm256_madd_epi16(a1_16, a1_16));

                    y2_acc = _mm256_add_epi32(y2_acc, _mm256_madd_epi16(b0_16, b0_16));
                    y2_acc = _mm256_add_epi32(y2_acc, _mm256_madd_epi16(b1_16, b1_16));

                    sx_acc = _mm256_add_epi32(sx_acc, _mm256_madd_epi16(a0_16, ones16));
                    sx_acc = _mm256_add_epi32(sx_acc, _mm256_madd_epi16(a1_16, ones16));

                    sy_acc = _mm256_add_epi32(sy_acc, _mm256_madd_epi16(b0_16, ones16));
                    sy_acc = _mm256_add_epi32(sy_acc, _mm256_madd_epi16(b1_16, ones16));
                }

                __m128i dot_128 = _mm_add_epi32(_mm256_castsi256_si128(dot_acc),
                                                _mm256_extracti128_si256(dot_acc, 1));
                dot_128 = _mm_hadd_epi32(dot_128, dot_128);
                dot_128 = _mm_hadd_epi32(dot_128, dot_128);
                dot += static_cast<int64_t>(_mm_cvtsi128_si32(dot_128));

                __m128i x2_128 = _mm_add_epi32(_mm256_castsi256_si128(x2_acc),
                                               _mm256_extracti128_si256(x2_acc, 1));
                x2_128 = _mm_hadd_epi32(x2_128, x2_128);
                x2_128 = _mm_hadd_epi32(x2_128, x2_128);
                sum_x2 += static_cast<int64_t>(_mm_cvtsi128_si32(x2_128));

                __m128i y2_128 = _mm_add_epi32(_mm256_castsi256_si128(y2_acc),
                                               _mm256_extracti128_si256(y2_acc, 1));
                y2_128 = _mm_hadd_epi32(y2_128, y2_128);
                y2_128 = _mm_hadd_epi32(y2_128, y2_128);
                sum_y2 += static_cast<int64_t>(_mm_cvtsi128_si32(y2_128));

                __m128i sx_128 = _mm_add_epi32(_mm256_castsi256_si128(sx_acc),
                                               _mm256_extracti128_si256(sx_acc, 1));
                sx_128 = _mm_hadd_epi32(sx_128, sx_128);
                sx_128 = _mm_hadd_epi32(sx_128, sx_128);
                sum_xi += static_cast<int64_t>(_mm_cvtsi128_si32(sx_128));

                __m128i sy_128 = _mm_add_epi32(_mm256_castsi256_si128(sy_acc),
                                               _mm256_extracti128_si256(sy_acc, 1));
                sy_128 = _mm_hadd_epi32(sy_128, sy_128);
                sy_128 = _mm_hadd_epi32(sy_128, sy_128);
                sum_yi += static_cast<int64_t>(_mm_cvtsi128_si32(sy_128));
#elif defined(USE_SVE2)
                svint32_t sum_sq1 = svdup_s32(0);
                svint32_t sum_sq2 = svdup_s32(0);
                svint32_t sum_prod = svdup_s32(0);
                svint32_t sum_sx = svdup_s32(0);
                svint32_t sum_sy = svdup_s32(0);
                const svint8_t ones = svdup_n_s8(1);

                const uint64_t num_bytes = svcntb();
                const size_t unroll_stride = num_bytes * 2;
                const svbool_t pg_all = svptrue_b8();

                for(; i + unroll_stride <= qty; i += unroll_stride) {
                    svint8_t v1_0 = svld1_s8(pg_all, pVect1 + i);
                    svint8_t v2_0 = svld1_s8(pg_all, pVect2 + i);
                    sum_sq1 = svdot_s32(sum_sq1, v1_0, v1_0);
                    sum_sq2 = svdot_s32(sum_sq2, v2_0, v2_0);
                    sum_prod = svdot_s32(sum_prod, v1_0, v2_0);
                    sum_sx = svdot_s32(sum_sx, v1_0, ones);
                    sum_sy = svdot_s32(sum_sy, v2_0, ones);

                    svint8_t v1_1 = svld1_s8(pg_all, pVect1 + i + num_bytes);
                    svint8_t v2_1 = svld1_s8(pg_all, pVect2 + i + num_bytes);
                    sum_sq1 = svdot_s32(sum_sq1, v1_1, v1_1);
                    sum_sq2 = svdot_s32(sum_sq2, v2_1, v2_1);
                    sum_prod = svdot_s32(sum_prod, v1_1, v2_1);
                    sum_sx = svdot_s32(sum_sx, v1_1, ones);
                    sum_sy = svdot_s32(sum_sy, v2_1, ones);
                }

                dot += static_cast<int64_t>(svaddv_s32(svptrue_b32(), sum_prod));
                sum_x2 += static_cast<int64_t>(svaddv_s32(svptrue_b32(), sum_sq1));
                sum_y2 += static_cast<int64_t>(svaddv_s32(svptrue_b32(), sum_sq2));
                sum_xi += static_cast<int64_t>(svaddv_s32(svptrue_b32(), sum_sx));
                sum_yi += static_cast<int64_t>(svaddv_s32(svptrue_b32(), sum_sy));
#elif defined(USE_NEON)
#    if defined(__ARM_FEATURE_DOTPROD)
                int32x4_t dot_vec = vdupq_n_s32(0);
                int32x4_t x2_vec = vdupq_n_s32(0);
                int32x4_t y2_vec = vdupq_n_s32(0);
                int32x4_t sx_vec = vdupq_n_s32(0);
                int32x4_t sy_vec = vdupq_n_s32(0);
                const int8x16_t ones = vdupq_n_s8(1);

                for(; i + 16 <= qty; i += 16) {
                    const int8x16_t va = vld1q_s8(pVect1 + i);
                    const int8x16_t vb = vld1q_s8(pVect2 + i);

                    dot_vec = vdotq_s32(dot_vec, va, vb);
                    x2_vec = vdotq_s32(x2_vec, va, va);
                    y2_vec = vdotq_s32(y2_vec, vb, vb);
                    sx_vec = vdotq_s32(sx_vec, va, ones);
                    sy_vec = vdotq_s32(sy_vec, vb, ones);
                }

                dot += static_cast<int64_t>(vaddvq_s32(dot_vec));
                sum_x2 += static_cast<int64_t>(vaddvq_s32(x2_vec));
                sum_y2 += static_cast<int64_t>(vaddvq_s32(y2_vec));
                sum_xi += static_cast<int64_t>(vaddvq_s32(sx_vec));
                sum_yi += static_cast<int64_t>(vaddvq_s32(sy_vec));
#    else
                for(; i + 16 <= qty; i += 16) {
                    const int8x16_t va = vld1q_s8(pVect1 + i);
                    const int8x16_t vb = vld1q_s8(pVect2 + i);

                    const int16x8_t prod_lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
                    const int16x8_t prod_hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
                    const int32x4_t dot_vec =
                            vaddq_s32(vpaddlq_s16(prod_lo), vpaddlq_s16(prod_hi));
                    dot += horizontal_sum_s32x4(dot_vec);

                    const int16x8_t sqx_lo = vmull_s8(vget_low_s8(va), vget_low_s8(va));
                    const int16x8_t sqx_hi = vmull_s8(vget_high_s8(va), vget_high_s8(va));
                    const int16x8_t sqy_lo = vmull_s8(vget_low_s8(vb), vget_low_s8(vb));
                    const int16x8_t sqy_hi = vmull_s8(vget_high_s8(vb), vget_high_s8(vb));

                    const int32x4_t x2_vec = vaddq_s32(vpaddlq_s16(sqx_lo), vpaddlq_s16(sqx_hi));
                    const int32x4_t y2_vec = vaddq_s32(vpaddlq_s16(sqy_lo), vpaddlq_s16(sqy_hi));
                    sum_x2 += horizontal_sum_s32x4(x2_vec);
                    sum_y2 += horizontal_sum_s32x4(y2_vec);

                    const int16x8_t sx16 = vpaddlq_s8(va);
                    const int16x8_t sy16 = vpaddlq_s8(vb);
                    sum_xi += horizontal_sum_s32x4(vpaddlq_s16(sx16));
                    sum_yi += horizontal_sum_s32x4(vpaddlq_s16(sy16));
                }
#    endif
#endif

                for(; i < qty; ++i) {
                    const int32_t a = static_cast<int32_t>(pVect1[i]);
                    const int32_t b = static_cast<int32_t>(pVect2[i]);
                    dot += static_cast<int64_t>(a) * b;
                    sum_x2 += static_cast<int64_t>(a) * a;
                    sum_y2 += static_cast<int64_t>(b) * b;
                    sum_xi += static_cast<int64_t>(a);
                    sum_yi += static_cast<int64_t>(b);
                }

                int64_t sum_pos_ai_yi = 0;
                int64_t sum_pos_bi_xi = 0;
                int64_t sum_pos_ai_xi = 0;
                int64_t sum_pos_bi_yi = 0;
                int64_t sum_ai_bi = 0;

                const size_t word_count = get_sign_word_count(qty);
                for(size_t w = 0; w < word_count; ++w) {
                    const size_t base = w * 64;
                    uint64_t mask = ~0ULL;
                    if(base + 64 > qty) {
                        const size_t remaining = qty - base;
                        mask = (remaining == 64) ? ~0ULL : ((1ULL << remaining) - 1ULL);
                    }

                    const uint64_t b1 = bits1[w] & mask;
                    const uint64_t b2 = bits2[w] & mask;

                    const int64_t remaining = static_cast<int64_t>(
                            (base + 64 <= qty) ? 64 : (qty - base));
                    const int64_t diff = static_cast<int64_t>(__builtin_popcountll(b1 ^ b2));
                    sum_ai_bi += remaining - (diff << 1);

                    uint64_t active1 = b1;
                    while(active1 != 0ULL) {
                        const size_t bit = static_cast<size_t>(__builtin_ctzll(active1));
                        const size_t idx = base + bit;
                        sum_pos_ai_yi += static_cast<int64_t>(pVect2[idx]);
                        sum_pos_ai_xi += static_cast<int64_t>(pVect1[idx]);
                        active1 &= (active1 - 1ULL);
                    }

                    uint64_t active2 = b2;
                    while(active2 != 0ULL) {
                        const size_t bit = static_cast<size_t>(__builtin_ctzll(active2));
                        const size_t idx = base + bit;
                        sum_pos_bi_xi += static_cast<int64_t>(pVect1[idx]);
                        sum_pos_bi_yi += static_cast<int64_t>(pVect2[idx]);
                        active2 &= (active2 - 1ULL);
                    }
                }

                const int64_t sum_ai_yi = (sum_pos_ai_yi << 1) - sum_yi;
                const int64_t sum_bi_xi = (sum_pos_bi_xi << 1) - sum_xi;
                const int64_t sum_ai_xi = (sum_pos_ai_xi << 1) - sum_xi;
                const int64_t sum_bi_yi = (sum_pos_bi_yi << 1) - sum_yi;

                const float s1s1 = scale1 * scale1;
                const float s2s2 = scale2 * scale2;
                const float s1s2 = scale1 * scale2;

                const float base = static_cast<float>(sum_x2) * s1s1
                                   + static_cast<float>(sum_y2) * s2s2
                                   - 2.0f * static_cast<float>(dot) * s1s2;

                // Linear correction from ai/4 and bi/4 terms.
                const float linear = 0.5f
                                     * (static_cast<float>(sum_ai_xi) * s1s1
                                        + static_cast<float>(sum_bi_yi) * s2s2
                                        - static_cast<float>(sum_ai_yi + sum_bi_xi) * s1s2);

                // Quadratic correction from (ai/4)^2, (bi/4)^2, and ai*bi/16 terms.
                const float quadratic = (static_cast<float>(qty) * 0.0625f) * (s1s1 + s2s2)
                                        - 0.125f * static_cast<float>(sum_ai_bi) * s1s2;

                return base + linear + quadratic;
            }

            static float L2SqrSim(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
                return -L2Sqr(pVect1v, pVect2v, qty_ptr);
            }

            static float
            InnerProductSim(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
                const int8_t* pVect1 = (const int8_t*)pVect1v;
                const int8_t* pVect2 = (const int8_t*)pVect2v;
                const auto* params = static_cast<const hnswlib::DistParams*>(qty_ptr);
                const size_t qty = params->dim;

                const float scale1 = extract_scale((const uint8_t*)pVect1, qty);
                const float scale2 = extract_scale((const uint8_t*)pVect2, qty);
                const uint64_t* bits1 = extract_sign_words((const uint8_t*)pVect1, qty);
                const uint64_t* bits2 = extract_sign_words((const uint8_t*)pVect2, qty);

                int64_t dot = 0;
                int64_t sum_xi = 0;
                int64_t sum_yi = 0;

                size_t i = 0;
#if defined(USE_AVX512)
                __m512i dot_acc = _mm512_setzero_si512();
                __m512i sx_acc = _mm512_setzero_si512();
                __m512i sy_acc = _mm512_setzero_si512();
                const __m512i ones16 = _mm512_set1_epi16(1);

                for(; i + 64 <= qty; i += 64) {
                    const __m256i a0 = _mm256_loadu_si256((const __m256i*)(pVect1 + i));
                    const __m256i b0 = _mm256_loadu_si256((const __m256i*)(pVect2 + i));
                    const __m256i a1 = _mm256_loadu_si256((const __m256i*)(pVect1 + i + 32));
                    const __m256i b1 = _mm256_loadu_si256((const __m256i*)(pVect2 + i + 32));

                    const __m512i a0_16 = _mm512_cvtepi8_epi16(a0);
                    const __m512i b0_16 = _mm512_cvtepi8_epi16(b0);
                    const __m512i a1_16 = _mm512_cvtepi8_epi16(a1);
                    const __m512i b1_16 = _mm512_cvtepi8_epi16(b1);

                    dot_acc = _mm512_add_epi32(dot_acc, _mm512_madd_epi16(a0_16, b0_16));
                    dot_acc = _mm512_add_epi32(dot_acc, _mm512_madd_epi16(a1_16, b1_16));

                    sx_acc = _mm512_add_epi32(sx_acc, _mm512_madd_epi16(a0_16, ones16));
                    sx_acc = _mm512_add_epi32(sx_acc, _mm512_madd_epi16(a1_16, ones16));

                    sy_acc = _mm512_add_epi32(sy_acc, _mm512_madd_epi16(b0_16, ones16));
                    sy_acc = _mm512_add_epi32(sy_acc, _mm512_madd_epi16(b1_16, ones16));
                }

                dot += static_cast<int64_t>(_mm512_reduce_add_epi32(dot_acc));
                sum_xi += static_cast<int64_t>(_mm512_reduce_add_epi32(sx_acc));
                sum_yi += static_cast<int64_t>(_mm512_reduce_add_epi32(sy_acc));
#elif defined(USE_AVX2)
                __m256i dot_acc = _mm256_setzero_si256();
                __m256i sx_acc = _mm256_setzero_si256();
                __m256i sy_acc = _mm256_setzero_si256();
                const __m256i ones16 = _mm256_set1_epi16(1);

                for(; i + 32 <= qty; i += 32) {
                    const __m128i a0 = _mm_loadu_si128((const __m128i*)(pVect1 + i));
                    const __m128i b0 = _mm_loadu_si128((const __m128i*)(pVect2 + i));
                    const __m128i a1 = _mm_loadu_si128((const __m128i*)(pVect1 + i + 16));
                    const __m128i b1 = _mm_loadu_si128((const __m128i*)(pVect2 + i + 16));

                    const __m256i a0_16 = _mm256_cvtepi8_epi16(a0);
                    const __m256i b0_16 = _mm256_cvtepi8_epi16(b0);
                    const __m256i a1_16 = _mm256_cvtepi8_epi16(a1);
                    const __m256i b1_16 = _mm256_cvtepi8_epi16(b1);

                    dot_acc = _mm256_add_epi32(dot_acc, _mm256_madd_epi16(a0_16, b0_16));
                    dot_acc = _mm256_add_epi32(dot_acc, _mm256_madd_epi16(a1_16, b1_16));

                    sx_acc = _mm256_add_epi32(sx_acc, _mm256_madd_epi16(a0_16, ones16));
                    sx_acc = _mm256_add_epi32(sx_acc, _mm256_madd_epi16(a1_16, ones16));

                    sy_acc = _mm256_add_epi32(sy_acc, _mm256_madd_epi16(b0_16, ones16));
                    sy_acc = _mm256_add_epi32(sy_acc, _mm256_madd_epi16(b1_16, ones16));
                }

                __m128i dot_128 = _mm_add_epi32(_mm256_castsi256_si128(dot_acc),
                                                _mm256_extracti128_si256(dot_acc, 1));
                dot_128 = _mm_hadd_epi32(dot_128, dot_128);
                dot_128 = _mm_hadd_epi32(dot_128, dot_128);
                dot += static_cast<int64_t>(_mm_cvtsi128_si32(dot_128));

                __m128i sx_128 = _mm_add_epi32(_mm256_castsi256_si128(sx_acc),
                                               _mm256_extracti128_si256(sx_acc, 1));
                sx_128 = _mm_hadd_epi32(sx_128, sx_128);
                sx_128 = _mm_hadd_epi32(sx_128, sx_128);
                sum_xi += static_cast<int64_t>(_mm_cvtsi128_si32(sx_128));

                __m128i sy_128 = _mm_add_epi32(_mm256_castsi256_si128(sy_acc),
                                               _mm256_extracti128_si256(sy_acc, 1));
                sy_128 = _mm_hadd_epi32(sy_128, sy_128);
                sy_128 = _mm_hadd_epi32(sy_128, sy_128);
                sum_yi += static_cast<int64_t>(_mm_cvtsi128_si32(sy_128));
#elif defined(USE_SVE2)
                svint32_t dot_acc = svdup_s32(0);
                svint32_t sx_acc = svdup_s32(0);
                svint32_t sy_acc = svdup_s32(0);
                const svint8_t ones = svdup_n_s8(1);
                const uint64_t num_bytes = svcntb();
                const size_t unroll_stride = num_bytes * 4;
                const svbool_t pg_all = svptrue_b8();

                for(; i + unroll_stride <= qty; i += unroll_stride) {
                    svint8_t v1_0 = svld1_s8(pg_all, pVect1 + i);
                    svint8_t v2_0 = svld1_s8(pg_all, pVect2 + i);
                    dot_acc = svdot_s32(dot_acc, v1_0, v2_0);
                    sx_acc = svdot_s32(sx_acc, v1_0, ones);
                    sy_acc = svdot_s32(sy_acc, v2_0, ones);

                    svint8_t v1_1 = svld1_s8(pg_all, pVect1 + i + num_bytes);
                    svint8_t v2_1 = svld1_s8(pg_all, pVect2 + i + num_bytes);
                    dot_acc = svdot_s32(dot_acc, v1_1, v2_1);
                    sx_acc = svdot_s32(sx_acc, v1_1, ones);
                    sy_acc = svdot_s32(sy_acc, v2_1, ones);

                    svint8_t v1_2 = svld1_s8(pg_all, pVect1 + i + 2 * num_bytes);
                    svint8_t v2_2 = svld1_s8(pg_all, pVect2 + i + 2 * num_bytes);
                    dot_acc = svdot_s32(dot_acc, v1_2, v2_2);
                    sx_acc = svdot_s32(sx_acc, v1_2, ones);
                    sy_acc = svdot_s32(sy_acc, v2_2, ones);

                    svint8_t v1_3 = svld1_s8(pg_all, pVect1 + i + 3 * num_bytes);
                    svint8_t v2_3 = svld1_s8(pg_all, pVect2 + i + 3 * num_bytes);
                    dot_acc = svdot_s32(dot_acc, v1_3, v2_3);
                    sx_acc = svdot_s32(sx_acc, v1_3, ones);
                    sy_acc = svdot_s32(sy_acc, v2_3, ones);
                }

                dot += static_cast<int64_t>(svaddv_s32(svptrue_b32(), dot_acc));
                sum_xi += static_cast<int64_t>(svaddv_s32(svptrue_b32(), sx_acc));
                sum_yi += static_cast<int64_t>(svaddv_s32(svptrue_b32(), sy_acc));
#elif defined(USE_NEON)
#    if defined(__ARM_FEATURE_DOTPROD)
                int32x4_t dot_vec = vdupq_n_s32(0);
                int32x4_t sx_vec = vdupq_n_s32(0);
                int32x4_t sy_vec = vdupq_n_s32(0);
                const int8x16_t ones = vdupq_n_s8(1);

                for(; i + 16 <= qty; i += 16) {
                    const int8x16_t va = vld1q_s8(pVect1 + i);
                    const int8x16_t vb = vld1q_s8(pVect2 + i);

                    dot_vec = vdotq_s32(dot_vec, va, vb);
                    sx_vec = vdotq_s32(sx_vec, va, ones);
                    sy_vec = vdotq_s32(sy_vec, vb, ones);
                }

                dot += static_cast<int64_t>(vaddvq_s32(dot_vec));
                sum_xi += static_cast<int64_t>(vaddvq_s32(sx_vec));
                sum_yi += static_cast<int64_t>(vaddvq_s32(sy_vec));
#    else
                for(; i + 16 <= qty; i += 16) {
                    const int8x16_t va = vld1q_s8(pVect1 + i);
                    const int8x16_t vb = vld1q_s8(pVect2 + i);

                    const int16x8_t prod_lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
                    const int16x8_t prod_hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
                    const int32x4_t dot_vec =
                            vaddq_s32(vpaddlq_s16(prod_lo), vpaddlq_s16(prod_hi));
                    dot += horizontal_sum_s32x4(dot_vec);

                    const int16x8_t sx16 = vpaddlq_s8(va);
                    const int16x8_t sy16 = vpaddlq_s8(vb);
                    sum_xi += horizontal_sum_s32x4(vpaddlq_s16(sx16));
                    sum_yi += horizontal_sum_s32x4(vpaddlq_s16(sy16));
                }
#    endif
#endif

                for(; i < qty; ++i) {
                    const int32_t a = static_cast<int32_t>(pVect1[i]);
                    const int32_t b = static_cast<int32_t>(pVect2[i]);
                    dot += static_cast<int64_t>(a) * b;
                    sum_xi += static_cast<int64_t>(a);
                    sum_yi += static_cast<int64_t>(b);
                }

                int64_t sum_pos_ai_yi = 0;
                int64_t sum_pos_bi_xi = 0;
                int64_t sum_ai_bi = 0;
                const size_t word_count = get_sign_word_count(qty);
                for(size_t w = 0; w < word_count; ++w) {
                    const size_t base = w * 64;
                    uint64_t mask = ~0ULL;
                    if(base + 64 > qty) {
                        const size_t remaining = qty - base;
                        mask = (remaining == 64) ? ~0ULL : ((1ULL << remaining) - 1ULL);
                    }

                    const uint64_t b1 = bits1[w] & mask;
                    const uint64_t b2 = bits2[w] & mask;

                    const int64_t remaining =
                            static_cast<int64_t>((base + 64 <= qty) ? 64 : (qty - base));
                    const int64_t diff = static_cast<int64_t>(__builtin_popcountll(b1 ^ b2));
                    sum_ai_bi += remaining - (diff << 1);

                    uint64_t active1 = b1;
                    while(active1 != 0ULL) {
                        const size_t bit = static_cast<size_t>(__builtin_ctzll(active1));
                        const size_t idx = base + bit;
                        sum_pos_ai_yi += static_cast<int64_t>(pVect2[idx]);
                        active1 &= (active1 - 1ULL);
                    }

                    uint64_t active2 = b2;
                    while(active2 != 0ULL) {
                        const size_t bit = static_cast<size_t>(__builtin_ctzll(active2));
                        const size_t idx = base + bit;
                        sum_pos_bi_xi += static_cast<int64_t>(pVect1[idx]);
                        active2 &= (active2 - 1ULL);
                    }
                }

                // Convert stored 0/1 bits into centered signs {-1, +1}:
                // sum(ai * yi) = 2 * sum(yi where ai_bit=1) - sum(yi)
                // sum(bi * xi) = 2 * sum(xi where bi_bit=1) - sum(xi)
                const int64_t sum_ai_yi = (sum_pos_ai_yi << 1) - sum_yi;
                const int64_t sum_bi_xi = (sum_pos_bi_xi << 1) - sum_xi;

                const float base = static_cast<float>(dot) * scale1 * scale2;
                const float correction = 0.25f * static_cast<float>(sum_ai_yi + sum_bi_xi) *
                                         scale1 * scale2;
                const float quadratic = 0.0625f * static_cast<float>(sum_ai_bi) * scale1 * scale2;
                return base + correction + quadratic;
            }

            static float
            InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
                return 1.0f - InnerProductSim(pVect1v, pVect2v, qty_ptr);
            }

            static float CosineSim(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
                return InnerProductSim(pVect1v, pVect2v, qty_ptr);
            }

            static float Cosine(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
                return 1.0f - CosineSim(pVect1v, pVect2v, qty_ptr);
            }

            static void L2SqrSimBatch(const void* query,
                                      const void* const* vectors,
                                      size_t count,
                                      const void* params,
                                      float* out) {
                for(size_t i = 0; i < count; ++i) {
                    out[i] = L2SqrSim(query, vectors[i], params);
                }
            }

            static void InnerProductSimBatch(const void* query,
                                             const void* const* vectors,
                                             size_t count,
                                             const void* params,
                                             float* out) {
                for(size_t i = 0; i < count; ++i) {
                    out[i] = InnerProductSim(query, vectors[i], params);
                }
            }

            static void CosineSimBatch(const void* query,
                                       const void* const* vectors,
                                       size_t count,
                                       const void* params,
                                       float* out) {
                for(size_t i = 0; i < count; ++i) {
                    out[i] = CosineSim(query, vectors[i], params);
                }
            }

            static std::vector<uint8_t> quantize_to_int8(const void* in, size_t dim) {
                const uint8_t* src = static_cast<const uint8_t*>(in);
                std::vector<uint8_t> out(int8::get_storage_size(dim));

                std::copy(src, src + dim * sizeof(int8_t), out.data());

                const float scale = extract_scale(src, dim);
                float* scale_ptr =
                    reinterpret_cast<float*>(out.data() + dim * sizeof(int8_t));
                *scale_ptr = scale;

                return out;
            }

        }  // namespace int8e

        class Int8EQuantizer : public ndd::quant::Quantizer {
        public:
            std::string name() const override { return "int8e"; }
            ndd::quant::QuantizationLevel level() const override {
                return ndd::quant::QuantizationLevel::INT8E;
            }

            ndd::quant::QuantizerDispatch getDispatch() const override {
                ndd::quant::QuantizerDispatch d;
                d.dist_l2 = &int8e::L2Sqr;
                d.dist_ip = &int8e::InnerProduct;
                d.dist_cosine = &int8e::Cosine;
                d.sim_l2 = &int8e::L2SqrSim;
                d.sim_ip = &int8e::InnerProductSim;
                d.sim_cosine = &int8e::CosineSim;
                d.sim_l2_batch = &int8e::L2SqrSimBatch;
                d.sim_ip_batch = &int8e::InnerProductSimBatch;
                d.sim_cosine_batch = &int8e::CosineSimBatch;
                d.quantize = &int8e::quantize;
                d.dequantize = &int8e::dequantize;
                d.quantize_to_int8 = &int8e::quantize_to_int8;
                d.get_storage_size = &int8e::get_storage_size;
                d.extract_scale = &int8e::extract_scale;
                return d;
            }
        };

        static ndd::quant::RegisterQuantizer reg_int8e(ndd::quant::QuantizationLevel::INT8E,
                                                       "int8e",
                                                       std::make_shared<Int8EQuantizer>());

    }  // namespace quant
}  // namespace ndd
