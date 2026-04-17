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

            constexpr size_t get_sign_word_count(size_t dimension) {
                return (dimension + 63) / 64;
            }

            constexpr size_t get_sign_storage_size(size_t dimension) {
                return get_sign_word_count(dimension) * sizeof(uint64_t);
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
                return *reinterpret_cast<const float*>(buffer + dimension * sizeof(int8_t)
                                                       + get_sign_storage_size(dimension));
            }

            // Storage layout:
            // [int8 payload | residual-sign bitset | scale]
            inline std::vector<uint8_t>
            quantize_vector_fp32_to_int8e_buffer(const std::vector<float>& input) {
                if(input.empty()) {
                    return std::vector<uint8_t>();
                }

                const size_t dimension = input.size();
                std::vector<uint8_t> buffer(get_storage_size(dimension));

                float abs_max = ndd::quant::math::find_abs_max(input.data(), dimension);
                if(abs_max == 0.0f) {
                    abs_max = 1.0f;
                }

                const float scale = abs_max / INT8_SCALE;
                const float inv_scale = 1.0f / scale;

                int8_t* data_ptr = reinterpret_cast<int8_t*>(buffer.data());
                uint64_t* sign_words = extract_sign_words(buffer.data(), dimension);
                const size_t sign_word_count = get_sign_word_count(dimension);
                for(size_t w = 0; w < sign_word_count; ++w) {
                    sign_words[w] = 0ULL;
                }

                for(size_t i = 0; i < dimension; ++i) {
                    const float scaled_real = input[i] * inv_scale;
                    const int8_t q = static_cast<int8_t>(std::round(scaled_real));
                    data_ptr[i] = q;

                    // Bit is 1 for non-negative residual, 0 for negative residual.
                    const float residual = scaled_real - static_cast<float>(q);
                    if(residual >= 0.0f) {
                        const size_t word = i >> 6;
                        const size_t bit = i & 63;
                        sign_words[word] |= (1ULL << bit);
                    }
                }

                float* scale_ptr = reinterpret_cast<float*>(buffer.data() + dimension * sizeof(int8_t)
                                                           + get_sign_storage_size(dimension));
                *scale_ptr = scale;
                return buffer;
            }

            inline std::vector<uint8_t>
            quantize_vector_fp32_to_int8e_buffer_auto(const std::vector<float>& input) {
                return quantize_vector_fp32_to_int8e_buffer(input);
            }

            inline std::vector<float> dequantize_int8e_buffer_to_fp32(const uint8_t* buffer,
                                                                      size_t dimension) {
                std::vector<float> out(dimension);

                const int8_t* payload = reinterpret_cast<const int8_t*>(buffer);
                const uint64_t* sign_words = extract_sign_words(buffer, dimension);
                const float scale = extract_scale(buffer, dimension);

                for(size_t i = 0; i < dimension; ++i) {
                    const size_t word = i >> 6;
                    const size_t bit = i & 63;
                    const float residual_center = ((sign_words[word] >> bit) & 1ULL) != 0ULL
                                                          ? 0.25f
                                                          : -0.25f;
                    out[i] = (static_cast<float>(payload[i]) + residual_center) * scale;
                }

                return out;
            }

            inline std::vector<uint8_t> quantize(const std::vector<float>& input) {
                return quantize_vector_fp32_to_int8e_buffer_auto(input);
            }

            inline std::vector<float> dequantize(const uint8_t* in, size_t dim) {
                return dequantize_int8e_buffer_to_fp32(in, dim);
            }

            static float L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
                const int8_t* pVect1 = static_cast<const int8_t*>(pVect1v);
                const int8_t* pVect2 = static_cast<const int8_t*>(pVect2v);
                const auto* params = static_cast<const hnswlib::DistParams*>(qty_ptr);
                const size_t qty = params->dim;

                const float scale1 = extract_scale(reinterpret_cast<const uint8_t*>(pVect1), qty);
                const float scale2 = extract_scale(reinterpret_cast<const uint8_t*>(pVect2), qty);
                const uint64_t* bits1 = extract_sign_words(reinterpret_cast<const uint8_t*>(pVect1),
                                                           qty);
                const uint64_t* bits2 = extract_sign_words(reinterpret_cast<const uint8_t*>(pVect2),
                                                           qty);

                float l2 = 0.0f;
                for(size_t i = 0; i < qty; ++i) {
                    const size_t word = i >> 6;
                    const size_t bit = i & 63;

                    const float a1 = ((bits1[word] >> bit) & 1ULL) != 0ULL ? 0.25f : -0.25f;
                    const float a2 = ((bits2[word] >> bit) & 1ULL) != 0ULL ? 0.25f : -0.25f;

                    const float x = (static_cast<float>(pVect1[i]) + a1) * scale1;
                    const float y = (static_cast<float>(pVect2[i]) + a2) * scale2;
                    const float d = x - y;
                    l2 += d * d;
                }

                return l2;
            }

            static float L2SqrSim(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
                return -L2Sqr(pVect1v, pVect2v, qty_ptr);
            }

            static float
            InnerProductSim(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
                const int8_t* pVect1 = static_cast<const int8_t*>(pVect1v);
                const int8_t* pVect2 = static_cast<const int8_t*>(pVect2v);
                const auto* params = static_cast<const hnswlib::DistParams*>(qty_ptr);
                const size_t qty = params->dim;

                const float scale1 = extract_scale(reinterpret_cast<const uint8_t*>(pVect1), qty);
                const float scale2 = extract_scale(reinterpret_cast<const uint8_t*>(pVect2), qty);
                const uint64_t* bits1 = extract_sign_words(reinterpret_cast<const uint8_t*>(pVect1),
                                                           qty);
                const uint64_t* bits2 = extract_sign_words(reinterpret_cast<const uint8_t*>(pVect2),
                                                           qty);

                int64_t dot = 0;
                int64_t sum_xi = 0;
                int64_t sum_yi = 0;
                for(size_t i = 0; i < qty; ++i) {
                    const int32_t a = static_cast<int32_t>(pVect1[i]);
                    const int32_t b = static_cast<int32_t>(pVect2[i]);
                    dot += static_cast<int64_t>(a) * b;
                    sum_xi += static_cast<int64_t>(a);
                    sum_yi += static_cast<int64_t>(b);
                }

                int64_t sum_ai_yi = 0;
                int64_t sum_bi_xi = 0;
                int64_t sum_ai_bi = 0;
                int64_t sum_pos_ai_yi = 0;
                int64_t sum_pos_bi_xi = 0;

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

                    const int64_t lane_count =
                            static_cast<int64_t>((base + 64 <= qty) ? 64 : (qty - base));
                    const int64_t diff = static_cast<int64_t>(__builtin_popcountll(b1 ^ b2));
                    sum_ai_bi += lane_count - (diff << 1);

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

                sum_ai_yi = (sum_pos_ai_yi << 1) - sum_yi;
                sum_bi_xi = (sum_pos_bi_xi << 1) - sum_xi;

                const float s1s2 = scale1 * scale2;
                const float base = static_cast<float>(dot) * s1s2;
                const float correction =
                        0.25f * static_cast<float>(sum_ai_yi + sum_bi_xi) * s1s2;
                const float quadratic = 0.0625f * static_cast<float>(sum_ai_bi) * s1s2;
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
                float* scale_ptr = reinterpret_cast<float*>(out.data() + dim * sizeof(int8_t));
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
