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
        namespace int4e {

            constexpr float INT4_SCALE = 7.0f;
            constexpr float INV_SQRT2 = 0.7071067811865475f;

            constexpr size_t get_packed_value_bytes(size_t dimension) {
                return (dimension + 1) / 2;
            }

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
                return get_packed_value_bytes(dimension) + get_sign_storage_size(dimension)
                       + sizeof(float);
            }

            inline const uint64_t* extract_sign_words(const uint8_t* buffer, size_t dimension) {
                return reinterpret_cast<const uint64_t*>(buffer + get_packed_value_bytes(dimension));
            }

            inline uint64_t* extract_sign_words(uint8_t* buffer, size_t dimension) {
                return reinterpret_cast<uint64_t*>(buffer + get_packed_value_bytes(dimension));
            }

            inline float extract_scale(const uint8_t* buffer, size_t dimension) {
                const size_t sign_bytes = get_sign_storage_size(dimension);
                return *reinterpret_cast<const float*>(buffer + get_packed_value_bytes(dimension)
                                                       + sign_bytes);
            }

            // Encode a signed 4-bit value in [-7, 7] to a nibble in [1, 15].
            inline uint8_t encode_q4(int8_t v) {
                return static_cast<uint8_t>(v + 8);
            }

            // Decode nibble back to signed value in [-7, 7].
            inline int8_t decode_q4(uint8_t nibble) {
                return static_cast<int8_t>(nibble) - 8;
            }

            inline void set_q4(uint8_t* packed, size_t index, int8_t v) {
                const uint8_t nibble = encode_q4(v) & 0x0F;
                const size_t byte_idx = index >> 1;
                if((index & 1) == 0) {
                    packed[byte_idx] = static_cast<uint8_t>((packed[byte_idx] & 0xF0) | nibble);
                } else {
                    packed[byte_idx] = static_cast<uint8_t>((packed[byte_idx] & 0x0F)
                                                            | (nibble << 4));
                }
            }

            inline int8_t get_q4(const uint8_t* packed, size_t index) {
                const size_t byte_idx = index >> 1;
                const uint8_t byte = packed[byte_idx];
                const uint8_t nibble = ((index & 1) == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                return decode_q4(nibble);
            }

            // Quantize with nearest rounding and store sign(residual) bits,
            // where residual = real - rounded:
            // 0 for negative residual, 1 for non-negative residual.
            inline std::vector<uint8_t>
            quantize_vector_fp32_to_int4_buffer(const std::vector<float>& input) {
                if(input.empty()) {
                    return std::vector<uint8_t>();
                }

                std::vector<float> rotated = input;
                rotate_pairwise_inplace(rotated);

                const size_t dimension = rotated.size();
                const size_t buffer_size = get_storage_size(dimension);
                std::vector<uint8_t> buffer(buffer_size);
                const size_t sign_word_count = get_sign_word_count(dimension);

                float abs_max = ndd::quant::math::find_abs_max(rotated.data(), dimension);
                if(abs_max == 0.0f) {
                    abs_max = 1.0f;
                }
                const float scale = abs_max / INT4_SCALE;
                const float inv_scale = 1.0f / scale;

                uint8_t* packed_ptr = buffer.data();
                std::fill(packed_ptr, packed_ptr + get_packed_value_bytes(dimension), 0x88);

                uint64_t* sign_words = extract_sign_words(buffer.data(), dimension);
                for(size_t w = 0; w < sign_word_count; ++w) {
                    sign_words[w] = 0ULL;
                }

                for(size_t i = 0; i < dimension; ++i) {
                    const float scaled_real = rotated[i] * inv_scale;
                    const float scaled = std::round(scaled_real);
                    const float clamped = std::max(-7.0f, std::min(7.0f, scaled));
                    set_q4(packed_ptr, i, static_cast<int8_t>(clamped));

                    const float residual = scaled_real - clamped;
                    if(residual >= 0.0f) {
                        const size_t word = i >> 6;
                        const size_t bit = i & 63;
                        sign_words[word] |= (1ULL << bit);
                    }
                }

                float* scale_ptr = reinterpret_cast<float*>(buffer.data()
                                                            + get_packed_value_bytes(dimension)
                                                            + get_sign_storage_size(dimension));
                *scale_ptr = scale;

                return buffer;
            }

            inline std::vector<uint8_t>
            quantize_vector_fp32_to_int4_buffer_auto(const std::vector<float>& input) {
                return quantize_vector_fp32_to_int4_buffer(input);
            }

            inline std::vector<float> dequantize_int4_buffer_to_fp32(const uint8_t* buffer,
                                                                     size_t dimension) {
                std::vector<float> out(dimension);
                const uint8_t* packed_ptr = buffer;
                const float scale = extract_scale(buffer, dimension);
                for(size_t i = 0; i < dimension; ++i) {
                    out[i] = static_cast<float>(get_q4(packed_ptr, i)) * scale;
                }
                // This rotation is self-inverse, so applying it restores original orientation.
                rotate_pairwise_inplace(out);
                return out;
            }

            inline std::vector<uint8_t> quantize(const std::vector<float>& input) {
                return quantize_vector_fp32_to_int4_buffer_auto(input);
            }

            inline std::vector<float> dequantize(const uint8_t* in, size_t dim) {
                return dequantize_int4_buffer_to_fp32(in, dim);
            }

            static float L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
                const uint8_t* pVect1 = static_cast<const uint8_t*>(pVect1v);
                const uint8_t* pVect2 = static_cast<const uint8_t*>(pVect2v);
                const auto* params = static_cast<const hnswlib::DistParams*>(qty_ptr);
                const size_t qty = params->dim;

                const float scale1 = extract_scale(pVect1, qty);
                const float scale2 = extract_scale(pVect2, qty);

                float res = 0.0f;
                for(size_t i = 0; i < qty; ++i) {
                    const float v1 = static_cast<float>(get_q4(pVect1, i)) * scale1;
                    const float v2 = static_cast<float>(get_q4(pVect2, i)) * scale2;
                    const float diff = v1 - v2;
                    res += diff * diff;
                }
                return res;
            }

            static float L2SqrSim(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
                return -L2Sqr(pVect1v, pVect2v, qty_ptr);
            }

            static float
            InnerProductSim(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
                const uint8_t* pVect1 = static_cast<const uint8_t*>(pVect1v);
                const uint8_t* pVect2 = static_cast<const uint8_t*>(pVect2v);
                const auto* params = static_cast<const hnswlib::DistParams*>(qty_ptr);
                const size_t qty = params->dim;

                const float scale1 = extract_scale(pVect1, qty);
                const float scale2 = extract_scale(pVect2, qty);
                const uint64_t* bits1 = extract_sign_words(pVect1, qty);
                const uint64_t* bits2 = extract_sign_words(pVect2, qty);

                int64_t dot = 0;
                int64_t sum_xi = 0;
                int64_t sum_yi = 0;
                for(size_t i = 0; i < qty; ++i) {
                    const int32_t a = static_cast<int32_t>(get_q4(pVect1, i));
                    const int32_t b = static_cast<int32_t>(get_q4(pVect2, i));
                    dot += static_cast<int64_t>(a) * b;
                    sum_xi += static_cast<int64_t>(a);
                    sum_yi += static_cast<int64_t>(b);
                }

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

                    uint64_t active1 = bits1[w] & mask;
                    while(active1 != 0ULL) {
                        const size_t bit = static_cast<size_t>(__builtin_ctzll(active1));
                        const size_t idx = base + bit;
                        sum_pos_ai_yi += static_cast<int64_t>(get_q4(pVect2, idx));
                        active1 &= (active1 - 1ULL);
                    }

                    uint64_t active2 = bits2[w] & mask;
                    while(active2 != 0ULL) {
                        const size_t bit = static_cast<size_t>(__builtin_ctzll(active2));
                        const size_t idx = base + bit;
                        sum_pos_bi_xi += static_cast<int64_t>(get_q4(pVect1, idx));
                        active2 &= (active2 - 1ULL);
                    }
                }

                // Convert stored 0/1 bits into centered signs {-1, +1}:
                // sum(ai * yi) = 2 * sum(yi where ai_bit=1) - sum(yi)
                // sum(bi * xi) = 2 * sum(xi where bi_bit=1) - sum(xi)
                const int64_t sum_ai_yi = (sum_pos_ai_yi << 1) - sum_yi;
                const int64_t sum_bi_xi = (sum_pos_bi_xi << 1) - sum_xi;

                const float base = static_cast<float>(dot) * scale1 * scale2;
                const float correction = 0.25f * static_cast<float>(sum_ai_yi + sum_bi_xi)
                                         * scale1 * scale2;
                return base + correction;
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
                std::vector<float> deq = dequantize_int4_buffer_to_fp32(static_cast<const uint8_t*>(in), dim);
                return int8::quantize_vector_fp32_to_int8_buffer(deq);
            }

        }  // namespace int4e

        class Int4EQuantizer : public ndd::quant::Quantizer {
        public:
            std::string name() const override { return "int4e"; }
            ndd::quant::QuantizationLevel level() const override {
                return ndd::quant::QuantizationLevel::INT4E;
            }

            ndd::quant::QuantizerDispatch getDispatch() const override {
                ndd::quant::QuantizerDispatch d;
                d.dist_l2 = &int4e::L2Sqr;
                d.dist_ip = &int4e::InnerProduct;
                d.dist_cosine = &int4e::Cosine;
                d.sim_l2 = &int4e::L2SqrSim;
                d.sim_ip = &int4e::InnerProductSim;
                d.sim_cosine = &int4e::CosineSim;
                d.sim_l2_batch = &int4e::L2SqrSimBatch;
                d.sim_ip_batch = &int4e::InnerProductSimBatch;
                d.sim_cosine_batch = &int4e::CosineSimBatch;
                d.quantize = &int4e::quantize;
                d.dequantize = &int4e::dequantize;
                d.quantize_to_int8 = &int4e::quantize_to_int8;
                d.get_storage_size = &int4e::get_storage_size;
                d.extract_scale = &int4e::extract_scale;
                return d;
            }
        };

        static ndd::quant::RegisterQuantizer reg_int4e(ndd::quant::QuantizationLevel::INT4E,
                                                       "int4e",
                                                       std::make_shared<Int4EQuantizer>());

    }  // namespace quant
}  // namespace ndd
