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
                return int8::get_storage_size(dimension);
            }

            inline float extract_scale(const uint8_t* buffer, size_t dimension) {
                return int8::extract_scale(buffer, dimension);
            }

            // Quantize using floor instead of nearest rounding.
            inline std::vector<uint8_t>
            quantize_vector_fp32_to_int8_buffer(const std::vector<float>& input) {
                if(input.empty()) {
                    return std::vector<uint8_t>();
                }

                std::vector<float> rotated = input;
                rotate_pairwise_inplace(rotated);

                size_t dimension = rotated.size();
                size_t buffer_size = get_storage_size(dimension);
                std::vector<uint8_t> buffer(buffer_size);

                float abs_max = ndd::quant::math::find_abs_max(rotated.data(), dimension);
                if(abs_max == 0.0f) {
                    abs_max = 1.0f;
                }
                float scale = abs_max / INT8_SCALE;
                float inv_scale = 1.0f / scale;

                int8_t* data_ptr = reinterpret_cast<int8_t*>(buffer.data());
                for(size_t i = 0; i < dimension; ++i) {
                    float scaled = std::floor(rotated[i] * inv_scale);
                    float clamped = std::max(-127.0f, std::min(127.0f, scaled));
                    data_ptr[i] = static_cast<int8_t>(clamped);
                }

                float* scale_ptr =
                        reinterpret_cast<float*>(buffer.data() + (dimension * sizeof(int8_t)));
                *scale_ptr = scale;

                return buffer;
            }

            inline std::vector<uint8_t>
            quantize_vector_fp32_to_int8_buffer_auto(const std::vector<float>& input) {
                return quantize_vector_fp32_to_int8_buffer(input);
            }

            inline std::vector<float> dequantize_int8_buffer_to_fp32(const uint8_t* buffer,
                                                                     size_t dimension) {
                std::vector<float> out = int8::dequantize_int8_buffer_to_fp32(buffer, dimension);
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
                return int8::L2Sqr(pVect1v, pVect2v, qty_ptr);
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

                int64_t dot = 0;
                int64_t sum1 = 0;
                int64_t sum2 = 0;
                for(size_t i = 0; i < qty; ++i) {
                    const int32_t a = static_cast<int32_t>(pVect1[i]);
                    const int32_t b = static_cast<int32_t>(pVect2[i]);
                    dot += static_cast<int64_t>(a) * b;
                    sum1 += a;
                    sum2 += b;
                }

                const float base = static_cast<float>(dot) * scale1 * scale2;
                const float correction =
                        0.5f * static_cast<float>(sum1 + sum2) * scale1 * scale2;
                const float omitted_mean = 0.25f * static_cast<float>(qty) * scale1 * scale2;
                return base + correction + omitted_mean;
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
                int8::L2SqrSimBatch(query, vectors, count, params, out);
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

            static std::vector<uint8_t> quantize_to_int8_identity(const void* in, size_t dim) {
                size_t size = get_storage_size(dim);
                const uint8_t* ptr = static_cast<const uint8_t*>(in);
                return std::vector<uint8_t>(ptr, ptr + size);
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
                d.quantize_to_int8 = &int8e::quantize_to_int8_identity;
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
