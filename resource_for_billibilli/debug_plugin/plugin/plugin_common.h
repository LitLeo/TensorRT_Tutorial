/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRT_PLUGIN_UTIL_H
#define TRT_PLUGIN_UTIL_H

#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cuda_fp16.hpp"

#include <numeric>
#include <vector>
#include <string>

#include "logger.h"
#include "half.h"

#define TRT_UNUSED (void)

// CUDA: various checks for different function calls.
#ifndef CUDA_CHECK
#define CUDA_CHECK(status) \
  if (status != cudaSuccess) { \
    gLogFatal << "Cuda failure! Error=" << cudaGetErrorString(status) << std::endl; \
  }
#endif

// cublas: various checks for different function calls.
#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(status) \
  if (status != CUBLAS_STATUS_SUCCESS) { \
    gLogFatal << "Cublas failure! Error=" <<  status << std::endl; \
  }
#endif

#define LOGG_NOT_SUPPORT(message) { \
    gLogError << message << std::endl; \
    assert(-1); \
  }

typedef __half half;

// if cuda version > 10
//constexpr uint32_t BDIM = 1; // batch dimension
//constexpr uint32_t SDIM = 0; // seq len dimension
constexpr uint32_t BDIM = 0; // batch dimension
constexpr uint32_t SDIM = 1; // seq len dimension
constexpr uint32_t HDIM = 2; // hidden dimension
constexpr uint32_t BERT_BDIM = 0; // batch dimension
constexpr uint32_t BERT_SDIM = 1; // seq len dimension
constexpr uint32_t BERT_HDIM = 2; // hidden dimension

constexpr uint32_t DPRNN_BDIM = 0; // batch dimension
constexpr uint32_t DPRNN_CDIM = 2; // channel dimension
constexpr uint32_t DPRNN_NDIM = 2; // N dimension
constexpr uint32_t DPRNN_TDIM = 4; // T dimension

#define HDI inline __host__ __device__

// for getWorkspaceSize
constexpr size_t kAlignment = 256;

template <typename IntType>
constexpr HDI IntType ceildiv(IntType a, IntType b)
{
    return (a + b - 1) / b;
}
template <typename IntType>
constexpr HDI IntType alignTo(IntType a, IntType b)
{
    return ceildiv(a, b) * b;
}

template <typename T>
inline T* deserToDev(const char*& buffer, size_t nbElem)
{
    T* dev = nullptr;
    const size_t len = sizeof(T) * nbElem;
    CUDA_CHECK(cudaMalloc(&dev, len));
    CUDA_CHECK(cudaMemcpy(dev, buffer, len, cudaMemcpyHostToDevice));

    buffer += len;
    return dev;
}

template <typename T>
inline void serFromDev(char*& buffer, const T* data, size_t nbElem)
{
    const size_t len = sizeof(T) * nbElem;
    CUDA_CHECK(cudaMemcpy(buffer, data, len, cudaMemcpyDeviceToHost));
    buffer += len;
}

template <typename T>
inline void deserToHost(const char*& buffer, T* data,
                        size_t name_len)
{
    ::memcpy(data, buffer, name_len);
    buffer += name_len;
}

inline void serFromHost(char*& buffer, const std::string& name,
                        size_t name_len)
{
    //::memcpy(static_cast<void*>(buffer), &name_len, sizeof(name_len));
    //buffer += sizeof(name_len);
    memcpy(buffer, name.c_str(), name_len);
    buffer += name_len;
}

// deserialize NvInfer Dims from buffer
inline void deserNvDimsToHost(const char*& buffer, nvinfer1::Dims &dims) {
  ::memcpy(&dims.nbDims, buffer, sizeof(int));
  buffer += sizeof(int);

  int offset = sizeof(int) * dims.MAX_DIMS;
  ::memcpy(dims.d, buffer, offset);
  buffer += offset;
}

// serialize NvInfer Dims to buffer
inline void serNvDimsFromHost(char*& buffer, const nvinfer1::Dims &dims) {
  int offset = sizeof(dims.nbDims);
  ::memcpy(buffer, &dims.nbDims, offset);
  buffer += offset;

  offset = sizeof(int) * dims.MAX_DIMS;
  ::memcpy(buffer, dims.d, offset);
  buffer += offset;
}

inline void checkDeviceData(int N, const float *B, const char *message) {
  int pl = N * sizeof(float);
  float b[N];
  cudaMemcpy(b, B, pl, cudaMemcpyDeviceToHost);
  float sum = 0.0f;
  for (int i = 0; i < N; i++)
      sum += b[i];
  std::cout << message << "[sum="<< sum <<"] :";
  for (int i = 0; i < N; i++)
      std::cout << b[i] << ", ";
  std::cout << std::endl;
}
inline void checkDeviceData(int N, const int *B, const char *message) {
  int pl = N * sizeof(float);
  int b[N];
  cudaMemcpy(b, B, pl, cudaMemcpyDeviceToHost);
  float sum = 0.0f;
  for (int i = 0; i < N; i++)
      sum += b[i];
  std::cout << message << "[sum="<< sum <<"] :";
  for (int i = 0; i < N; i++)
      std::cout << b[i] << ", ";
  std::cout << std::endl;
}

inline const nvinfer1::IDimensionExpr *NvDimsOpMod(nvinfer1::IExprBuilder& expr_builder, const nvinfer1::IDimensionExpr &first, const nvinfer1::IDimensionExpr &second) {
  // mod(a, b) = a - a/b*b
  auto sub1 = expr_builder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV, first, second);
  auto sub2 = expr_builder.operation(nvinfer1::DimensionOperation::kPROD, *sub1, second);
  auto ret = expr_builder.operation(nvinfer1::DimensionOperation::kSUB, first, *sub2);
  return ret;
}

struct CublasConfigHelper
{
    cublasPointerMode_t pm;
    cublasMath_t mm;
    cublasHandle_t cublas;
    CublasConfigHelper(cublasHandle_t cublas_)
        : cublas(cublas_)
    {
        cublasGetPointerMode(cublas, &pm);
        cublasGetMathMode(cublas, &mm);
        cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST);
        cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH);
    }
    ~CublasConfigHelper()
    {
        cublasSetMathMode(cublas, mm);
        cublasSetPointerMode(cublas, pm);
    }
};

inline void convertAndCopyToDevice(const nvinfer1::Weights& src, float* destDev)
{

    size_t wordSize = sizeof(float);
    size_t nbBytes = src.count * wordSize;
    if (src.type == nvinfer1::DataType::kFLOAT)
    {
        gLogVerbose << "Float Weights(Host) => Float Array(Device)" << std::endl;
        CUDA_CHECK(cudaMemcpy(destDev, src.values, nbBytes, cudaMemcpyHostToDevice));
    }
    else
    {
#ifdef __SCORE_HALF__
        gLogVerbose << "Half Weights(Host) => Float Array(Device)" << std::endl;
        std::vector<float> tmp(src.count);
        const half* values = reinterpret_cast<const half*>(src.values);

        for (int it = 0; it < tmp.size(); it++)
        {
            tmp[it] = __half2float(values[it]);
        }

        CUDA_CHECK(cudaMemcpy(destDev, &tmp[0], nbBytes, cudaMemcpyHostToDevice));
#else
        LOGG_NOT_SUPPORT("half not support!")
#endif
    }
}

inline void convertAndCopyToDevice(const nvinfer1::Weights& src, half* destDev)
{
    size_t wordSize = sizeof(half);
    size_t nbBytes = src.count * wordSize;
    if (src.type == nvinfer1::DataType::kHALF)
    {
        gLogVerbose << "Half Weights(Host) => Half Array(Device)" << std::endl;
        CUDA_CHECK(cudaMemcpy(destDev, src.values, nbBytes, cudaMemcpyHostToDevice));
    }
    else
    {
#ifdef __SCORE_HALF__
        gLogVerbose << "Float Weights(Host) => Half Array(Device)" << std::endl;
        std::vector<half> tmp(src.count);
        const float* values = reinterpret_cast<const float*>(src.values);

        for (int it = 0; it < tmp.size(); it++)
        {
            tmp[it] = __float2half(values[it]);
        }
        CUDA_CHECK(cudaMemcpy(destDev, &tmp[0], nbBytes, cudaMemcpyHostToDevice));
#else
        LOGG_NOT_SUPPORT("half not support!")
#endif
    }
}

inline nvinfer1::DataType fieldTypeToDataType(const nvinfer1::PluginFieldType ftype)
{
    switch (ftype)
    {
    case nvinfer1::PluginFieldType::kFLOAT32:
    {
        gLogVerbose << "PluginFieldType is Float32" << std::endl;
        return nvinfer1::DataType::kFLOAT;
    }
    case nvinfer1::PluginFieldType::kFLOAT16:
    {
        gLogVerbose << "PluginFieldType is Float16" << std::endl;
        return nvinfer1::DataType::kHALF;
    }
    case nvinfer1::PluginFieldType::kINT32:
    {
        gLogVerbose << "PluginFieldType is Int32" << std::endl;
        return nvinfer1::DataType::kINT32;
    }
    case nvinfer1::PluginFieldType::kINT8:
    {
        gLogVerbose << "PluginFieldType is Int8" << std::endl;
        return nvinfer1::DataType::kINT8;
    }
    default: throw std::invalid_argument("No corresponding datatype for plugin field type");
    }
}

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline std::string dims2String(const nvinfer1::Dims &d) {
  std::string str = "[";
  for (int i = 0; i < d.nbDims-1; i++) {
    str += std::to_string(d.d[i]) + ", ";
  }
  str += std::to_string(d.d[d.nbDims-1]) + "]";
  return str;
}

#endif // TRT_PLUGIN_UTIL_H
