
#include <iostream>
#include <algorithm>
#include <numeric>

#include "SkipLayerNormV2Plugin.h"
#include "common.cuh"

using namespace nvinfer1;

PluginFieldCollection SkipLayerNormV2PluginCreator::fc_{};
std::vector<PluginField> SkipLayerNormV2PluginCreator::attr_;

template <typename T, unsigned TPB>
__global__ void skip_layer_norm_v2_kernel_small(
    const int ld, const T* input, const T* skip, const T* beta, const T* gamma, T* output, T* add_output)
{

    const T rld = T(1) / T(ld);
    const int offset = blockIdx.x * ld;

    cub::Sum pairSum;
    // reduce x and x^2
    kvp<T> threadData(0, 0);
    const int idx = offset + threadIdx.x;
    T val = 0;

    if (threadIdx.x < ld)
    {

        val = input[idx] + skip[idx];
        add_output[idx] = val;

        const T rldval = rld * val;
        threadData = pairSum(threadData, kvp<T>(rldval, rldval * val));
    }

    layerNormSmall<T, T, TPB>(val, threadData, ld, idx, beta, gamma, output);
}

template <typename T, unsigned TPB>
__global__ void skip_layer_norm_v2_kernel(
    const int ld, const T* input, const T* skip, const T* beta, const T* gamma, T* output, T* add_output)
{
    const T rld = T(1) / T(ld);
    const int offset = blockIdx.x * ld;

    cub::Sum pairSum;
    // reduce x and x^2
    kvp<T> threadData(0, 0);

    for (int i = threadIdx.x; i < ld; i += TPB)
    {
        const int idx = offset + i;
        T val = T(input[idx] + skip[idx]);
        add_output[idx] = val;

        const T rldval = rld * val;
        threadData = pairSum(threadData, kvp<T>(rldval, rldval * val));
        output[idx] = val;
    }

    layerNorm<T, T, T, TPB>(threadData, ld, offset, beta, gamma, output);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("%f %f %f %f\n", __half2float(gamma[0]), __half2float(beta[0]), __half2float(input[0]), __half2float(output[0]));
    }
}

template <typename T>
int compute_skip_layer_norm_v2_tpl(cudaStream_t stream, const int ld, const int n, 
        const T* input, const T* skip, const T* beta, const T* gamma, T* output, T* add_output) {

    // this must be true because n is the total size of the tensor
    assert(n % ld == 0);
    const int gridSize = n / ld;
    /*constexpr int VPT = 16 / sizeof(T);*/
    if (ld <= 32) {
        constexpr int blockSize = 32;
        skip_layer_norm_v2_kernel_small<T, blockSize>
            <<<gridSize, blockSize, 0, stream>>>(ld, input, skip, beta, gamma, output, add_output);
    } else if (ld <= 128) {
        constexpr int blockSize = 128;
        skip_layer_norm_v2_kernel_small<T, blockSize>
            <<<gridSize, blockSize, 0, stream>>>(ld, input, skip, beta, gamma, output, add_output);
    } else if (ld <= 384) {
        constexpr int blockSize = 384;
        skip_layer_norm_v2_kernel_small<T, blockSize>
            <<<gridSize, blockSize, 0, stream>>>(ld, input, skip, beta, gamma, output, add_output);
    } else {
        constexpr int blockSize = 256;
        skip_layer_norm_v2_kernel<T, blockSize>
            <<<gridSize, blockSize, 0, stream>>>(ld, input, skip, beta, gamma, output, add_output);
    }
    (cudaPeekAtLastError());

    return 0;
}

int compute_skip_layer_norm_v2(cudaStream_t stream, const int ld, const int n, const float* input,
                       const float* skip, const float* gamma, const float* beta, float* output, float* add_output) {
    return compute_skip_layer_norm_v2_tpl<float>(stream, ld, n, input, skip, beta, gamma, output, add_output);
}

int compute_skip_layer_norm_v2(cudaStream_t stream, const int ld, const int n, const half* input,
                       const half* skip, const half* gamma, const half* beta, half* output, half* add_output) {
    return compute_skip_layer_norm_v2_tpl<half>(stream, ld, n, input, skip, beta, gamma, output, add_output);
}

inline int64_t volume(const nvinfer1::Dims& d) {
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

int32_t SkipLayerNormV2Plugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    /*const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];*/

    /*layerNormKernel <<<nBlock, 128, 0, stream>>>((float *)inputs[0], (float *)outputs[0]);*/

  const int input_volume = volume(inputDesc[0].dims);
  const int dim = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];
  const int S = input_volume / dim;

  int status = -1;

  /*const size_t word_size = getElementSize(DataType::kFLOAT);*/

  // Our plugin outputs only one tensor
  const float* input = static_cast<const float*>(inputs[0]);
  const float* skip = static_cast<const float*>(inputs[1]);
  const float* gamma_ptr = static_cast<const float*>(inputs[2]);
  const float* beta_ptr = static_cast<const float*>(inputs[3]);
  float* output = static_cast<float*>(outputs[0]);
  float* add_output = static_cast<float*>(outputs[1]);

  status = compute_skip_layer_norm_v2(stream, dim, input_volume, input, skip, gamma_ptr, beta_ptr, output, add_output);

  return 0;
}


REGISTER_TENSORRT_PLUGIN(SkipLayerNormV2PluginCreator);

