#ifndef PLUGIN_DEBUG_KERNEL_H_
#define PLUGIN_DEBUG_KERNEL_H_

#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "common.h"
#include "plugin_common.h"

BEGIN_LIB_NAMESPACE {
BEGIN_PLUGIN_NAMESPACE {

void p(const float *data, std::vector<int>& dims);

void p_sum(const float *data, std::vector<int>& dims, std::string message);

} // BEGIN_PLUGIN_NAMESPACE
} // BEGIN_LIB_NAMESPACE

#endif // PLUGIN_DEBUG_KERNEL_H_
