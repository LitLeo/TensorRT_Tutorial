#ifndef PLUGIN_DEBUG_KERNEL_H_
#define PLUGIN_DEBUG_KERNEL_H_

#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "plugin_common.h"

namespace debug_plugin {

void p(const float *data, std::vector<int>& dims);

void p_sum(const float *data, std::vector<int>& dims, std::string message);

} // debug_plugin

#endif // PLUGIN_DEBUG_KERNEL_H_
