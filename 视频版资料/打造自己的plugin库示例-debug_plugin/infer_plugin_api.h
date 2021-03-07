#ifndef INFER_PLUGIN_API_H_
#define INFER_PLUGIN_API_H_

#include "NvInfer.h"

//!
//! \file NvInferPlugin.h
//!
//! This is the API for the TRT_DIY plugins.
//!

namespace my_plugin {

extern "C"
{
  nvinfer1::IPluginV2* createDebugDynamicPlugin(const char* layer_name, const int input_num);
  nvinfer1::IPluginV2* createDebugPlugin(const char* layer_name, const int input_num);

  bool initLibDiyInferPlugins(void* logger, const char* libNamespace);

}

} // 

#endif // INFER_PLUGIN_API_H_
