#include "infer_plugin_api.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <unordered_map>

#include "NvInfer.h"

#include "util.h"

#include "debug_plugin/debug_dynamic_plugin.h"
#include "debug_plugin/debug_plugin.h"

using namespace nvinfer1;
using namespace std;

BEGIN_LIB_NAMESPACE {
BEGIN_PLUGIN_NAMESPACE {

ILogger* mLogger{};


REGISTER_TENSORRT_PLUGIN(DebugPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(DebugPluginCreator);

} // namespace plugin

static std::unordered_map<std::string, nvinfer1::IPluginCreator*> gPluginRegistry;

extern "C" {

nvinfer1::IPluginV2* createPluginInternal(const char *layer_name, const string plugin_name,
                                          map<string, const void *> params,
                                          map<string, const nvinfer1::Weights> weights) {
  // step1: get creator
  auto it = gPluginRegistry.find(plugin_name);
  if (it == gPluginRegistry.end())
    FLOG("Can not find plugin creator by plugin_name: %s", plugin_name.c_str());

  auto creator = gPluginRegistry[plugin_name];

  // step2: prepare parms and weights
  int fcs_num = params.size() + weights.size();
  CHECK(fcs_num > 0, "Error! fcs_num == 0")

  std::vector<PluginField> fcs(fcs_num);

  // the first param is data_type
  int fcs_idx = 0;
  for (auto it = params.begin(); it != params.end(); it ++) {
    fcs[fcs_idx++] = PluginField{it->first.c_str(), it->second};
  }

  for (auto it = weights.begin(); it != weights.end(); it ++) {
    auto w = it->second;
    fcs[fcs_idx++] = PluginField{it->first.c_str(), w.values, PluginFieldType::kFLOAT32,
                                 static_cast<int>(w.count)};
  }

  PluginFieldCollection fc{fcs_num, fcs.data()};
  return creator->createPlugin(layer_name, &fc);
}

/*******************************Debug************************************/
nvinfer1::IPluginV2* createDebugPluginInernal(string plugin_name, const char *layer_name,
                                              const int input_num) {
  map<string, const void *> params;
  map<string, const nvinfer1::Weights> weights;

  int data_type = 0;
  params["type_id"] = &data_type;
  params["input_num"] = &input_num;

  return createPluginInternal(layer_name, plugin_name, params, weights);
}

nvinfer1::IPluginV2* createDebugPlugin(const char *layer_name, const int input_num)
{
  std::string plugin_name = "DebugPlugin";
  return createDebugPluginInernal(plugin_name, layer_name, input_num);
}

nvinfer1::IPluginV2* createDebugDynamicPlugin(const char *layer_name, const int input_num)
{
  std::string plugin_name = "DebugDynamicPlugin";
  return createDebugPluginInernal(plugin_name, layer_name, input_num);
}

bool initLibDiyInferPlugins(void* logger, const char* libNamespace)
{
  // The function can only be called once
  if (!gPluginRegistry.empty()) {
    return false;
  }

  int num_creators = 0;
  std::string plgin_list_str = "[ ";
  auto tmp_list = getPluginRegistry()->getPluginCreatorList(&num_creators);

  for (int k = 0; k < num_creators; ++k)
  {
    if (!tmp_list[k])
      gLogError << "Plugin Creator for plugin " << k << " is a nullptr!\n";
    std::string plugin_name = tmp_list[k]->getPluginName();
    gPluginRegistry[plugin_name] = tmp_list[k];
    plgin_list_str += plugin_name + " ";
  }
  plgin_list_str += " ]";
  if (num_creators == 0) {
    gLogWarning << "initLibxxInferPlugins() warning! num_creators="
                << num_creators << std::endl;
  } else {
    gLogInfo << "initLibxxInferPlugins() success! num_creators="
                << num_creators << std::endl;
  }
  gLogInfo << "Support plugin list: " << plgin_list_str << std::endl;

  return true;
}
} // extern "C"

} // 
