#ifndef PLUGIN_DEBUG_PLUGIN_H_
#define PLUGIN_DEBUG_PLUGIN_H_

#include "NvInferPlugin.h"
#include "NvInferRuntime.h"

#include <string>
#include <vector>

#include "plugin_common.h"

namespace debug_plugin {

class DebugPlugin : public nvinfer1::IPluginV2IOExt {
 public:
  DebugPlugin(const std::string &name, const nvinfer1::DataType type, int input_num,
              std::vector<nvinfer1::Dims> outputs_dims = std::vector<nvinfer1::Dims>());

  DebugPlugin(const std::string &name, const void* data, size_t length);

  // It makes no sense to construct DebugPlugin without arguments.
  DebugPlugin() = delete;

  virtual ~DebugPlugin() {}

 public:
  int getNbOutputs() const override;
  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override;
  int initialize() override;
  void terminate() override;
  size_t getWorkspaceSize(int maxBatchSize) const override;
  int enqueue(int batchSize, const void* const* inputs, void** outputs,
              void* workspace, cudaStream_t stream) override;
  size_t getSerializationSize() const override;
  void serialize(void* buffer) const override;
  void configurePlugin(const nvinfer1::PluginTensorDesc* in, int nbInput,
                       const nvinfer1::PluginTensorDesc* out, int nbOutput) override;
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs, int nbOutputs) const override;
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const override;
  const char* getPluginType() const override;
  const char* getPluginVersion() const override;
  void destroy() override;
  nvinfer1::IPluginV2Ext* clone() const override;
  void setPluginNamespace(const char* libNamespace) override;
  const char* getPluginNamespace() const override;
  bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted,
                                    int nbInputs) const override;
  bool canBroadcastInputAcrossBatch(int inputIndex) const override;

 private:
  std::string layer_name_;
  std::string namespace_;
  nvinfer1::DataType data_type_;

  size_t num_inputs_;

  std::vector<nvinfer1::Dims> outputs_dims_;

 protected:
  // To prevent compiler warnings.
  using nvinfer1::IPluginV2IOExt::configurePlugin;
};

class DebugPluginCreator: public nvinfer1::IPluginCreator {
 public:
  const char* getPluginName() const override;
  const char* getPluginVersion() const override;
  const nvinfer1::PluginFieldCollection* getFieldNames() override;
  nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override;
  nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;
  void setPluginNamespace(const char* libNamespace) override;
  const char* getPluginNamespace() const override;

 private:
  std::string namespace_;
  std::string plugin_name_;
  nvinfer1::PluginFieldCollection field_collection_{0, nullptr};
};

} // debug_plugin

#endif // PLUGIN_DEBUG_PLUGIN_H_
