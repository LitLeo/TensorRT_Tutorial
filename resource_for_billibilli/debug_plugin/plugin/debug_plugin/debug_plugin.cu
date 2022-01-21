#include "debug_plugin.h"

#include <cassert>
#include <string>
#include <vector>

#include "NvInfer.h"

#include "debug_kernel.h"
#include "serialize.hpp"

using namespace nvinfer1;
using namespace std;

namespace debug_plugin {

// Clip plugin specific constants
namespace
{
static const char* DEBUG_VERSION{"1"};
static const char* DEBUG_NAME{"DebugPlugin"};
} // namespace

/*REGISTER_TENSORRT_PLUGIN(DebugPluginCreator);*/

DebugPlugin::DebugPlugin(const std::string &name, const DataType data_type, int input_num,
                         std::vector<nvinfer1::Dims> outputs_dims)
    : layer_name_(name)
    , data_type_(data_type)
    , num_inputs_(input_num)
    , outputs_dims_(outputs_dims)
{  }

DebugPlugin::DebugPlugin(const std::string &name, const void* data, size_t length)
  : layer_name_(name) {
  deserialize_value(&data, &length, &data_type_);
  deserialize_value(&data, &length, &num_inputs_);
  size_t name_len = 0;
  deserialize_value(&data, &length, &name_len);

  // deserialize dims
  size_t outputs_dims_size = 0;
  deserialize_value(&data, &length, &outputs_dims_size);

  outputs_dims_.resize(outputs_dims_size);
  const char *d = static_cast<const char*>(data); 

  for (int i = 0; i < outputs_dims_size; i++) {
    deserNvDimsToHost(d, outputs_dims_[i]);
  }

  char tmp[name_len];
  deserToHost(d, tmp, name_len);
  layer_name_.resize(name_len);
  layer_name_ = std::string(tmp);
  gLogVerbose << "Starting to deserialize DEBUG plugin: " << layer_name_ << std::endl;
}

IPluginV2Ext* DebugPlugin::clone() const {
  auto p = new DebugPlugin(layer_name_, data_type_, num_inputs_, outputs_dims_);
  return p;
}

Dims DebugPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) {
  outputs_dims_.push_back(inputs[index]);
  return inputs[index];
}

bool DebugPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, 
                                            int nbInputs, int nbOutputs) const {
  return true;
}

void DebugPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, 
                                  const PluginTensorDesc* out, int nbOutput) 
{  }

size_t DebugPlugin::getWorkspaceSize(int maxBatchSize) const {
  return 0;
}

int DebugPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, 
                         void* workspace, cudaStream_t stream)  {

  for (size_t n = 0; n < num_inputs_; n++) {
    auto dims = outputs_dims_[n];
    const int inputVolume = volume(dims) * batchSize;
    // remove dim = 1 or 0
    vector<int> v_dims;
    v_dims.push_back(batchSize);
    for (int i = 0; i < dims.nbDims; i++) {
      int d = dims.d[i];
      if (d > 1) v_dims.push_back(d);
    }

    if (data_type_ == DataType::kFLOAT) {
        const float* input = static_cast<const float*>(inputs[n]);
        float *arr = new float[inputVolume];
        memset(arr, 0, inputVolume*sizeof(float));

        cudaMemcpy(arr, input, inputVolume*sizeof(float), cudaMemcpyDeviceToHost);
        printf("layer_name=%s, dims=%s\n", 
               layer_name_.c_str(), dims2String(dims).c_str());

        p(arr, v_dims);
        delete [] arr;

        float* output = static_cast<float*>(outputs[n]);
        cudaMemcpy(output, input, inputVolume*sizeof(float), cudaMemcpyDeviceToDevice);

    } else if (data_type_ == DataType::kHALF) {
#ifdef __SCORE_HALF__
        const half* input = static_cast<const half*>(inputs[0]);
#endif
    } else {
        assert(false);
    }
  }

  return 0;
}

// IPluginV2Ext Methods
DataType DebugPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const {
  assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
  return inputTypes[0];
}

const char* DebugPlugin::getPluginType() const {
  return DEBUG_NAME;
}

const char* DebugPlugin::getPluginVersion() const {
  return DEBUG_VERSION;
}

int DebugPlugin::getNbOutputs() const {
  return num_inputs_;
}

int DebugPlugin::initialize() {
  return 0;
}

void DebugPlugin::terminate()
{  }

size_t DebugPlugin::getSerializationSize() const
{
  return sizeof(data_type_) + sizeof(num_inputs_) + 
         sizeof(int) * outputs_dims_.size() * (nvinfer1::Dims::MAX_DIMS+ 1) +
         sizeof(layer_name_.size()) + layer_name_.size() + 10;
}

void DebugPlugin::serialize(void* buffer) const {
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, num_inputs_);
  serialize_value(&buffer, layer_name_.size());

  serialize_value(&buffer, outputs_dims_.size());
  char *d = static_cast<char*>(buffer);
  for (size_t i = 0; i < outputs_dims_.size(); i++) {
    serNvDimsFromHost(d, outputs_dims_[i]);
  }

  serFromHost(d, layer_name_, (size_t)layer_name_.size());
}

void DebugPlugin::destroy() {
  delete this;
}

void DebugPlugin::setPluginNamespace(const char* libNamespace) {
  namespace_ = libNamespace;
}

const char* DebugPlugin::getPluginNamespace() const {
  return namespace_.c_str();
}

bool DebugPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const  {
    return false;
}

bool DebugPlugin::canBroadcastInputAcrossBatch(int inputIndex) const  {
    return false;
}

const char* DebugPluginCreator::getPluginName() const {
  return DEBUG_NAME;
}

const char* DebugPluginCreator::getPluginVersion() const {
  return DEBUG_VERSION;
}

const PluginFieldCollection* DebugPluginCreator::getFieldNames() {
  return &field_collection_;
}

IPluginV2* DebugPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) {
  gLogVerbose << "Creating DebugPlugin...\n";

  int typeId = -1;
  int input_num = 0;
  for (int i = 0; i < fc->nbFields; i++) {
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("type_id") == 0) {
      typeId = *static_cast<const int*>(fc->fields[i].data);
      gLogVerbose << "Building typeId: " << typeId << std::endl;
    }
    if (field_name.compare("input_num") == 0) {
      input_num = *static_cast<const int*>(fc->fields[i].data);
      gLogVerbose << "Building input_num: " << input_num << std::endl;
    }
  }

  if (typeId < 0 || typeId > 2) {
    gLogError << "DEBUG: invalid typeId " << typeId << std::endl;
    return nullptr;
  }
  DataType type = static_cast<DataType>(typeId);
  gLogVerbose << "Creating DebugPlugin...\n";
  return new DebugPlugin(name, type, input_num);
}

IPluginV2* DebugPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) {
  return new DebugPlugin(name, serialData, serialLength);
}

void DebugPluginCreator::setPluginNamespace(const char* libNamespace) {
  namespace_ = libNamespace;
}

const char* DebugPluginCreator::getPluginNamespace() const {
  return namespace_.c_str();
}

} // debug_plugin

