#include "debug_dynamic_plugin.h"

#include <cassert>
#include <cstring>
#include <cstdint>
#include <iomanip>
#include <vector>

#include "debug_kernel.h"
#include "plugin_common.h"
#include "serialize.hpp"

using namespace nvinfer1;
using namespace std;

BEGIN_LIB_NAMESPACE {
BEGIN_PLUGIN_NAMESPACE {

namespace
{
static const char* DEBUG_PLUGIN_VERSION{"1"};
static const char* DEBUG_PLUGIN_NAME{"DebugDynamicPlugin"};
} // namespace

// Static class fields initialization
PluginFieldCollection DebugPluginDynamicCreator::mFC{};
std::vector<PluginField> DebugPluginDynamicCreator::mPluginAttributes;

/*REGISTER_TENSORRT_PLUGIN(DebugPluginDynamicCreator);*/

DebugPluginDynamic::DebugPluginDynamic(const std::string name, const DataType data_type, int input_num)
    : layer_name_(name)
    , data_type_(data_type)
    , num_inputs_(input_num)
{  }

DebugPluginDynamic::DebugPluginDynamic(const std::string name, const void* data, size_t length)
  : layer_name_(name)
{
  deserialize_value(&data, &length, &data_type_);
  deserialize_value(&data, &length, &num_inputs_);

  size_t name_len = 0;
  deserialize_value(&data, &length, &name_len);

  const char *d = static_cast<const char*>(data); 
  char tmp[name_len];
  deserToHost(d, tmp, name_len);
  layer_name_.resize(name_len);
  layer_name_ = std::string(tmp);
  gLogVerbose << "Starting to deserialize DEBUG plugin: " << layer_name_ << std::endl;
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* DebugPluginDynamic::clone() const
{
  return new DebugPluginDynamic(layer_name_, data_type_, num_inputs_);
}

nvinfer1::DimsExprs DebugPluginDynamic::getOutputDimensions(
  int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{
  return inputs[outputIndex];
}

bool DebugPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
  return true;
}

void DebugPluginDynamic::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
  assert(data_type_ == in[0].desc.type);
}

size_t DebugPluginDynamic::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const
{
  return 0;
}

int DebugPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
  for (size_t n = 0; n < num_inputs_; n++) {
    const int inputVolume = common::Volume(inputDesc[n].dims);
    // remove dim = 1 or 0
    vector<int> v_dims;
    for (int i = 0; i < inputDesc[n].dims.nbDims; i++) {
      int d = inputDesc[n].dims.d[i];
      if (d > 1) v_dims.push_back(d);
    }

    if (data_type_ == DataType::kFLOAT) {
        const float* input = static_cast<const float*>(inputs[n]);

        int p_size = 100;
        if (v_dims[v_dims.size()-1] < p_size) 
          p_size = v_dims[v_dims.size()-1];
        /*checkDeviceData(p_size, input, layer_name_.c_str());*/

        float *arr = new float[inputVolume];
        memset(arr, 0, inputVolume*sizeof(float));

        cudaMemcpy(arr, input, inputVolume*sizeof(float), cudaMemcpyDeviceToHost);
        printf("layer_name=%s, dims=%s\n", 
               layer_name_.c_str(), common::Dims2String(inputDesc[n].dims).c_str());

        p_sum(arr, v_dims, layer_name_);
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
nvinfer1::DataType DebugPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
  assert(inputTypes[index] == DataType::kFLOAT || inputTypes[index] == DataType::kHALF);
  return inputTypes[index];
}

// IPluginV2 Methods

const char* DebugPluginDynamic::getPluginType() const
{
  return DEBUG_PLUGIN_NAME;
}

const char* DebugPluginDynamic::getPluginVersion() const
{
  return DEBUG_PLUGIN_VERSION;
}

int DebugPluginDynamic::getNbOutputs() const {
  return num_inputs_;
}

int DebugPluginDynamic::initialize() {
  return 0;
}

void DebugPluginDynamic::terminate()
{  }

size_t DebugPluginDynamic::getSerializationSize() const 
{
  return sizeof(data_type_) + sizeof(num_inputs_) + sizeof(layer_name_.size()) + layer_name_.size();
}

void DebugPluginDynamic::serialize(void* buffer) const
{
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, num_inputs_);
  serialize_value(&buffer, layer_name_.size());

  char* d = static_cast<char*>(buffer);
  serFromHost(d, layer_name_, (size_t)layer_name_.size());
}

void DebugPluginDynamic::destroy()
{
  // This gets called when the network containing plugin is destroyed
  delete this;
}

void DebugPluginDynamic::setPluginNamespace(const char* libNamespace)
{
  namespace_ = libNamespace;
}

const char* DebugPluginDynamic::getPluginNamespace() const
{
  return namespace_.c_str();
}

///////////////

DebugPluginDynamicCreator::DebugPluginDynamicCreator() {
  // Fill PluginFieldCollection width PluginField arguments metadata
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* DebugPluginDynamicCreator::getPluginName() const {
  return DEBUG_PLUGIN_NAME;
}

const char* DebugPluginDynamicCreator::getPluginVersion() const {
  return DEBUG_PLUGIN_VERSION;
}

const PluginFieldCollection* DebugPluginDynamicCreator::getFieldNames() {
  return &mFC;
}

IPluginV2* DebugPluginDynamicCreator::createPlugin(const char* name, const PluginFieldCollection* fc) {
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

  if (typeId < 0 || typeId > 3) {
    gLogError << "DEBUG: invalid typeId " << typeId << std::endl;
    return nullptr;
  }
  DataType type = static_cast<DataType>(typeId);
  gLogVerbose << "Creating DebugPluginDynamic...\n";
  return new DebugPluginDynamic(name, type, input_num);
}

IPluginV2* DebugPluginDynamicCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
  // This object will be deleted when the network is destroyed, which will
  // call DebugPluginDynamic::destroy()
  return new DebugPluginDynamic(name, serialData, serialLength);
}

void DebugPluginDynamicCreator::setPluginNamespace(const char* libNamespace)
{
  namespace_ = libNamespace;
}

const char* DebugPluginDynamicCreator::getPluginNamespace() const
{
  return namespace_.c_str();
}

} // BEGIN_PLUGIN_NAMESPACE
} // BEGIN_LIB_NAMESPACE

