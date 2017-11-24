# TensorRT Plugin使用方式简介-以leaky relu层为例

------
## 写在前面
TensorRT plugin用于实现TensorRT不支持的网络层，比如leaky relu。本文以leaky relu为例，简单介绍plugin的使用方式以及plugin层的serialization和de-serialization的原理。

之前我已经分享过使用leaky relu曲线救国的解决方案，但是实验结果表明该方案比较慢，leaky relu的plugin实现方式性能更好。

## leaky relu的plugin实现
添加自定义层主要包括两个步骤，
1. 继承IPlugin接口，创建自定义层类
2. 将该自定义层添加到网络中

首先来简单介绍IPlugin接口类的成员函数，详细见TensorRT-3.0.0\include\NvInfer.h文件中的类定义。

``` c++

    // 获得该自定义层的输出个数，比如leaky relu层的输出个数为1
  virtual int getNbOutputs() const = 0;
  
  // 得到输出Tensor的维数
  virtual Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) = 0;

  // 配置该层的参数。该函数在initialize()函数之前被构造器调用。它为该层提供了一个机会，可以根据其权重、尺寸和最大批量大小来做出算法选择。
  virtual void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) = 0;

  // 对该层进行初始化，在engine创建时被调用。
  virtual int initialize() = 0;

  // 该函数在engine被摧毁时被调用
  virtual void terminate() = 0;

    // 获得该层所需的临时显存大小。
  virtual size_t getWorkspaceSize(int maxBatchSize) const = 0;

    // 执行该层
  virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) = 0;

    // 获得该层进行serialization操作所需要的内存大小
  virtual size_t getSerializationSize() = 0;

  // 序列化该层，根据序列化大小getSerializationSize()，将该类的参数和额外内存空间全都写入到系列化buffer中。
  virtual void serialize(void* buffer) = 0;
```
根据类成员函数和leaky relu层的原理，设计LeakyReluPlugin类，可以很容易计算出的成员变量和各个成员函数的返回值。LeakyReluPlugin类实现代码如下。
``` c++

__global__ void _leakyReluKer(float const *in, float *out, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= size)
        return ;

    if (in[index] < 0)
        out[index] = in[index] * 0.1;
    else
        out[index] = in[index];
}

class LeakyReluPlugin : public IPlugin
{
public:
    LeakyReluPlugin() {}

    // 输出个数为1
    int getNbOutputs() const override
    {
        return 1;
    }
    // getOutputDimensions()=输入维度
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(nbInputDims == 1);
        assert(index == 0);
        assert(inputs[index].nbDims == 3);
        return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    }

    int initialize() override
    {
        return 0;
    }

    void terminate() override
    {
    }
    
    // 不需要额外的内存空间
    size_t getWorkspaceSize(int) const override
    {
        return 0;
    }

    // 执行_leakyReluKer核函数
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
    {
        int block_size = 256;
        int grid_size = (mSize + block_size - 1) / block_size;
        _leakyReluKer<<<grid_size, block_size>>>(
            reinterpret_cast<float const*>(inputs[0]), 
            reinterpret_cast<float*>(outputs[0]), mSize);
        getLastCudaError("_leakyReluKer");
        return 0;
    }
    
    // 只有一个成员变量
    size_t getSerializationSize() override
    {
        return sizeof(mSize);
    }
    
    // 把成员变量写入到buffer中，顺序自定义，但要与反序列化时一致。 
    void serialize(void* buffer) override
    {
        *reinterpret_cast<size_t*>(buffer) = mSize;
    }

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)    override
    {
        mSize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
    }

protected:
    // 所需成员变量：该层的总大小，因为拿不到输入维度的大小。
    size_t mSize;
};
```

然后插入到网络中即可，代码如下。

```
LeakyReluPlugin *lr = new LeakyReluPlugin();
auto plugin_lr = network->addPlugin(&inputs_v[0], 1, *lr);
plugin_lr->setName(PLUGIN_LEAKY_RELU_NAME);
```

然后运行网络即可。

## plugin层的serialization和deserialization的详解
plugin的创建和使用的文档比较健全，照着文档来就行了。但序列化和反序列化这一部分文档中说的比较少，故在这里做详解。

序列化非常简单，在plugin类中实现getSerializationSize()和serialize()函数，然后一行代码序列化即可。
gieModelStream = engine_->serialize();

重点在于反序列化，反序列化的步骤如下。
1. 根据序列化serialize()函数内的写入buffer的顺序构建IPluginFactory类。
2. 在反序列化时将IPluginFactory传入，用于将buffer中的数据反序列化为自定义层类。

IPluginFactory接口类代码解释如下。
请注意layerName参数。
```
class IPluginFactory
{
public:
  /**
   * \brief 根据序列化数据，反序列化为plugin类
   *
   * \param 网络层的名字，该参数非常重要，是反序列化为哪种plugin类的唯一凭证。
   * \param 序列化数据
   * \param 该层序列化后的序列化数据的长度
   *
   * \return the plugin
   *
   * \see IPlugin::serialize()
   */
  virtual IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) = 0;
};

```
以leaky relu为例，PluginFactory类实现如下。

```
class PluginFactory : public nvinfer1::IPluginFactory
{
public:
    // deserialization plugin implementation
    IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
    {       
        IPlugin *plugin = nullptr;
        if (strstr(layerName, PLUGIN_LEAKY_RELU_NAME) != NULL)
        {
            plugin = new LeakyReluPlugin(serialData, serialLength);
        }

        return plugin;
    }


    std::unique_ptr<LeakyReluPlugin> mLR{ nullptr };
};
```
然后在deserialize的时候，将PluginFactory传入即可，代码如下。

```
engine_ = runtime->deserializeCudaEngine(buffer, length, &pluginFactory);
```
**实验结果表明，leaky relu的plugin实现方式速度明显快于曲线救国的实现方式！**

