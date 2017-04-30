# TensorRT 2.1.0 User Guide

---

[toc]

## 介绍
NVIDIA TensorRT是一个C++库，在NVIDIA GPU上能够实现高性能的推理（inference ）过程。TensorRT优化网络的方式有：对张量和层进行合并，转换权重，选择高效的中间数据格式，以及依据层的参数和实测性能，从一个丰富的核仓库中进行筛选。

编译TensorRT 2.1.0 要求GCC >= 4.8

TensorRT 2.1.0 现在支持以下layer类型：

 - **Convolution**：卷积层，可无bias。目前仅支持2D卷积（即对4D的输入进行卷积并输出4D输出）。**Note：**该卷积层的操作实际计算的是“相关”而不是“卷积”（严格的卷积定义需要卷积核反转），如果你想通过TensorRT的API而不是通过caffe parser library导入权重，这是一个需要注意的地方。
 - **Activation**: 激活层，支持ReLU, tanh和sigmoid.
 - **Pooling**: 池化层，支持最大值池化和均值池化
 - **Scale**: 可以使用常量对每一个张量, 通道或权重进行仿射变换和取幂操作。**BatchNormalization**可由该层实现。
 - **ElementWise**: 两个张量按元素求和、求乘积或取最大
 - **LRN**: 局部响应归一化层，仅支持通道间归一化
 - **Fully-connected**：全连接层，可无bias
 - **SoftMax**: Softmax层，仅支持通道间计算softmax
 - **Deconvolution**： 反卷积层，可无bias
 - **RNN**： 循环网络层，支持GRU和LSTM

TensorRT是一个独立的深度学习部署框架，对caffe尤其友好。TensorRT提供了一个针对caffe的模型解析器NvCaffeParser，可以通过几行代码解析caffe生成的model并定义网络。NvCaffeParer使用上面定义的层来实现Caffe中的Convolution, ReLU, Sigmoid, TanH, Pooling, Power, BatchNorm, Eltwise, LRN, InnerProduct, SoftMax, Scale, 和Deconvolution。而目前，NvCaffeParse不支持下面的Caffe层：

- Deconvolution groups
- Dilated convolutions
- PReLU
- Leaky ReLU
- 除通道间scale的其他Scale层
- 含有两个以上输入的ElementWise操作

**Note：** TensorRT不支持caffe的旧prototxt格式，特别地，prototxt中定义的层类型应当为由双引号分割的字符串。

## 快速开始指南
【注】本部分由[TensorRT下载页面][1]翻译而来。

TensorRT原名GIE。GIE又名TensorRT 1.0，TensorRT 2.0正式改名。
TensorRT 2.0非常大的改动点是支持INT8类型（TensorRT 1.0支持FP16）。
使用TensorRT 2.0的硬件要求：Tesla P4, Tesla P40, GeForce TitanX Pascal, GeForce GTX 1080, DRIVE PX 2 dGPU
软件要求：CUDA 8.0
安装命令：

 1. 验证你是否安装了CUDA 8.0 .
 2. 下载TensorRT的deb包
 3. 从TensrRT的deb包安装，命令为：
 ```bash
 sudo dpkg -i nv-tensorrt-repo-ubuntu1404-7-ea-cuda8.0_2.0.1-1_amd64.deb
    sudo apt-get update
    sudo apt-get install tensorrt-2
```
 4. 验证安装:
  `dpkg -l | grep tensorrt-2`

 5. 若安装成功，你应看到:
  `tensorrt-2 2.0.0-1+cuda amd64 Meta package of TensorRT`
同样，通过命令：
  `dpkg -l | grep nvinfer2`

 4. 你应看到:
  `libnvinfer2 2.0.0-1+cuda amd64 TensorRT runtime libraries`

注意：TensorRT 2.0现在只提供了Ubuntu 14.04和16.04两个版本。对Ubuntu系统友好，如果是企业级系统（比如centos）可以下载下来解压然后手动安装。
再注意：我Linux学的不好，折腾了一遍，Centos6.x安不上（gcc和libstdc++.so版本问题），建议Centos 7以上

## 快速开始
使用TensorRT包括两部步骤（1）打开冰箱；（2）把大象装进去：

 - build阶段，TensorRT进行网络定义、执行优化并生成推理引擎
 - execution阶段，需要将input和output在GPU上开辟空间并将input传输到GPU上，调用推理接口得到output结果，再将结果拷贝回host端。

build阶段比较耗时间，特别是在嵌入式平台上。所以典型的使用方式就是将build后的引擎序列化（序列化后可写到硬盘里）供以后使用。

build阶段对网络进行了以下优化：

 - 去掉没有被使用过的输出层
 - 将convolution、bias和ReLU操作融合到一起
 - 将相似度比较高的参数和相同的Tensor进行聚合（例如，GoogleNet v5的初始模块中的1*1卷积）
 - 通过将层的输出直接导向其最终位置来简化串接的层

此外，TensorRT在虚拟数据（Dummy Data）上运行层，以在kernel仓库中筛选出运行最快的，并在适当的时候执行权重预格式化和内存优化。

### 网络定义

## SampleMNIST：简单使用方法
### 日志
### Build过程：caffeToGIEModel
### 引擎反序列化（引擎重构）
### Execution过程：doInference()

## SampleGoogleNet:性能分析与16-bit推断
### 性能分析
### half2模式

## SampleINT8：8-bit校准与推断
SampleINT8说明了以8 bit整数（INT8）进行推理的步骤。SampleINT8使用MNIST训练集进行验证，但也可用于校准和评分其他网络。使用以下命令在MNIST上运行示例。
`./sample_int8 mnist`
**注意**：INT8只有在计算能力6.1以上的GPU上使用。

INT8的引擎仍从32-bit（float）的网络定义中构建，但是要比32-bit 和16-bit的引擎复杂的多。具体而言，TensorRT在构建网络时，必须校准网络以确定如何最好的用8-bit表示权重和激活值。这需要一组该网络的代表性的输入数据-校准集（the calibration set）和两个参数， 回归截断（regression cutoff）和分位数（quantile）。

应用程序必须通过实现INT8Calibrator接口来指定校准集和参数。对于ImageNet网络和MNIST，500张图像一个合理的校准集规模。请参考[选择校准集参数](#choose_calibration_parameters)一节查看确定回归截断点与分位数的设置细节。

### IInt8Calibrator接口

`IInt8Calibrator`接口含有为builder指定校准集和校准参数的方法。此外，因为校准是一个需要运行很多次，代价较高的过程，`IInt8Calibrator`还提供了缓存中间值的方法。缓存的细节将在[缓存](#caching)一节讨论。最简单的实现方式是立即从`write()`方法返回，并从`read()`方法中返回`nullptr`。

#### 校准集

一旦校准开始，builder就会调用`getBatchSize()`以获取校准集的Batch Size，校准集的每一个batch数据大小都必须为该值。接着，方法`getBatch()`会被反复调用以获得batch数据，直到它返回false为止：
```C++
bool getBatch(void* bindings[], const char* names[], int nbBindings) override
{
    if (!mStream.next())
        return false;

    CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
    assert(!strcmp(names[0], INPUT_BLOB_NAME));
    bindings[0] = mDeviceInput;
    return true;
}
```
对于每个输入张量，指向GPU内存中数据的指针必须被写入`bindings`数组中，而`names`数组包含了输入张量的名字，`names`数组中的名字与`bindings`数组中的指针按位置一一对应。两个数组的大小都是`nbBindings`。

**注意：**校准集必须能够代表在GIE运行时的输入数据。例如，对图像分类任务而言，校准集不能只由来自一部分类别的图片构成。另外，任何在推断前执行的图像处理过程，如缩放、裁剪或去均值，也必须对校准集的样本执行。

#### 校准集参数
这些方法是很明确的：
```c++
double getQuantile() const override             { return mQuantile; }
double getRegressionCutoff() const override     { return mCutoff; }
```

### 配置Builder
对于INT8推断，输入模型必须由32-bit的权重确定。
```c++
const IBlobNameToTensor* blobNameToTensor = 
    parser->parse(locateFile(deployFile).c_str(),
                  locateFile(modelFile).c_str(),
                   *network,
                   DataType::kFLOAT);
```
builder有额外的两个方法：
```c++
builder->setInt8Mode(true);
builder->setInt8Calibrator(calibrator);
```

一旦模型被builder构建完成，它可以与Float32的网络一样使用：输入和输出仍然是32-bit的浮点数。


<span id = "caching">
###校准集缓存
</span>
校准过程可能较为缓慢，因此`IInt8Calibrator`提供了用于缓存中间结果的方法。高效使用这些方法需要对校准过程的细节有一定了解。

当构建一个INT8的引擎时，builder执行了下面的步骤：

1. 构建一个32-bit的引擎，并在其上运行校准集，对校准集的每一个数据，记录其表示其激活值分布的直方图
2. 由直方图构建一个校准表，并构建截断参数和分位数
3. 从网络定义和校准表构建INT8引擎

直方图与校准表都可以被缓存

当重复构建一个模型多次（例如在不同平台）时，对校准表缓存是很有用的。它捕获了由网络推断的参数、校准集、截断参数与分位数。参数被记录在校准表中，当表中的参数与校准器指定的参数不匹配时，校准表将被忽略。当网络或校准集发生变化，应该由应用程序指校准表无效。

当基于同样的校准集对同一个网络进行校准参数搜索时，直方图缓存是很有用的，因为它使得直方图的构建只被运行一次。同样，当网络或校准集发生变化，应该由应用程序指校准表无效。

缓存按照下面的方式使用：

- 如果校准表存在，则跳过校准过程，否则：
	- 如果直方图缓存存在，则跳过直方图构造过程，否则：
		- 构造Float32网络，并在校准集上运行网络，得到直方图
	- 依据直方图与参数构造校准表
- 依据校准表和网络定义构造INT8的网络

已经缓存的数据通过指针和长度参数传递，例如：
```c++
const void* readHistogramCache(size_t& length) override
{
    length = mHistogramCache.size();
    return length ? &mHistogramCache[0] : nullptr;
}

void writeHistogramCache(const void* cache, size_t length) override
{
    mHistogramCache.clear();
    std::copy_n(reinterpret_cast<const char*>(cache), length, std::back_inserter(mHistogramCache));
}
```


<span id = "choose_calibration_parameters">
###选择校准集参数
</span>
截断参数与分位数都是[0,1]间的数字，其具体含义在附带的white paper中讨论。为了找到最佳的校准集参数，我们可以基于额外图片得到参数组合和其对应的网络分数，并从中寻找校准集参数。`searchCalibrations()`展示了如何这样做。对ImageNet网络而言，5000张图片被用于进行最佳校准。因为校准过程会仅在截断参数与分位数不同的情况下运行多次，我们强烈建议使用直方图缓存。

## giexec：一个命令行包装器
在示例程序的文件夹中包含有一个TensorRT的命令行包装，它在基于任意数据对网络做benchmark，以及从这些模型生成序列化引擎很有用。命令行参数如下：
```bash
Mandatory params:
  --deploy=<file>      Caffe deploy file
  --output=<name>      Output blob name (can be specified multiple times)

Optional params:
  --model=<file>       Caffe model file (default = no model, random weights used)
  --batch=N            Set batch size (default = 1)
  --device=N           Set cuda device to N (default = 0)
  --iterations=N       Run N iterations (default = 10)
  --avgRuns=N          Set avgRuns to N - perf is measured as an average of avgRuns (default=10)
  --workspace=N        Set workspace size in megabytes (default = 16)
  --half2              Run in paired fp16 mode (default = false)
  --int8               Run in int8 mode (default = false)
  --verbose            Use verbose logging (default = false)
  --hostTime           Measure host time rather than GPU time (default = false)
  --engine=<file>      Generate a serialized GIE engine
  --calib=<file>       Read INT8 calibration cache file
```
例如：
```bash
giexec --deploy=mnist.prototxt --model=mnist.caffemodel --output=prob
```
如果没有提供“--model”，则权重将被随机生成

该样例没有展示任何前述未曾包含的TensorRT特性

## 在多GPU上使用TensorRT
每个`ICudaEngine`对象在通过builder或反序列化而实例化时均被builder限制于一个指定的GPU内。要进行GPU的选择，需要在进行反序列化或调用builder之前调用`cudaSetDeviec()`。每个`IExecutionContext`都被限制在产生它的引擎所在的GPU内，当调用`execute()`或`enqueue()`时，请在必要时调用`cudaSetDevice()`以保证线程与正确的设备绑定。

## 数据格式
TensorRT的输入输出张量均为以NCHW形式存储的32-bit张量。NCHW指张量的维度顺序为batch维（N）-通道维（C）-高度（H）-宽度（W）

对权重而言：

- 卷积核存储为KCRS形式，其中K轴为卷积核数目的维度，即卷积层输出通道维。C轴为是输入张量的通道维。R和S分别是卷积核的高和宽
- 全连接层按照行主序形式存储
- 反卷积层按照CKRS形式存储，各维含义同上

## FAQ

**Q：如何构造对若干不同可能的batch size优化了的引擎？**
A：尽管TensorRT允许在给定的一个batch size下优化模型，并在运行时送入任何小于该batch size的数据，但模型在更小size的数据上的性能可能没有被很好的优化。为了面对不同batch大小优化模型，你应该对每种batch size都运行一下builder和序列化。未来的TensorRT可能能基于单一引擎对多种batch size进行优化，并允许在当不同batch size下层使用相同的权重形式时，共享层的权重。

**Q：如何选择最佳的工作空间大小**:
A: 一些TensorRT算法需要GPU上额外的工作空间。方法`IBuilder::setMaxWorkspaceSize()`控制了能够分配的工作空间的最大值，并阻止builder考察那些需要更多空间的算法。在运行时，当创造一个`IExecutionContext`时，空间将被自动分配。分配的空间将不多于所需求的空间，即使在`IBuilder::setMaxWorspaceSize()`中设置的空间还有很多。应用程序因此应该允许TensorRT builder使用尽可能多的空间。在运行时，TensorRT分配的空间不会超过此数，通常而言要少得多。

  [1]: https://developer.nvidia.com/nvidia-tensorrt-20-download
