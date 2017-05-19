# TensorRT 2.0 User Guide

---

[toc]

## 介绍
NVIDIA TensorRT是一个C++库，在NVIDIA GPU上能够实现高性能的推理（inference ）过程。TensorRT优化网络的方式有：对张量和层进行合并，转换权重，选择高效的中间数据格式，以及依据层的参数和实测性能，从一个丰富的核仓库中进行筛选。

编译TensorRT 2.0 要求GCC >= 4.8

TensorRT 2.0 现在支持以下layer类型：

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
### Ubuntu 下安装方式
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
  ```bash
  dpkg -l | grep tensorrt-2
  ```

 5. 若安装成功，你应看到:
  `tensorrt-2 2.0.0-1+cuda amd64 Meta package of TensorRT`
同样，通过命令：
  `dpkg -l | grep nvinfer2`

 4. 你应看到:
  `libnvinfer2 2.0.0-1+cuda amd64 TensorRT runtime libraries`

注意：TensorRT 2.0现在只提供了Ubuntu 14.04和16.04两个版本。

### Centos 7 下安装方式

TensorRT对Ubuntu系统友好，如果是企业级系统（比如centos）可以下载下来解压然后手动安装。
前提条件：建议Centos 7以上，即gcc 版本要大于4.8，因为TensorRT内使用了大量的c++ 11特性。如果你是大神可以在Centos 6上升级gcc 到4.8并把一些依赖问题搞定。
安装步骤如下：

 1. 下载deb安装包，然后解压，一路挑着大文件解压，找到两个头文件NvCaffeParser.h。NvInfer.h和对应的so文件，libnvcaffe_parser.so.2.0.0，libnvinfer.so.2.0.0。

 2.然后安装方式就跟cudnn一样了，*.h上传到CUDA_HOME/include下，lib文件上传到CUDA_HOME/lib64目录下（lib文件记得添加libnvinfer.so和libnvcaffe_parser.so的链接）

 3.安装完毕，如果要在Centos上跑samples，记得要修改一下Makefile

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
网络定义是由Layers和Tensors组成的。

每一层都一组输入tensor和一组输出tensor，根据层类型和输入tensor来计算输出tensor。不同类型的层具有不同的参数，比如卷积size和stride，以及卷积滤波器权值。

tensor是网络的输入或者输出。tensor的数据目前支持16bit和32bit浮点数和三维（通道，宽，高）。输入tensor的数据大小由程序猿指定，输出tensor的大小自动就算出来了。

每一层和tensor都有一个名字，在分析或者读构建日志时非常有用。

当使用caffe parser时，tensor和层的名字直接从caffe prototxt读取。

## SampleMNIST：简单使用方法

## SampleGoogleNet:性能分析与16-bit推断
### 性能分析
### half2模式

## SampleINT8：8-bit校准与推断

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
- 全连接层按照行主序形式存储  <font color="red">这里是错的！！全连接层中weights的存储方式是col-major，详见[Bugs](https://github.com/LitLeo/TensorRT_Tutorial/blob/master/Bug.md)</font>
- 反卷积层按照CKRS形式存储，各维含义同上

## FAQ
**Q：如何在TensorRT中使用自定义层？**
A：当前版本的TensorRT不支持自定义层。要想在TensorRT中使用自定义层，可以创建两个TensorRT工作流，自定义层夹在中间执行。比如：

``` c++
IExecutionContext *contextA = engineA->createExecutionContext();
IExecutionContext *contextB = engineB->createExecutionContext();

<...>

contextA.enqueue(batchSize, buffersA, stream, nullptr);
myLayer(outputFromA, inputToB, stream);
contextB.enqueue(batchSize, buffersB, stream, nullptr);
```

**Q：如何构造对若干不同可能的batch size优化了的引擎？**
A：尽管TensorRT允许在给定的一个batch size下优化模型，并在运行时送入任何小于该batch size的数据，但模型在更小size的数据上的性能可能没有被很好的优化。为了面对不同batch大小优化模型，你应该对每种batch size都运行一下builder和序列化。未来的TensorRT可能能基于单一引擎对多种batch size进行优化，并允许在当不同batch size下层使用相同的权重形式时，共享层的权重。

**Q：如何选择最佳的工作空间大小**:
A: 一些TensorRT算法需要GPU上额外的工作空间。方法`IBuilder::setMaxWorkspaceSize()`控制了能够分配的工作空间的最大值，并阻止builder考察那些需要更多空间的算法。在运行时，当创造一个`IExecutionContext`时，空间将被自动分配。分配的空间将不多于所需求的空间，即使在`IBuilder::setMaxWorspaceSize()`中设置的空间还有很多。应用程序因此应该允许TensorRT builder使用尽可能多的空间。在运行时，TensorRT分配的空间不会超过此数，通常而言要少得多。

  [1]: https://developer.nvidia.com/nvidia-tensorrt-20-download
