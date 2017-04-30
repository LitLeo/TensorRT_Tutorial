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

#### 校准集参数

### 配置Builder

### 校准集缓存

<span id = "caching">
###校准集缓存
</span>

<span id = "choose_calibration_parameters">
###选择校准集参数
</span>

## giexec：一个命令行包装器
## 在多GPU上使用TensorRT
## 数据格式
## FAQ

  [1]: https://developer.nvidia.com/nvidia-tensorrt-20-download
