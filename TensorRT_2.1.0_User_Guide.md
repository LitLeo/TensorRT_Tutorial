# TensorRT 2.1.0 User Guide

---

[toc]

## Introduction
NVIDIA TensorRT是一个C++库，在NVIDIA GPU上能够实现高性能的推理（inference ）过程。TensorRT优化网络的方式有：（1）合并Tensor和layer，转换weight，选择有效的中间数据格式 and selecting from a large kernel catalog based on layer parameters and measured performance.

编译TensorRT 2.1.0 要求GCC >= 4.8

TensorRT 2.1.0 现在支持一下layer类型：

 - **Convolution**：可无bias。目前仅支持2D卷积（即4D的输入和输出Tensor）。**Note：**该卷积层的操作实际上一个<font color="red">correlation</font>，如果你想通过TensorRT的API而不是通过caffe parser library导入权重，这是一个需要考虑的因素。
 - **Activation**: ReLU, tanh and sigmoid.
 - **Pooling**: max 和 average.
 - **Scale**: 可以根据定常量（constant value）对每一个tensor, channel 或weight进行仿射变换和取幂操作。
 - **ElementWise**: 两个tensor的sum, product 或 max操作.
 - **LRN**: <font color="red">cross-channel only.</font>
 - **Fully-connected**：可无bias
 - **SoftMax**: cross-channel only
 - **Deconvolution**： 可无bias
 - **RNN**： 包括 GRU 和 LSTM

(**note**: Batch Normalization 操作可以用Scale层实现.)

TensorRT是一个独立的深度学习部署框架，对caffe尤其友好。TensorRT提供了一个针对caffe的模型解析器-NvCaffeParser，可以通过几行代码解析caffe生成的model并定义网络。Caffe中的Convolution, ReLU, Sigmoid, TanH, Pooling, Power, BatchNorm, Eltwise, LRN, InnerProduct, SoftMax, Scale, and Deconvolution层类型可以直接被NvCaffeParser直接解析。而caffe中下列层类型是不被NvCaffeParser支持的：
Deconvolution groups
Dilated convolutions
PReLU
Leaky ReLU
Scale, other than per-channel scaling
EltWise with more than two inputs
**Note：** TensorRT不支持caffe的旧prototxt格式的。

## Quick Start Instructions
PS：这部分是从[TensorRT下载页面][1]翻译过来的。
TensorRT原名GIE。GIE又名TensorRT 1.0，TensorRT 2.0正式改名。
TensorRT 2.0非常大的改动点是支持INT8类型（TensorRT 1.0支持FP16）。
使用TensorRT 2.0的硬件要求：Tesla P4, Tesla P40, GeForce TitanX Pascal, GeForce GTX 1080, DRIVE PX 2 dGPU
软件要求：CUDA 8.0
安装命令：

 1. Verify that you have the CUDA toolkit installed, release 8.0 .
 2. Download the TensorRT debian package (below)
 3. Install TensorRT from the debian package:
 `sudo dpkg -i nv-tensorrt-repo-ubuntu1404-7-ea-cuda8.0_2.0.1-1_amd64.deb
    sudo apt-get update
    sudo apt-get install tensorrt-2`

 4. Verify your installation:
  `dpkg -l | grep tensorrt-2`

 4. and you should see:
  `tensorrt-2 2.0.0-1+cuda amd64 Meta package of TensorRT
  dpkg -l | grep nvinfer2`

 4. and you should see:
  `libnvinfer2 2.0.0-1+cuda amd64 TensorRT runtime libraries`

注意：TensorRT 2.0现在只提供了Ubuntu 14.04和16.04两个版本。对Ubuntu系统友好，如果是企业级系统（比如centos）可以下载下来解压然后手动安装。
再注意：我Linux学的不好，折腾了一遍，Centos6.x安不上（gcc和libstdc++.so版本问题），建议Centos 7以上

## Getting Started
使用TensorRT包括两部步骤（（1）打开冰箱；（2）把大象装进去）：
 - build阶段，主要内容为进行网络定义和生成推理引擎，TensorRT会根据你的网络定义进行优化。
 - execution阶段，需要将input和output在GPU上开辟空间并将input传输到GPU上，调用推理接口得到output结果，再将结果拷贝回host端。

build阶段比较耗时间，特别是在嵌入式平台上。所以典型的使用方式就是将build后的引擎序列化（序列化后可写到硬盘里）供以后使用。

build阶段对网络进行了以下优化：

 - 去掉没有被使用过的输出层
 - 将convolution、bias和ReLU操作融合到一起
 - 将相似度比较高的参数和相同的Tensor进行聚合（例如，GoogleNet v5的初始模块中的1*1卷积）
 - <font color="red">elision of concatenation layers by directing layer outputs to the correct eventual destination.</font>

<font color="red">In addition it runs layers on dummy data to select the fastest from its kernel catalog, and performs weight preformatting and memory optimization where appropriate.</font>

## The Network Definition

  [1]: https://developer.nvidia.com/nvidia-tensorrt-20-download
