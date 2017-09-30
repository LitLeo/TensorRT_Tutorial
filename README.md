
## 进度记录
 - 2017-04-27 项目发起，创建GitHub仓库。
 - 2017-09-30 TensorRT 3发布，整理一下目前的资源。

----
## 资源整理
 - [TensorRT 3 RC][1]和[TensorRT 2.1][2] 下载链接
 - [TensorRT 2.1 官方在线文档][3] 
 - NVIDIA 介绍TensorRT的blog-[Deploying Deep Neural Networks with NVIDIA TensorRT][4]
 - GTC 2017介绍TensorRT 的[PPT][5]，内含INT8 Quantization和Calibration的实现原理。
 - 新增cublas 和 cudnn的INT8 demo，见cublas&cudnn_int8_demo文件夹
 - 新增本人在GTC China 2017 Community Corner主题NVIDIA INT8的PPT，见《GTC-China-2017-NVIDIA-INT8.pdf》

# TensorRT_Tutorial

TensorRT作为NVIDIA推出的c++库，能够实现高性能推理（inference）过程。最近，NVIDIA发布了TensorRT 2.0 Early Access版本，重大更改就是支持INT8类型。在当今DL大行其道的时代，INT8在缩小模型大小、加速运行速度方面具有非常大的优势。Google新发布的TPU就采用了8-bit的数据类型。

本人目前在使用TensorRT进行INT8的探究。已经被TensorRT不完善的文档坑了一次了。所以想自力更生做一个TensorRT Tutorial，主要包括三部分：
 - TensorRT User Guide 翻译；
 - TensorRT samples 介绍分析讲解；
 - TensorRT 使用经验。

使用TensorRT者请先阅读《[TensorRT目前存在的BUG][6]》。

 感谢每一位为该翻译项目做出贡献的同学.
 
 内容来源：
 TensorRT 下载页面：
 https://developer.nvidia.com/nvidia-tensorrt-20-download
 
 TensorRT 文档、Samples
 安装后对应目录中
 
## 参与者（按参与时间排序）
TensorRT User Guide 翻译
 - [LitLeo][7]
 - [MoyanZitto][8]

翻译校对

 - 赵开勇

TensorRT samples 介绍分析讲解
- [LitLeo][9]

TensorRT 使用经验。

欲参与者请加QQ群：483063470

支持捐赠项目

 <img src="https://raw.githubusercontent.com/LitLeo/blog_pics/master/WeChat_collection.png" width = "200px" height = "200"/>


  [1]: https://developer.nvidia.com/nvidia-tensorrt3rc-download
  [2]: https://developer.nvidia.com/nvidia-tensorrt-download
  [3]: http://docs.nvidia.com/deeplearning/sdk/tensorrt-user-guide/index.html
  [4]: https://devblogs.nvidia.com/parallelforall/deploying-deep-learning-nvidia-tensorrt/
  [5]: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
  [6]: https://github.com/LitLeo/TensorRT_Tutorial/blob/master/Bug.md
  [7]: https://github.com/LitLeo
  [8]: https://github.com/MoyanZitto
  [9]: https://github.com/LitLeo