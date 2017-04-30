# TensorRT_Tutorial

TensorRT作为NVIDIA推出的c++库，能够实现高性能推理（inference）过程。最近，NVIDIA发布了TensorRT 2.0 Early Access版本，重大更改就是支持INT8类型。在当今DL大行其道的时代，INT8在缩小模型大小、加速运行速度方面具有非常大的优势。Google新发布的TPU就采用了8-bit的数据类型。

本人目前在使用TensorRT进行INT8的探究。已经被TensorRT不完善的文档坑了一次了。所以想自力更生做一个TensorRT Tutorial，主要包括三部分：
 - TensorRT User Guide 翻译；
 - TensorRT samples 介绍分析讲解；
 - TensorRT 使用经验。

使用TensorRT者请先阅读《[TensorRT目前存在的BUG][1]》。

 感谢每一位为该翻译项目做出贡献的同学.
 
 内容来源：
 TensorRT 下载页面：
 https://developer.nvidia.com/nvidia-tensorrt-20-download
 
 TensorRT 文档、Samples
 安装后对应目录中
 
## 参与者（按参与时间排序）
TensorRT User Guide 翻译
 - [LitLeo][2]
 - [MoyanZitto][3]

翻译校对

 - 赵开勇

TensorRT samples 介绍分析讲解
- [LitLeo][4]

TensorRT 使用经验。

## 进度记录
 - 2017-04-27，项目发起，创建GitHub仓库。

 
 
欲参与者请加QQ群：483063470


  [1]: https://github.com/LitLeo/TensorRT_Tutorial/blob/master/Bug.md
  [2]: https://github.com/LitLeo
  [3]: https://github.com/MoyanZitto
  [4]: https://github.com/LitLeo