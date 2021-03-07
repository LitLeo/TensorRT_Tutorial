# 建议看最新视频版本！列表如下
 - [《TensorRT Tutorial（一）如何选择TensorRT版本》][21]
 - [《TensorRT Tutorial（二）编译 TensorRT 的开源源码》][22]
 - [《TensorRT Tutorial（3.1）讲解 TensorRT 文档-基本使用》][23]
 - [《TensorRT Tutorial（3.2）讲解 TensorRT 文档-TRT可借鉴的代码样例》][24]
 - [《TensorRT Tutorial（3.3.1）plugin 例子和原理》][25]
 - [《TensorRT Tutorial（3.3.2）如何打造自己的plugin库》][26]
 - 视频版资料见目录-视频版资料

## 进度记录
 - 2017-04-27 项目发起，创建GitHub仓库。
 - 2017-09-30 TensorRT 3最近发布，整理一下目前的资源。
 - 2017-10-18 增加博客-使用TensorRT实现leaky relu层
 - 2017-11-11 资源：新增google的INT8开源库
 - 2017-11-25 增加博客-TensorRT Plugin使用方式简介-以leaky relu层为例
 - 2020-8-31 增加博客《TensorRT Github 开源部分介绍》
 - 2020-9-7 增加博客《TensorRT 可借鉴代码汇总》

----

## 资源整理
 - [TensorRT 3 RC][1]和[TensorRT 2.1][2] 下载链接
 - [TensorRT 2.1 官方在线文档][3] 
 - NVIDIA 介绍TensorRT的blog-[Deploying Deep Neural Networks with NVIDIA TensorRT][4]
 - GTC 2017介绍TensorRT 的[PPT][5]和[视频][6]，内含INT8 Quantization和Calibration的实现原理。
 - 新增cublas 和 cudnn的INT8 [demo][7]
 - 新增本人在GTC China 2017 Community Corner主题NVIDIA INT8的PPT， [GTC-China-2017-NVIDIA-INT8.pdf][8]
 - 新增google的INT8开源库[gemmlowp][9]，目前支持ARM和CPU优化
 - “子棐之GPGPU”公众号所写的《TensorRT系列》博客，NVIDIA的工程师出的，从入门篇到INT8篇再到FP16篇最后收尾于Custom Layer篇，内容逻辑清楚，干货满满，自愧不如。附四篇博客链接：[TensorRT 系列之入门篇][10]，[TensorRT系列之INT8篇][11]，[TensorRT系列之FP16篇][12]，[TensorRT系列之Custom Layer篇][13]。
 - [《高性能深度学习支持引擎实战——TensorRT》][14]，主要内容：一、TensorRT理论介绍：基础介绍TensorRT是什么；做了哪些优化；为什么在有了框架的基础上还需要TensorRT的优化引擎。二、TensorRT高阶介绍：对于进阶的用户，出现TensorRT不支持的网络层该如何处理；

---
## 博客
 - [使用TensorRT实现leaky relu层][15]
 - [TensorRT Plugin使用方式简介-以leaky relu层为例][16]

# TensorRT_Tutorial

TensorRT作为NVIDIA推出的c++库，能够实现高性能推理（inference）过程。最近，NVIDIA发布了TensorRT 2.0 Early Access版本，重大更改就是支持INT8类型。在当今DL大行其道的时代，INT8在缩小模型大小、加速运行速度方面具有非常大的优势。Google新发布的TPU就采用了8-bit的数据类型。

本人目前在使用TensorRT进行INT8的探究。已经被TensorRT不完善的文档坑了一次了。所以想自力更生做一个TensorRT Tutorial，主要包括三部分：
 - TensorRT User Guide 翻译；
 - TensorRT samples 介绍分析讲解；
 - TensorRT 使用经验。

 感谢每一位为该翻译项目做出贡献的同学.
 
 内容来源：
 TensorRT 下载页面：
 https://developer.nvidia.com/nvidia-tensorrt-20-download
 
 TensorRT 文档、Samples
 安装后对应目录中
 
## 参与者（按参与时间排序）
TensorRT User Guide 翻译
 - [LitLeo][18]
 - [MoyanZitto][19]

翻译校对

 - 赵开勇

TensorRT samples 介绍分析讲解
- [LitLeo][20]

TensorRT 使用经验。

欲参与者请加QQ群：483063470

支持捐赠项目

 <img src="https://raw.githubusercontent.com/LitLeo/blog_pics/master/WeChat_collection.png" width = "200px" height = "200"/>

## 招实习生
【实习】【腾讯北京AILAB】招募AI异构加速实习生  
简历直接给负责人，给简历保证迅速反馈。  
基本条件: 熟悉c++，至少实习6个月  
工作内容：
1. 使用c++复现框架训练的模型并进行CPU、GPU、ARM加速，达到上线的性能要求。
2. 调研各种inference框架并投入生产
加分项：
1. 写过或者维护过深度学习框架代码； 
2. 会CUDA 开发，能自己写kernel，会用cublas，cudnn等库； 
3. linux cpu c++编程能力，会写avx、会用mkl；
4. 熟悉深度学习计算过程
5. 学习能力强，实习时间长
联系方式: leowgyang@tencent.com

  [1]: https://developer.nvidia.com/nvidia-tensorrt3rc-download
  [2]: https://developer.nvidia.com/nvidia-tensorrt-download
  [3]: http://docs.nvidia.com/deeplearning/sdk/tensorrt-user-guide/index.html
  [4]: https://devblogs.nvidia.com/parallelforall/deploying-deep-learning-nvidia-tensorrt/
  [5]: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
  [6]: http://on-demand.gputechconf.com/gtc/2017/video/s7310-szymon-migacz-8-bit-inference-with-tensorrt.mp4
  [7]: https://github.com/LitLeo/TensorRT_Tutorial/tree/master/cublas&cudnn_int8_demo
  [8]: https://github.com/LitLeo/TensorRT_Tutorial/blob/master/GTC-China-2017-NVIDIA-INT8.pdf
  [9]: https://github.com/google/gemmlowp
  [10]: https://mp.weixin.qq.com/s/E5qbMsuc7UBnNmYBzq__5Q
  [11]: https://mp.weixin.qq.com/s/wyqxUlXxgA9Eaxf0AlAVzg
  [12]: https://mp.weixin.qq.com/s/nuEVZlS6JfqRQo30S0W-Ww?scene=25#wechat_redirect
  [13]: https://mp.weixin.qq.com/s/xabDoauJc16z3-gpyre8zA
  [14]: https://mp.weixin.qq.com/s/F_VvLTWfg-COZKrQAtOSwg
  [15]: https://github.com/LitLeo/TensorRT_Tutorial/blob/master/blogs/%E4%BD%BF%E7%94%A8TensorRT%E5%AE%9E%E7%8E%B0leaky%20relu%E5%B1%82.md
  [16]: https://github.com/LitLeo/TensorRT_Tutorial/blob/master/blogs/TensorRT%20Plugin%E4%BD%BF%E7%94%A8%E6%96%B9%E5%BC%8F%E7%AE%80%E4%BB%8B-%E4%BB%A5leaky%20relu%E5%B1%82%E4%B8%BA%E4%BE%8B.md
  [17]: https://github.com/LitLeo/TensorRT_Tutorial/blob/master/Bug.md
  [18]: https://github.com/LitLeo
  [19]: https://github.com/MoyanZitto
  [20]: https://github.com/LitLeo
  [21]: https://www.bilibili.com/video/BV1Nf4y1v7sa/
  [22]: https://www.bilibili.com/video/BV1x5411n76K/
  [23]: https://www.bilibili.com/video/BV19V411t7LV/
  [24]: https://www.bilibili.com/video/BV1DT4y1A7Rx/
  [25]: https://www.bilibili.com/video/BV1op4y1p7bj/
  [26]: https://www.bilibili.com/video/BV1Qi4y1N7YS/

