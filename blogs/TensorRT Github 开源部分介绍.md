此文档基于TensorRT7.0。

TensorRT（下面简称“TRT”）目前由两个部分组成，闭源部分和开源部分。闭源部分就是官方提供的库，是TRT的核心部分；开源部分在github上，包含Parser（caffe，onnx）、Sample和一些plugin。

其中开源部分中，有一部分要值得提一下，就是bert相关的plugin，位于demo/bert目录下。TRT在5.0版本中，提供了一个bert的demo，包括了一套完整的代码，实现将bert tf模型使用TRT上线，内容包括提取tf模型权值的python脚本，bert模型所需要的plugin，以及读取权值、搭建bert网络并运行和测试的代码，对于tf bert模型的使用者来说，可以说是很方便了。在6.0和7.0中，demo/bert目录被删除，其中的plugin修改为dynamic input shape版本，并放到plugin中，而其余代码则在7.1版本中才再次提供。这部分会在另一篇博客中（如果我能更下去的话==）做更详细的解释，这里就不赘述了。

## 简单介绍
TRT作为NV inference 但大杀器，出世以来虽然速度很快（真香），但一直被“闭源”、“难用”所诟病。TRT github 虽然只是TRT开源出来的一小部分非核心代码，但也是有很多干货的。下面对各目录做一个简单的介绍。

parser目录，主要包括官方维护的两个parser。caffe和onnx parser，能够将caffe格式模型和onnx格式模型直接转成TRT格式。其中caffe parser随着caffe逐渐退出训练框架舞台已经慢慢不维护了。

plugin目录，包含官方提供的一些plugin，大部分plugin都是跟计算机视觉网络相关的，6.0开始加入了bert网络的plugin。

sample目录，就是怕你不会用，提供的一些demo。还提供了一个trtexec工具，我没用过，就不多说了。

## 无网情况下编译

NV虽然只开源了部分非核心代码，但还是很有用的。有用就要用上，第一步当然是编译。但是部分情况下，服务器可能是没有网的（懂的人自然懂）。而trt git在手动源码编译的时候，竟然需要联网下载protoubf……这里针对这个问题简单说一下。

TRT的cmake编译选项中，有parser、plugin、sample等多个选项开关。其中部分sample编译依赖parser（caffe和onnx）模块，而parser模块依赖protobuf，而这个protobuf是根据你指定的版本先联网下载再编译的。这个设计对于无网的服务器，实在是不友好……

因为这个小麻烦而大改CMakeLists.txt实在是有点不值当。下面简单介绍一下比较简单的解决方案。

1. 不编译依赖parser的那部分sample，直接在TensorRT/samples/opensource/CMakeLists.txt中删掉即可。
2. 替换protobuf的下载链接，在另一台机器或者本机上搭建一个apache，把相应版本的protobuf放上去即可。具体修改 TensorRT/third_party/protobuf.cmake line22
3. 直接把protobuf放到编译目录下，然后修改TensorRT/third_party/protobuf.cmake 的部分代码，不从网络下载，而直接选择本地文件并编译即可。（这个cmake大佬可以试试，我没试过，感觉还挺麻烦的zz）

