本文档主要针对API方式build网络方式。
语音领域现在仍处于成熟前的发展期，网络结构日新月异，模型大小逐渐变大，性能要求随之提高。因此我不太习惯用parser的形式去解析模型并构造网络。原因一是TRT目前支持功能有限，就算成熟框架（pytorch，tf）也容易遇到op不支持的情况，更何况还有kadi的存在；另一个是有时候性能要求高，全靠parser不好手动做优化。
TensorRT的文档目前可以说是已经很成熟了，但仍架不住是闭源软件，使用API 构造网络的时候，不太耐操，仍需小心翼翼。学会API的正确使用方法，事半功倍。学习API的方法，自然是要借鉴别人的成熟代码~~
下面列一些比较成熟的代码，供借鉴。
1. 首先肯定是TenorRT自身的sample和plugin
2. onnx parser的源码，主要是builtin_op_importers.cpp
3. pytorch 源码中TensorRT相关部分
4. TF 源码中TensorRT相关部分
5. GitHub TensorRT issue和NVIDIA 论坛中的问答。

踩过的一些路：
1. RNNv2 layer中，双向lstm要求seql_len是固定的，无法适用于dynamic shape. 可以用loop layer特性和其他计算接口“拼接”出来一个支持动态seqlen的lstm。详见 TensorRT sample sampleCharRNN.cpp
2. IFullyConnectedLayer 进行矩阵乘的时候，会把{C, H, W} reshape 成 {1, C*H*W}，输出为{K, 1, 1}，就比较蛋疼。如何在不同场景下使用不同的接口实现矩阵乘，进行性能和显存的优化，详见以上各种代码。
