# Bug
本页面为TensorRT存在的bug or error。

 - 版本：TensorRT 2.0
 - BUG类别：文档错误
 - 贡献者：[LitLeo][1]
 - BUG描述：
TensorRT中Weights的存储方式是col-major的，在文档中却写的是row-major。  <br />
在文档Data Formats章节，原文为  <br />
`Fully Connected weights are in contiguous row-major layout`  <br />
但是在Samples/samplePlugin的enqueue函数(该函数实现了一个全连接)中，weights的存储方式是col-major的，代码如下  <br />
`
CHECK(cublasSgemm(mCublas, CUBLAS_OP_T, CUBLAS_OP_N, nbOutputChannels, batchSize, nbInputChannels, &kONE, 
                reinterpret_cast<const float*>(mKernelWeights.values), nbInputChannels, 
                reinterpret_cast<const float*>(inputs[0]), nbInputChannels, &kZERO, 
                reinterpret_cast<float*>(outputs[0]), nbOutputChannels));
`


  [1]: https://github.com/LitLeo
