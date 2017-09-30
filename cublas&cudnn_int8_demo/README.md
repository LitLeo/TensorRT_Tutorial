cublas使用cublasGemmEx函数的CUDA_R_32I计算模式来实现INT8加速。需要注意的坑是，alpha 和 beta 这两个参数必须为 int类型，cublas文档没有写明白。

cudnn 的卷积INT8加速为使用cudnnConvolutionForward的四种INT8配置（INT8, INT8_EXT, INT8x4, INT8x4_EXT），按自己需求决定使用哪个函数。[demo在这里][1]，他的这个代码有点小错误，cudnn cudnnConvolutionForward INT8输入要求是4的倍数，详细要求见cudnn文档，[问题讨论在这里][2]。


  [1]: https://github.com/jesryu/cudnn_conv_int8
  [2]: https://devtalk.nvidia.com/default/topic/1005119/cudnn-v6-int8-convolution-failing-with-cudnn_status_not_supported/