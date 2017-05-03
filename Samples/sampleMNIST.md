
## SampleMNIST：简单使用方法
SampleMNIST 使用训练好的MNIST caffe模型来演示典型的构建和执行过程。
构建阶段，直接调用ICaffeParser接口的parse()函数读取caffe model。

### 日志
构建网络之前需要先重载实现log类，可以用来报告error、warning和informational 信息。

### Build过程：caffeToGIEModel
### 引擎反序列化（引擎重构）
### Execution过程：doInference()