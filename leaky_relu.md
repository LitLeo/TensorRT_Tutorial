# 使用TensorRT实现leaky relu层

------

最近在使用TensorRT加速CNN的时候，发现TensorRT支持relu但不支持leaky relu，这里分享出网上找到的曲线求国的[解决方案][1]和实现。
## 解决方案
解决方案比较简单，使用scale层、relu层和ElementWise层实现leaky relu，具体流程如下：
![flow_of_leaky_relu][2]

## 实现方式
实现方式有两种。

 1. 在训练时就这样定义，然后直接把训练好的模型丢给TensorRT。比如使用caffe，在prototxt文件中这样定义leaky relu层（[详见这里][3]），然后再使用TensorRT中的NvCaffeParser转化就行了。
 2. 自己用API实现，代码如下：
``` c++
void LeakyRelu(INetworkDefinition *network, ITensor *it)
{
    Weights power{DataType::kFLOAT, nullptr, 0};
    Weights shift{DataType::kFLOAT, nullptr, 0};

    float *scale_params = new float[2];

    // scale_1 * 0.1
    scale_params[0] = 0.1f;

    Weights scale{DataType::kFLOAT, &scale_params[0], 1};
    auto scale_1 = network->addScale(*it, ScaleMode::kUNIFORM, shift, scale, power);
    assert(scale_1 != nullptr);

    // relu + scale_2 * 0.9;
    auto relu = network->addActivation(*it, ActivationType::kRELU);
    assert(relu != nullptr);

    scale_params[1] = 0.9f;
    Weights scale1{DataType::kFLOAT, &scale_params[1], 1};
    auto scale_2 = network->addScale(*relu->getOutput(0), ScaleMode::kUNIFORM, shift, scale1, power);
    assert(scale_2 != nullptr);

    // result = scale_1 + scale_2
    auto ew = network->addElementWise(*scale_1->getOutput(0), *scale_2->getOutput(0), ElementWiseOperation::kSUM);
    assert(ew != nullptr);
}
```
## 坑
其实这个实现是比较简单的，没太有必要写出来。但是TensorRT给的scale层demo里有个坑，分享给大家。
sampleMNISTAPI里给出的scale层代码如下：
```c++
  // Create a scale layer with default power/shift and specified scale parameter.
  float scale_param = 0.0125f;
  Weights power{DataType::kFLOAT, nullptr, 0};
  Weights shift{DataType::kFLOAT, nullptr, 0};
  Weights scale{DataType::kFLOAT, &scale_param, 1};
  auto scale_1 = network->addScale(*data, ScaleMode::kUNIFORM, shift, scale, power);
  assert(scale_1 != nullptr);
```
第5行是scale的参数，根据参数个数可以实现对于per-tensor, per-channel, or per-element进行scale操作，所以其中scale的第二个参数是个指针，指向一个数组。

这个坑在于第2行和第5行。因为是要对整个Tensor进行scale操作，所以Weights scale内参数个数为1，因此就声明了一个局部变量scale_param。问题就在这，scale_param是局部变量，在栈空间，函数结束就会被释放掉！所以在实现LeakyRelu()函数的时候，第六行的scale_params一定不能用局部变量存储，我选择放到堆空间中。

从这个例子中深深反思自己的c++能力，以后在用现成代码的时候一定要仔细检查，看是否符合目前的需求。


  [1]: https://github.com/TLESORT/YOLO-TensorRT-GIE-
  [2]: https://raw.githubusercontent.com/LitLeo/TensorRT_Tutorial/master/img/flow_of_leaky_relu.png
  [3]: https://devtalk.nvidia.com/default/topic/990426/jetson-tx1/tensorrt-yolo-inference-error/post/5087820/