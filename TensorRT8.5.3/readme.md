# 开发者指南 :: NVIDIA深度学习TensorRT文档

## [修订历史](#revision-history)

这是《NVIDIA TensorRT 8.5 开发者指南》的修订历史。
## 第三章 更新

| 日期 | 变更摘要 |
| --- | --- |
| 2022年8月25日 | 从 [构建引擎](#build_engine_c "下一步是创建一个构建配置，指定TensorRT应如何优化模型。") 节中添加到新的 _优化构建性能_ 节的链接。|
| 2022年8月26日 | 为C++ API重写了 [执行推理](#perform-inference "引擎保存了优化后的模型，但要执行推理，我们必须管理中间激活状态的其他状态。这是通过使用ExecutionContext接口完成的：")。|

## 第4章 更新

| 日期 | 变更摘要 |
| --- | --- |
| 2022年8月25日 | 从[构建阶段](#importing_trt_python "要创建一个构建器，您必须首先创建一个记录器。Python绑定包括一个简单的记录器实现，它将所有严重性前的消息记录到stdout。")部分添加到新的“优化构建器性能”部分的链接。 |
| 2022年8月26日 | 为Python API重写[执行推理](#perform_inference_python "引擎保存了优化的模型，但要执行推理需要额外的中间激活状态。这是使用IExecutionContext接口完成的:")。 |

## 第5章 更新

| 日期 | 变更摘要 |
| --- | --- |
| 2022年8月25日 | *   解释了如何在[运行时阶段](#memory-runtime-phase "在运行时，TensorRT使用相对较少的主机内存，但可以使用大量的设备内存。")中使用PreviewFeature标志。<br>*   添加了[延迟模块加载](#lazy-module-loading)部分。 |
| 2022年8月30日 | 添加了[L2持久缓存管理](#persistent-cache-management "NVIDIA Ampere及更高架构支持L2缓存持久性，这是一项功能，允许优先保留L2缓存行，当选择要驱逐的行时。TensorRT可以利用此功能在缓存中保留激活，从而减少DRAM流量和功耗。")部分。 |
| 2022年9月19日 | 添加了[IFillLayer确定性](#ifilllayer-determinism "当使用RANDOM_UNIFORM或RANDOM_NORMAL操作将IFillLayer添加到网络时，上述确定性保证将不再有效。在每次调用时，这些操作基于RNG状态生成张量，然后更新RNG状态。此状态存储在每个执行上下文的基础上。")部分。 |

## 第6章 更新

| 日期 | 变更摘要 |
| --- | --- |
| 2022年8月25日 | 更新了C++和Python的[稀疏性](#structured-sparsity)示例代码。 |
| 2022年8月26日 | 重写了[重用输入缓冲区](#reusing-input-buffers "TensorRT允许指定一个CUDA事件，在输入缓冲区可以被重用时发出信号。这允许应用程序立即开始填充下一个推理的输入缓冲区区域，与完成当前推理并行。例如：")。 |
| 2022年8月30日 | 添加了[预览功能](#preview-feature "预览功能API是IBuilderConfig的扩展，允许逐步引入TensorRT的新功能。选择的新功能在此API下公开，允许您选择加入。预览功能在一两个TensorRT发布周期内保持预览状态，然后要么作为主流功能整合，要么被删除。当预览功能完全集成到TensorRT中时，它将不再通过预览API进行控制。")部分。 |

## 第8章 更新内容

| 日期 | 变更摘要 |
| --- | --- |
| 2022年8月25日 | 添加了[命名维度](#named-dimensions "常量和运行时维度都可以被命名。命名维度提供了两个好处：")部分。 |
| 2022年8月26日 | *   添加了一个名为[动态形状输出](#dynamic-shaped-output "如果网络的输出具有动态形状，则可以使用几种策略来分配输出内存。")和[查找多个优化配置文件的绑定索引](#binding-indices-opt-profiles "如果使用enqueueV3而不是已弃用的enqueueV2，则可以跳过此部分，因为基于名称的方法（如IExecutionContext::setTensorAddress）不需要配置文件后缀。")的新部分。<br>*   重写了[执行张量与形状张量](#exe_shape_tensors "TensorRT 8.5在执行张量和形状张量之间基本上消除了区别。但是，如果设计网络或分析性能，了解内部工作原理以及内部同步发生的位置可能会有所帮助。")。 |

## 第12章 更新

| 日期 | 变更摘要 |
| --- | --- |
| 2022年8月25日 | 在 [层支持和限制](#dla-lay-supp-rest "以下列表提供了在DLA上运行时对指定层的支持和限制:") 部分添加了Shuffle层和Equal运算符。 |
| 2022年9月30日 | *   在 [自定义DLA内存池](#customize-dla-mem-pools "您可以使用IBuilderConfig::setMemoryPoolLimit C++ API或IBuilderConfig.set_memory_pool_limit Python API自定义为网络中的每个DLA子网络分配的内存池的大小。有三种类型的DLA内存池（有关详细信息，请参阅MemoryPoolType枚举）:") 部分添加了关于NVIDIA Orin的附加信息。<br>*   在 [确定DLA内存池使用情况](#determine-dla-memory-pool-usage "成功从给定网络编译可加载项后，生成器将报告成功编译为可加载项的子网络候选集的数量，以及这些可加载项每个池使用的总内存量。对于由于内存不足而失败的每个子网络候选项，将发出一条消息以指出哪个内存池不足。在详细日志中，生成器还报告了每个可加载项的内存池要求。") 部分添加了。 |
| 2022年10月5日 | *   添加了 [使用C++构建DLA可加载项](#building-safety-nvmedia-dla-engine) 部分。<br>*   添加了 [使用trtexec生成DLA可加载项](#using-trtexec-gen-dla-load "trtexec工具可以生成DLA可加载项而不是TensorRT引擎。指定--useDLACore和--safe参数将生成器能力设置为EngineCapability::kDLA_STANDALONE。此外，指定--inputIOFormats和--outputIOFormats限制了I/O数据类型和内存布局。通过指定--saveEngine参数，DLA可加载项将保存到文件中。") 部分。 |

## 第13章 更新

| 日期 | 变更摘要 |
| --- | --- |
| 2022年8月25日 | 添加了[优化构建器性能](#opt-builder-perf "对于每个层，TensorRT构建器会对所有可用的策略进行分析，以寻找最快的推理引擎方案。如果模型具有大量层或复杂的拓扑结构，构建器的时间可能会很长。以下部分提供了减少构建器时间的选项。")部分。 |
| 2022年10月20日 | 在[CUDA性能分析工具](#nvprof)部分中添加了如何理解Nsight Systems时间线视图的说明。 |

## 附录更新

| 日期 | 变更摘要 |
| --- | --- |
| 2022年8月25日 | 在[常用命令行标志](#trtexec-flags "本部分列出了常用的trtexec命令行标志。")中添加了\--heuristic标志，用于策略启发。 |
| 2022年9月17日 | 移除了"Layers"章节，并创建了一个新的[_TensorRT运算符参考_](https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/index.html)。 |

## [摘要](#摘要)

本《NVIDIA TensorRT 开发者指南》演示了如何使用 C++ 和 Python API 来实现最常见的深度学习层。它展示了如何使用提供的解析器，采用深度学习框架构建的现有模型来构建 TensorRT 引擎。开发者指南还提供了逐步说明常见用户任务的步骤，例如创建 TensorRT 网络定义、调用 TensorRT 构建器、序列化和反序列化，以及如何使用 C++ 或 Python API 提供数据并执行推理。

有关以前发布的 TensorRT 开发者文档，请参阅[TensorRT 存档](https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html)。

## [1. 介绍](#概述)

NVIDIA® TensorRT™ 是一个能够实现高性能机器学习推理的 SDK。它旨在与 TensorFlow、PyTorch 和 MXNet 等训练框架相辅相成。它专注于在 NVIDIA 硬件上快速高效地运行已经训练好的网络。

请参考 _[NVIDIA TensorRT 安装指南](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html)_ 了解如何安装 TensorRT 的说明。

[_NVIDIA TensorRT 快速入门指南_](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html) 面向希望尝试 TensorRT SDK 的用户；具体来说，您将学习如何快速构建一个应用程序来在 TensorRT 引擎上运行推理。### [1.1. 本指南的结构](#structure)

第1章提供了有关TensorRT的打包和支持信息，以及它如何适应开发者生态系统。

第2章概述了TensorRT的广泛功能。

第三章和第四章分别介绍了C++和Python的API。

后续章节提供了关于高级功能的更多细节。

附录包含了层参考和常见问题的答案。

### [1.2. 示例](#samples)

[_NVIDIA TensorRT 示例支持指南_](https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html)展示了本指南中讨论的许多主题。更多关注嵌入式应用的示例可以在[这里](https://github.com/dusty-nv/jetson-inference)找到。

### [1.3. 互补的GPU功能](#gpu-features)

[多实例GPU](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html)，简称MIG，是NVIDIA Ampere架构或更新架构的NVIDIA GPU的一项功能，它可以将一个单一的GPU划分为多个较小的GPU。这些物理分区提供了具有QoS的专用计算和内存切片，并可以在GPU的一部分上独立执行并行工作负载。对于GPU利用率较低的TensorRT应用程序，MIG可以在对延迟几乎没有影响的情况下提供更高的吞吐量。最佳的划分方案因应用程序而异。

### [1.4. 辅助软件](#comp-software)

[NVIDIA Triton™](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/index.html) 推理服务器是一个更高级别的库，可在 CPU 和 GPU 上提供优化的推理。它提供了启动和管理多个模型的功能，并为提供推理的 REST 和 gRPC 端点。

[NVIDIA DALI®](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/#nvidia-dali-documentation) 提供了用于预处理图像、音频和视频数据的高性能原语。TensorRT 推理可以作为 DALI 流水线中的自定义运算符集成。可以在[此处](https://github.com/NVIDIA/DL4AGX)找到将 TensorRT 推理集成为 DALI 一部分的工作示例。

[TensorFlow-TensorRT (TF-TRT)](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html) 是将 TensorRT 直接集成到 TensorFlow 中的工具。它选择 TensorFlow 图的子图以由 TensorRT 加速，而将其余部分以 TensorFlow 的本机方式执行。结果仍然是一个可以像往常一样执行的 TensorFlow 图。有关 TF-TRT 示例，请参阅[TensorFlow 中的 TensorRT 示例](https://github.com/tensorflow/tensorrt)。

[Torch-TensorRT (Torch-TRT)](https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/) 是一个将 PyTorch 模块转换为 TensorRT 引擎的 PyTorch-TensorRT 编译器。在内部，PyTorch 模块首先根据所选的中间表示（IR）转换为 TorchScript/FX 模块。编译器选择 PyTorch 图的子图以由 TensorRT 加速，而将其余部分以 Torch 的本机方式执行。结果仍然是一个可以像往常一样执行的 PyTorch 模块。有关示例，请参阅[Torch-TRT 示例](https://github.com/pytorch/TensorRT/tree/master/notebooks)。[TensorFlow-Quantization toolkit](https://github.com/NVIDIA/TensorRT/tree/main/tools/tensorflow-quantization) 提供了在降低精度的情况下训练和部署基于Tensorflow 2的Keras模型的工具。该工具包根据操作符名称、类和模式匹配来量化图中的不同层。然后可以将量化图转换为ONNX格式，再转换为TensorRT引擎。有关示例，请参考[模型库](https://github.com/NVIDIA/TensorRT/tree/main/tools/tensorflow-quantization/examples)。

[PyTorch Quantization Toolkit](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html) 提供了在降低精度的情况下训练PyTorch模型的功能，然后可以导出为TensorRT进行优化。

此外，[PyTorch Automatic SParsity (ASP)](https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity) 工具提供了用于训练具有结构化稀疏性的模型的功能，然后可以导出并允许TensorRT在NVIDIA Ampere架构的GPU上使用更快的稀疏策略。

TensorRT与NVIDIA的性能分析工具[NVIDIA Nsight™ Systems](https://developer.nvidia.com/nsight-systems)和[NVIDIA® Deep Learning Profiler (DLProf)](https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/)集成。

TensorRT的一部分功能经过认证，可在[NVIDIA DRIVE®](https://developer.nvidia.com/drive)产品中使用。某些API仅标记为适用于NVIDIA DRIVE，并不支持一般用途。

### [1.5. ONNX](#onnx-intro)

TensorRT的主要方式是通过[ONNX](https://onnx.ai/)交换格式从框架中导入训练好的模型。TensorRT附带了一个ONNX解析器库，用于辅助导入模型。在可能的情况下，解析器向后兼容到opset 7；ONNX [Model Opset Version Converter](https://github.com/onnx/onnx/blob/master/docs/VersionConverter.md)可以帮助解决不兼容性问题。

[GitHub版本](https://github.com/onnx/onnx-tensorrt/)可能支持比TensorRT附带版本更高的opset，请参考ONNX-TensorRT [operator support matrix](https://github.com/onnx/onnx-tensorrt/blob/master/docs/operators.md)以获取支持的opset和运算符的最新信息。

有关TensorRT的ONNX运算符支持列表，请点击[此处](https://github.com/onnx/onnx-tensorrt/blob/master/docs/operators.md)。

PyTorch本地支持[ONNX导出](https://pytorch.org/docs/stable/onnx.html)。对于TensorFlow，推荐的方法是使用[tf2onnx](https://github.com/onnx/tensorflow-onnx)。

在将模型导出为ONNX后，一个很好的第一步是使用[Polygraphy](#polygraphy-ovr)运行常量折叠。这通常可以解决ONNX解析器中的TensorRT转换问题，并且通常可以简化工作流程。有关详细信息，请参考[此示例](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/cli/surgeon/02_folding_constants)。在某些情况下，可能需要进一步修改ONNX模型，例如，用插件替换子图或者用其他运算重新实现不支持的操作。为了使这个过程更容易，您可以使用[ONNX-GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)。

### [1.6. 代码分析工具](#code-analysis-tools)

有关使用valgrind和clang sanitizer工具与TensorRT的指导，请参阅[故障排除](#troubleshooting "以下部分帮助回答了关于典型用例的最常见问题。")章节。

### [1.7. API 版本控制](#versioning)

TensorRT 版本号（MAJOR.MINOR.PATCH）遵循 [语义化版本控制 2.0.0](https://semver.org/#semantic-versioning-200) 来管理其公共 API 和库 ABI。版本号的变化如下：

1. 当进行不兼容的 API 或 ABI 更改时，MAJOR 版本号会增加
2. 当以向后兼容的方式添加功能时，MINOR 版本号会增加
3. 当进行向后兼容的错误修复时，PATCH 版本号会增加

需要注意的是，语义化版本控制不适用于序列化对象。为了重用计划文件和计时缓存，版本号必须在主要版本、次要版本、修订版本和构建版本上保持一致（但在 _NVIDIA DRIVE OS 6.0_ _Developer Guide_ 中的安全运行时有一些例外情况）。校准缓存通常可以在一个主要版本内重用，但不保证兼容性。

### [1.8. 废弃策略](#deprecation)

废弃用于通知开发人员某些API和工具不再推荐使用。从8.0版本开始，TensorRT采用以下废弃策略：

* 废弃通知在[_TensorRT发布说明_](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)中进行沟通。
* 使用C++ API时：
    * API函数使用TRT\_DEPRECATED\_API宏进行标记。
    * 枚举类型使用TRT\_DEPRECATED\_ENUM宏进行标记。
    * 其他所有位置使用TRT\_DEPRECATED宏进行标记。
    * 类、函数和对象将有一个声明，说明它们何时被废弃。
* 使用Python API时，如果使用了废弃的方法和类，将在运行时发出废弃警告。
* TensorRT在废弃后提供12个月的迁移期。
* 在迁移期间，API和工具仍然可用。
* 迁移期结束后，API和工具将按照语义版本控制的方式进行删除。

对于在TensorRT 7.x中明确废弃的API和工具，12个月的迁移期从TensorRT 8.0 GA发布日期开始。

### [1.9. 硬件支持寿命](#hw-support-lifetime)

TensorRT 8.5.3 将是最后一个支持 NVIDIA Kepler（SM 3.x）和 NVIDIA Maxwell（SM 5.x）设备的版本。这些设备将不再在 TensorRT 8.6 中受到支持。NVIDIA Pascal（SM 6.x）设备将在 TensorRT 8.6 中被弃用。

### [1.10. 支持](#支持)

有关TensorRT的支持、资源和信息可以在[https://developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt)上找到。这包括博客、示例等内容。

此外，您可以访问NVIDIA DevTalk TensorRT论坛，网址为[https://devtalk.nvidia.com/default/board/304/tensorrt/](https://devtalk.nvidia.com/default/board/304/tensorrt/)，论坛上有关于TensorRT的一切。该论坛提供了寻找答案、建立联系以及参与与客户、开发人员和TensorRT工程师的讨论的可能性。

### [1.11. 报告错误](#bug-reporting)

NVIDIA非常重视各种类型的反馈。如果您遇到任何问题，请按照[报告TensorRT问题](#reporting-issues)部分的说明来报告问题。

## [2. TensorRT的功能](#fit)

本章概述了您可以使用TensorRT做什么。它旨在为所有TensorRT用户提供帮助。
### [2.1. C++和Python API](#api)

TensorRT的API提供了C++和Python的语言绑定，几乎具有相同的功能。Python API可以与Python数据处理工具包和库（如NumPy和SciPy）进行互操作。C++ API可能更高效，并且在一些合规性要求方面可能更好，例如在汽车应用中。

注意：Python API并非适用于所有平台。有关更多信息，请参阅_[NVIDIA TensorRT支持矩阵](https://docs.nvidia.com/deeplearning/sdk/tensorrt-support-matrix/index.html)_。
### [2.2. 编程模型](#prog-model)

TensorRT分为两个阶段。在第一个阶段，通常是离线进行的，您提供TensorRT模型定义，TensorRT会针对目标GPU进行优化。在第二个阶段，您使用优化后的模型来进行推理。
### [2.2.1. 构建阶段](#build-phase)

TensorRT构建阶段的最高级接口是_Builder_（[C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_builder.html)，[Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Builder.html)）。构建器负责优化模型并生成_Engine_。

要构建一个引擎，您必须：

*   创建网络定义。
*   为构建器指定配置。
*   调用构建器来创建引擎。

_NetworkDefinition_接口（[C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_network_definition.html)，[Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/Network.html#inetworkdefinition)）用于定义模型。将模型从框架以ONNX格式导出，并使用TensorRT的ONNX解析器填充网络定义是将模型传输到TensorRT的最常见路径。但是，您也可以使用TensorRT的_Layer_（[C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_layer.html)，[Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/LayerBase.html#ilayer)）和_Tensor_（[C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_tensor.html)，[Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/LayerBase.html#itensor)）接口逐步构建定义。

无论您选择哪种方式，您还必须定义哪些张量是网络的输入和输出。没有标记为输出的张量被视为可以由构建器优化掉的临时值。输入和输出张量必须命名，以便在运行时，TensorRT知道如何将输入和输出缓冲区绑定到模型。
_BuilderConfig_ 接口（[C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_builder_config.html)，[Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/BuilderConfig.html)）用于指定 TensorRT 如何优化模型。在可用的配置选项中，您可以控制 TensorRT 减少计算精度的能力，控制内存和运行时执行速度之间的权衡，以及限制 CUDA® 内核的选择。由于构建器可能需要几分钟甚至更长时间来运行，您还可以控制构建器如何搜索内核，并缓存搜索结果以供后续运行使用。

在拥有网络定义和构建器配置之后，您可以调用构建器来创建引擎。构建器会消除无效计算，折叠常量，并重新排序和组合操作，以在 GPU 上更高效地运行。它可以选择减少浮点计算的精度，要么仅使用 16 位浮点数运行它们，要么通过量化浮点值来执行计算，以便可以使用 8 位整数进行计算。它还会使用不同的数据格式计时每个层的多个实现，然后计算一个最佳的调度来执行模型，最小化内核执行和格式转换的综合成本。

构建器以一种称为 _plan_ 的序列化形式创建引擎，该引擎可以立即反序列化，或保存到磁盘以供以后使用。

注意：

*   由 TensorRT 创建的引擎特定于创建它们的 TensorRT 版本和 GPU。
*   TensorRT 的网络定义不会深度复制参数数组（例如卷积的权重）。因此，在构建阶段完成之前，您不能释放这些数组的内存。当使用 ONNX 解析器导入网络时，解析器拥有权重，因此在构建阶段完成之前不能销毁解析器。
*   构建器会计算算法的执行时间以确定最快的算法。与其他GPU任务同时运行构建器可能会干扰计时，导致优化效果不佳。
### [2.2.2. 运行时阶段](#runtime-phase)

TensorRT 执行阶段的最高级接口是 _运行时_ ([C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_runtime.html), [Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Runtime.html))。

在使用运行时时，通常会执行以下步骤：

*   反序列化计划以创建引擎。
*   从引擎创建执行上下文。

然后，重复执行以下步骤：

*   为推理填充输入缓冲区。
*   在执行上下文上调用 enqueueV3() 来运行推理。

_引擎_ 接口 ([C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_cuda_engine.html), [Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Engine.html)) 表示一个优化的模型。您可以查询引擎以获取有关网络的输入和输出张量的信息 - 期望的维度、数据类型、数据格式等等。

从引擎创建的 _执行上下文_ 接口 ([C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_execution_context.html), [Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/ExecutionContext.html)) 是调用推理的主要接口。执行上下文包含与特定调用相关联的所有状态 - 因此您可以有多个与单个引擎相关联的上下文，并且可以并行运行它们。

在调用推理时，您必须在适当的位置设置输入和输出缓冲区。根据数据的性质，这可以是在 CPU 或 GPU 内存中。如果根据您的模型不明显，您可以查询引擎以确定在哪个内存空间提供缓冲区。
在设置好缓冲区之后，可以异步地调用推理（enqueueV3）。所需的内核会在一个CUDA流上被排队，并且尽快将控制返回给应用程序。某些网络需要在CPU和GPU之间进行多次控制传输，因此控制可能不会立即返回。要等待异步执行完成，请使用[cudaStreamSynchronize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html)在流上进行同步。
### [2.3. 插件](#plugins)

TensorRT具有一个“插件”接口，允许应用程序提供TensorRT本身不支持的操作的实现。通过在TensorRT的PluginRegistry中创建和注册插件，ONNX解析器可以在翻译网络时找到这些插件。

TensorRT附带了一套插件库，并且可以在[这里](https://github.com/NVIDIA/TensorRT/tree/main/plugin)找到其中许多插件的源代码以及一些其他插件。

有关更多详细信息，请参阅[使用自定义层扩展TensorRT](#extending "NVIDIA TensorRT支持许多类型的层，并且其功能不断扩展；然而，有时支持的层不能满足模型的特定需求。在这种情况下，可以通过实现自定义层来扩展TensorRT，通常称为插件。")章节。
### [2.4. 类型和精度](#types-precision)

TensorRT支持使用FP32、FP16、INT8、Bool和INT32数据类型进行计算。

当TensorRT选择CUDA内核来实现网络中的浮点运算时，默认情况下会使用FP32实现。有两种方法可以配置不同精度的级别：

*   在模型级别控制精度，可以使用BuilderFlag选项（[C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/namespacenvinfer1.html#abdc74c40fe7a0c3d05d2caeccfbc29c1)，[Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/BuilderConfig.html#tensorrt.BuilderFlag)）来指示TensorRT在搜索最快实现时可能选择较低精度的实现（因为较低精度通常更快，如果允许的话，它通常会选择较低精度）。
    
    因此，您可以轻松地指示TensorRT在整个模型中使用FP16进行计算。对于输入动态范围约为1的规则化模型，这通常会产生显著的加速，而准确性变化可以忽略不计。
    
*   对于更细粒度的控制，如果网络的一部分对数值敏感或需要较高的动态范围，则可以为该层指定算术精度。

有关更多详细信息，请参阅[减少精度](#reduced-precision)部分。
### [2.5. 量化](#quantization)

TensorRT支持量化浮点数，即将浮点值线性压缩并舍入为8位整数。这显著提高了算术吞吐量，同时减少了存储需求和内存带宽。在量化浮点张量时，TensorRT必须知道其动态范围-即重要表示的值范围-在量化时超出此范围的值将被截断。

动态范围信息可以由构建器计算（称为_校准_），基于代表性输入数据。或者，您可以在框架中执行量化感知训练，并将具有必要动态范围信息的模型导入到TensorRT中。

有关详细信息，请参阅[使用INT8](#working-with-int8)章节。
### [2.6. 张量和数据格式](#data-layout)

在定义网络时，TensorRT假设张量由多维C风格数组表示。每个层对其输入有特定的解释：例如，2D卷积会假设其输入的最后三个维度以CHW格式表示 - 不能使用例如WHC格式的选项。有关每个层如何解释其输入，请参阅[TensorRT运算符参考](https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/index.html)。

请注意，张量的元素数量最多限制为2^31-1个。

在优化网络时，TensorRT在内部执行转换（包括到HWC，以及更复杂的格式），以使用最快的CUDA核心。通常，格式的选择是为了优化性能，应用程序无法控制选择。然而，底层数据格式在I/O边界（网络输入和输出，以及传递数据给插件和从插件传递数据）处是可见的，以允许应用程序最小化不必要的格式转换。

有关更多详细信息，请参阅[I/O格式](#reformat-free-network-tensors "TensorRT使用许多不同的数据格式优化网络。为了允许在TensorRT和客户端应用程序之间高效传递数据，这些底层数据格式在网络I/O边界处是可见的，即对于标记为网络输入或输出的张量以及在传递数据给插件和从插件传递数据时。对于其他张量，TensorRT选择能够实现最快整体执行速度的格式，并可能插入重新格式化以提高性能。")部分。
### [2.7. 动态形状](#dynamic-shapes)

默认情况下，TensorRT根据模型在其定义时的输入形状（批处理大小、图像大小等）进行优化。然而，构建器可以配置为允许在运行时调整输入维度。为了实现这一点，您可以在构建器配置中指定一个或多个优化配置文件（[C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html)，[Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/OptimizationProfile.html?highlight=optimizationprofile)），其中包含每个输入的最小和最大形状，以及该范围内的优化点。

TensorRT为每个配置文件创建一个优化引擎，选择适用于\[最小值，最大值\]范围内所有形状并且对于优化点最快的CUDA核。然后，您可以在运行时从配置文件中选择。

有关更多详细信息，请参阅[使用动态形状](#work_dynamic_shapes "动态形状是在运行时延迟指定一些或所有张量维度的能力。动态形状可以通过C++和Python接口使用。")章节。
### [2.8. DLA](#dla-ovr)

TensorRT支持NVIDIA的深度学习加速器（DLA），它是许多NVIDIA SoCs上的专用推理处理器，支持TensorRT的部分层次。TensorRT允许您在DLA上执行网络的一部分，其余部分在GPU上执行；对于可以在任一设备上执行的层次，您可以在生成器配置中按层次选择目标设备。

有关更多详细信息，请参阅[与DLA一起使用](#dla_topic)章节。
### [2.9. 更新权重](#updating-weights)

在构建引擎时，您可以指定它以后可能会更新其权重。如果您经常更新模型的权重而不改变结构，例如在强化学习或在保留相同结构的情况下重新训练模型时，这将非常有用。使用 _Refitter_ ([C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_refitter.html), [Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Refitter.html)) 接口执行权重更新。

有关更多详细信息，请参阅[重新安装引擎](#refitting-engine-c "TensorRT可以使用新的权重重新安装引擎，而无需重新构建引擎，但构建时必须指定此选项：") 部分。
### [2.10. trtexec工具](#trtexec-ovr)

在示例目录中包含了一个命令行包装工具，名为trtexec。trtexec是一个无需开发自己的应用程序即可使用TensorRT的工具。trtexec工具有三个主要用途：

*   在随机或用户提供的输入数据上进行网络基准测试。
*   从模型中生成序列化引擎。
*   从构建器生成序列化的计时缓存。

有关详细信息，请参阅[trtexec](#trtexec "在示例目录中包含了一个命令行包装工具，名为trtexec。trtexec是一个无需开发自己的应用程序即可快速利用TensorRT的工具。trtexec工具有三个主要用途：")部分。
### [2.11. Polygraphy](#polygraphy-ovr)

Polygraphy是一个旨在帮助在TensorRT和其他框架中运行和调试深度学习模型的工具包。它包括一个使用此API构建的[Python API](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/polygraphy)和[命令行界面（CLI）](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/polygraphy/tools)。

使用Polygraphy，您可以：

*   在多个后端（如TensorRT和ONNX-Runtime）之间运行推断并比较结果（例如[API](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/api/01_comparing_frameworks)，[CLI](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/run/01_comparing_frameworks)）。
*   将模型转换为不同的格式，例如使用训练后量化的TensorRT引擎（例如[API](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/api/04_int8_calibration_in_tensorrt)，[CLI](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/convert/01_int8_calibration_in_tensorrt)）。
*   查看有关各种类型模型的信息（例如[CLI](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/inspect)）。
*   在命令行上修改ONNX模型：
    *   提取子图（例如[CLI](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/surgeon/01_isolating_subgraphs)）。
    *   简化和清理（例如[CLI](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/surgeon/02_folding_constants)）。
*   隔离TensorRT中的故障策略（例如[CLI](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/debug/01_debugging_flaky_trt_tactics)）。

有关更多详细信息，请参阅[Polygraphy存储库](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy)。
