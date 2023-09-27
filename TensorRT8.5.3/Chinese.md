# 开发者指南 :: NVIDIA深度学习TensorRT文档## [修订历史](#revision-history)

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