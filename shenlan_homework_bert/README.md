# BERT Inference Using TensorRT Python API

深蓝学院《CUDA入门与深度神经网络加速》 课程，TensorRT 部分的作业。

内容是使用TensorRT Python API 手动搭建 BERT 模型。学员不需要从零开始搭建，这里已经搭好了一个模型结构框架，进行相应的填充即可。


### 一 文件信息
1. model2onnx.py 使用pytorch 运行 bert 模型，生成demo 输入输出和onnx模型
2. onnx2trt.py 将onnx，使用onnx-parser转成trt模型，并infer
3. builder.py 输入onnx模型，并进行转换
4. trt_helper.py 对trt的api进行封装，方便调用
5. calibrator.py int8 calibrator 代码
6. 基础款LayerNormPlugin.zip 用于学习的layer_norm_plugin

## 二 模型信息
### 2.1 介绍
1. 标准BERT 模型，12 层, hidden_size = 768
2. 不考虑tokensizer部分，输入是ids，输出是score
3. 为了更好的理解，降低作业难度，将mask逻辑去除，只支持batch=1 的输入

BERT模型可以实现多种NLP任务，作业选用了fill-mask任务的模型

```
输入：
The capital of France, [mask], contains the Eiffel Tower.

topk10输出：
The capital of France, paris, contains the Eiffel Tower.
The capital of France, lyon, contains the Eiffel Tower.
The capital of France,, contains the Eiffel Tower.
The capital of France, tolilleulouse, contains the Eiffel Tower.
The capital of France, marseille, contains the Eiffel Tower.
The capital of France, orleans, contains the Eiffel Tower.
The capital of France, strasbourg, contains the Eiffel Tower.
The capital of France, nice, contains the Eiffel Tower.
The capital of France, cannes, contains the Eiffel Tower.
The capital of France, versailles, contains the Eiffel Tower.
```

### 2.2 输入输出信息
输入
1. input_ids[1, -1]： int 类型，input ids，从BertTokenizer获得
2. token_type_ids[1, -1]：int 类型，全0
3. position_ids[1, -1]：int 类型，[0, 1, ..., len(input_ids) - 1]

输出
1. logit[1, -1, 768]

## 三 作业内容
### 1. 学习 使用trt python api 搭建网络
填充trt_helper.py 中的空白函数。学习使用api 搭建网络的过程。
### 2. 编写plugin
trt不支持layer_norm算子，编写layer_norm plugin，并将算子添加到网络中，进行验证。
### 3. 观察GELU算子的优化过程
GELU算子使用一堆基础算子堆叠实现的（详细见trt_helper.py addGELU函数），直观上感觉很分散，计算量比较大。  
但在实际build过程中，这些算子会被合并成一个算子。build 过程中需要设置log为trt.Logger.VERBOSE，观察build过程。
### 4. 进行 fp16 加速

### 5. 进行 int8 加速

## 深度思考
### 1. 还有那些算子能合并？
1. emb_layernorm 模块，3个embedding和一个layer_norm，是否可以合并到一个kernel中？
2. self_attention_layer 中，softmax和scale操作，是否可以合并到一个kernel中？
3. self_attention_layer，要对qkv进行三次矩阵乘和3次转置。三个矩阵乘是否可以合并到一起，相应三个转置是否可以？如果合并了，那么后面的q*k和attn*v，该怎么计算？
4. self_output_layer中，add 和 layer_norm 层是否可以合并？

以上问题的答案，见 https://github.com/NVIDIA/TensorRT/tree/release/6.0/demo/BERT

### 2. 除了上面那些，还能做那些优化？
1. 增加mask逻辑，多batch一起inference，padding的部分会冗余计算。比如一句话10个字，第二句100个字，那么padding后的大小是[2, 100]，会有90个字的冗余计算。这部分该怎么优化？除了模型内部优化，是否还可以在外部拼帧逻辑进行优化？
2. self_attention_layer层，合并到一起后的QkvToContext算子，是否可以支持fp16计算？int8呢？

以上问题答案，见 https://github.com/NVIDIA/TensorRT/tree/release/8.2/demo/BERT

