import tensorrt as trt
from glob import glob
import ctypes
import os

import numpy as np
import onnx

import onnx_graphsurgeon as gs
import onnx
import numpy as np
from onnx import TensorProto

from calibrator import *

def encoder_surgeon_layer_norm(graph):
    # layer_norm
    start_node = None
    end_node = None

    weight_node = None
    bias_node = None

    layer_norm_layer_idx = 0

    for node in graph.nodes:

        if node.op == 'ReduceMean' and node.o(0).op == "Sub":
            start_node = node
            sub_node = node.o(0)
            if sub_node.o(0).op == "Pow":
                pow_node = sub_node.o(0)
                # print(pow_node)
                if pow_node.o(0).op == "ReduceMean":
                    rm_node = pow_node.o(0)
                    # print(rm_node)
                    if rm_node.o(0).op == "Add":
                        add_node = rm_node.o(0)
                        # print(add_node)
                        if add_node.o(0).op == "Sqrt":
                            sqrt_node = add_node.o(0)
                            if sqrt_node.o(0).op == "Div":
                                div_node = sqrt_node.o(0)
                                if div_node.o(0).op == "Mul":
                                    mul_node = div_node.o(0)
                                    weight_node = mul_node
                                    if mul_node.o(0).op == "Add":
                                        add_node = mul_node.o(0)
                                        bias_node = add_node
                                        end_node = add_node

                                        layer_norm_plugin = gs.Node("LayerNorm", "LayerNorm-" + str(layer_norm_layer_idx))
                                        layer_norm_layer_idx = layer_norm_layer_idx + 1
                                        graph.nodes.append(layer_norm_plugin)

                                        # print(start_node)
                                        # print(end_node)
                                        # print(weight_node.inputs)
                                        # print(bias_node.inputs)
                                        # print("=======================")
                                        layer_norm_plugin.inputs = [start_node.inputs[0], weight_node.inputs[1], bias_node.inputs[1]]
                                        layer_norm_plugin.outputs = end_node.outputs

                                        start_node.inputs = []
                                        end_node.outputs = []

                                        # layer_norm
                                        start_node = None
                                        end_node = None

                                        weight_node = None
                                        bias_node = None

    return graph

def get_pre_node(graph, node):
    for n in graph.nodes:
        if n.outputs[0] == node.inputs[0]:
            return n


def encoder_surgeon_skip_layer_norm(graph):
    # layer_norm
    start_node = None
    end_node = None

    weight_node = None
    bias_node = None

    layer_norm_layer_idx = 0

    skip_layer_norm_idx = 0

    remove_nodes = []
    for node in graph.nodes:
        merge_flag = False
        if node.op == 'LayerNorm':
            pre_node = get_pre_node(graph, node)
            if (pre_node.op == 'Add'):
                add_node = pre_node
                layer_norm_node = node

                if len(add_node.inputs) == 2:
                    # v2, 2 inputs and 2 outputs
                    plugin = gs.Node("SkipLayerNormV2", "SkipLayerNormV2-" + str(skip_layer_norm_idx))
                    graph.nodes.append(plugin)

                    plugin.inputs = [add_node.inputs[0], add_node.inputs[1], layer_norm_node.inputs[1], layer_norm_node.inputs[2]]
                    plugin.outputs = [layer_norm_node.outputs[0], add_node.outputs[0]]
                    print("add SkipLayerNormV2")
                    print(add_node.name)
                    print(layer_norm_node.name)
                    print(plugin)
                    print("========")

                    remove_nodes.append(add_node)
                    remove_nodes.append(layer_norm_node)

                    # merge_flag = True
                # if merge_flag:
                    # graph.nodes.remove(add_node)
                    # graph.nodes.remove(layer_norm_node)
                    # add_node.inputs = []
                    # add_node.outputs = []
                    # layer_norm_node.inputs = []
                    # layer_norm_node.outputs = []
                    skip_layer_norm_idx = skip_layer_norm_idx + 1

    for n in remove_nodes:
        n.inputs = []
        n.outputs = []

    return graph


def encoder_surgeon(src_onnx, dst_onnx):

    graph = gs.import_onnx(onnx.load(src_onnx))

    GreaterOrEqual_27_node = None
    Slice_79_node = None
    Slice_84_node = None
    Cast_26_node = None
    Not_30_node = None
    Expand_23_node = None
    Unsqueeze_29_node = None
    log_softmax_node = None

    unsqueeze_len = gs.Node(op="Unsqueeze", name="Unsqueeze_len")
    unsqueeze_len.outputs = [gs.Variable(name="unsqueeze_len_output", dtype=None, shape=None)]
    unsqueeze_mask = gs.Node(op="Unsqueeze", name="unsqueeze_mask")
    unsqueeze_mask.outputs = [gs.Variable(name="unsqueeze_mask_output", dtype=None, shape=None)]

    softmax_node = gs.Node(op="Softmax", name="softmax")
    log_node = gs.Node(op="Log", name="log")

    graph.nodes.append(unsqueeze_len)
    graph.nodes.append(unsqueeze_mask)
    graph.nodes.append(softmax_node)
    graph.nodes.append(log_node)

    for node in graph.nodes:

        if node.name == 'Slice_79':
            Slice_79_node = node

        if node.name == 'Slice_84':
            Slice_84_node = node

        if node.name == 'GreaterOrEqual_27':
            GreaterOrEqual_27_node = node

        if node.name == 'Not_30':
            Not_30_node = node

        if node.name == 'Cast_26':
            Cast_26_node = node

        if node.name == 'Expand_23':
            Expand_23_node = node

        if node.name == 'Unsqueeze_29':
            Unsqueeze_29_node = node

        if node.name == 'Not_30':
            Not_30_node = node

        if node.name == 'LogSoftmax_1987':
            log_softmax_node = node

    # print("Slice_79_node===============")
    # print(Slice_79_node)
    # print("Slice_84_node===============")
    # print(Slice_84_node)
    # print("GreaterOrEqual_27_node===============")
    # print(GreaterOrEqual_27_node)
    # print("Cast_26_node===============")
    # print(Cast_26_node)
    # print("Expand_23_node===============")
    # print(Expand_23_node)

    # print("Not_30_node===============")
    # print(Not_30_node)

    # print("Unsqueeze_29===============")
    # print(Unsqueeze_29_node)

    unsqueeze_len.inputs = [Cast_26_node.outputs[0], Unsqueeze_29_node.inputs[1]]
    unsqueeze_mask.inputs = [Expand_23_node.outputs[0], Unsqueeze_29_node.inputs[1]]

    # print("===============")
    # print(unsqueeze_len)
    # print("===============")

    Slice_79_node.inputs[0] = unsqueeze_mask.outputs[0]
    v_613 = Slice_84_node.outputs[0]
    Slice_84_node.outputs[0] = Not_30_node.outputs[0]

    GreaterOrEqual_27_node.inputs = [Slice_84_node.outputs[0], unsqueeze_len.outputs[0]]

    Not_30_node.inputs[0] = GreaterOrEqual_27_node.outputs[0]
    Not_30_node.outputs[0] = v_613

    softmax_out_var = gs.Variable(name="softmax_out", dtype=None, shape=None)
    softmax_node.inputs = [log_softmax_node.inputs[0]]
    softmax_node.outputs = [softmax_out_var]
    log_node.inputs = [softmax_out_var]
    log_node.outputs = [log_softmax_node.outputs[0]]

    log_softmax_node.inputs = []
    log_softmax_node.outputs = []

    # dixoap not
    for node in graph.nodes:
        if node.op == "Not" and node.inputs[0].name == '613':
            node.o(0).inputs[0] = GreaterOrEqual_27_node.outputs[0]

    graph = encoder_surgeon_layer_norm(graph)
    graph.cleanup()
    graph = encoder_surgeon_skip_layer_norm(graph)

    # for node in graph.nodes:
        # if node.op == "LayerNorm":
            # print(node.outputs[0])
            # if node.name == "LayerNorm-12":
                # print(node)
                # node.outputs[0].shape = ['B', 'T', 256]
                # node.outputs[0].dtype = np.float32
                # print(node.outputs[0].shape)
                # print(graph.outputs)
                # graph.outputs = [node.outputs[0]]
                # print(graph.outputs)

    # graph.outputs.append(Expand_23_node.outputs[0])
    # print(graph.outputs )
    graph.cleanup()
    onnx.save(gs.export_onnx(graph), dst_onnx)

    model = onnx.load(dst_onnx)
    # print(graph.outputs )
    # assert(0)

def subsample_fp16(network):
    print("===================subsample_fp16======================")
    include_list = ['ConvTranspose']
    check_list = ['Conv_35', 'Relu_36', 'Conv_37', 'Relu_38', 'Transpose_51', \
                  'Reshape_60', 'MatMul_61', 'Add_62']
    for i, n_i in enumerate(network):
        if n_i.name in check_list or n_i.name.split('_')[0] in include_list:
            layer = network[i]
            layer.precision = trt.float16
            print(f'Network Layer {i}:  {n_i.name}, {n_i.type}, {n_i.precision}, is_set: {n_i.precision_is_set }')
    print("===================subsample_fp16======================")
    return network

def matmul_add_fp16(network):
    print("===================matmul_add_fp16======================")
    fp16_layers = []
    matmul_layer = None
    add_layer = None
    for i, n_i in enumerate(network):
        if n_i.name.split('_')[0] == 'MatMul':
            matmul_layer = network[i]

        if n_i.name.split('_')[0] == 'Add':
            add_layer = network[i]

            if matmul_layer is not None and add_layer is not None:
                matmul_layer.precision = trt.float16
                add_layer.precision = trt.float16
                print(f'Network Layer {i}:  {n_i.name}, {n_i.type}, {n_i.precision}, is_set: {n_i.precision_is_set }')
                matmul_layer = None

            add_layer = None

    print("===================matmul_add_fp16======================")

    return network

def onnx2trt(onnxFile, plan_name):

    soFileList = glob("LayerNormPlugin/*.so")

    if len(soFileList) > 0:
        print("Find Plugin %s!"%soFileList)
    else:
        print("No Plugin!")
    for soFile in soFileList:
        ctypes.cdll.LoadLibrary(soFile)

    logger = trt.Logger(trt.Logger.VERBOSE)

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config.max_workspace_size = 3<<30

    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnxFile):
        print("Failed finding onnx file!")
        exit()
    print("Succeeded finding onnx file!")
    with open(onnxFile, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed parsing ONNX file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
    print("Succeeded parsing ONNX file!")

    inputTensor = network.get_input(0)
    inputlengths = network.get_input(1)
    profile.set_shape(inputTensor.name, (1, 16, 80), (16, 64, 80), (64,256,80))
    profile.set_shape(inputlengths.name, [1], [16], [64])
    config.add_optimization_profile(profile)

    if 0:
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)
        encoder_calibrator = EncoderCalibrator("/workspace/data/calibration.npz", "encoder.cache", 100)
        config.int8_calibrator = encoder_calibrator

        not_int8_layer_type = [trt.LayerType.SHAPE, trt.LayerType.SELECT, trt.LayerType.FILL, \
                               trt.LayerType.IDENTITY, trt.LayerType.GATHER, trt.LayerType.CONSTANT, \
                               trt.LayerType.SLICE, trt.LayerType.SHUFFLE, trt.LayerType.CONCATENATION]

        for i, n_i in enumerate(network):
            # if n_i.name in check_list or n_i.name.split('_')[0] in include_list:
            layer = network[i]
            # if layer.type not in layer_types:
                # layer_types.append(layer.type)
            if layer.type not in not_int8_layer_type:
                # print(layer.precision)
                layer.precision = trt.float32
            print(f'Network Layer {i}:  {n_i.name}, {n_i.type}, {n_i.precision}, is_set: {n_i.precision_is_set }')

        # # network = subsample_fp16(network)
        # network = matmul_add_fp16(network)

    config.flags = config.flags & ~(1 << int(trt.BuilderFlag.TF32))
    config.set_flag(trt.BuilderFlag.STRICT_TYPES)

    # layer_types = []
    # for i, n_i in enumerate(network):
        # layer = network[i]
        # if layer.type not in layer_types:
            # layer_types.append(layer.type)

    # print(layer_types)
    # assert(0)

    if 0:
        config.set_flag(trt.BuilderFlag.FP16)

        # not_fp16_layer_type = [trt.LayerType.SHAPE, trt.LayerType.SELECT, trt.LayerType.FILL, \
                               # trt.LayerType.IDENTITY, trt.LayerType.GATHER, trt.LayerType.CONSTANT, \
                               # trt.LayerType.SLICE, trt.LayerType.SHUFFLE, trt.LayerType.CONCATENATION]

        not_fp16_layer_type = [trt.LayerType.SHAPE, \
                               trt.LayerType.IDENTITY, trt.LayerType.GATHER, trt.LayerType.CONSTANT, \
                               trt.LayerType.SLICE, trt.LayerType.SHUFFLE, trt.LayerType.CONCATENATION]

        for i, n_i in enumerate(network):
            # if n_i.name in check_list or n_i.name.split('_')[0] in include_list:
            layer = network[i]
            # if layer.type not in layer_types:
                # layer_types.append(layer.type)
            if layer.type not in not_fp16_layer_type and layer.precision != trt.int32:
                # print(layer.precision)
                layer.precision = trt.float32
            # print(f'Network Layer {i}:  {n_i.name}, {n_i.type}, {n_i.precision}, is_set: {n_i.precision_is_set }')
                for oi in range(0, layer.num_outputs):
                    if (layer.get_output_type(oi) == trt.DataType.FLOAT):
                        layer.set_output_type(oi, trt.DataType.FLOAT)
                    print(f'Network Layer {i}:  {n_i.name}, {n_i.type}, {n_i.get_output_type(oi)}, {n_i.output_type_is_set(oi)}')

        # network = subsample_fp16(network)
        # network = matmul_add_fp16(network)
        # assert(0)

    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")

    with open(plan_name, "wb") as fout:
        fout.write(engineString)

if __name__ == '__main__':
    src_onnx = '/workspace/encoder.onnx'
    surgeon_onnx = 'encoder_surgeon.onnx'
    plan_name = 'encoder.plan'
    encoder_surgeon(src_onnx, surgeon_onnx)
    onnx2trt(surgeon_onnx, plan_name)
