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

def decoder_surgeon(src_onnx, dst_onnx):

    graph = gs.import_onnx(onnx.load(src_onnx))

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

                                        print(start_node)
                                        print(end_node)
                                        print(weight_node.inputs)
                                        print(bias_node.inputs)
                                        print("=======================")
                                        layer_norm_plugin.inputs = [start_node.inputs[0], weight_node.inputs[1], bias_node.inputs[1]]
                                        layer_norm_plugin.outputs = end_node.outputs

                                        start_node.inputs = []
                                        end_node.outputs = []

                                        # layer_norm
                                        start_node = None
                                        end_node = None

                                        weight_node = None
                                        bias_node = None

    # graph.outputs.append(Expand_23_node.outputs[0])
    # print(graph.outputs )
    graph.cleanup()
    onnx.save(gs.export_onnx(graph), dst_onnx)

    model = onnx.load(dst_onnx)
    # print(graph.outputs )
    # assert(0)

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
    config.max_workspace_size = 13<<30
    
    #config.flags = config.flags & ~(1 << int(trt.BuilderFlag.TF32))

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

    min_batch = 1
    opt_batch = 16
    max_batch = 64
    # max_batch = 16

    min_seq_len = 16
    opt_seq_len = 64
    max_seq_len = 256
    # max_seq_len = 64


    inputTensor = network.get_input(0)
    profile.set_shape(inputTensor.name, (min_batch, min_seq_len, 256), (opt_batch, opt_seq_len, 256), (max_batch, max_seq_len, 256))

    inputTensor = network.get_input(1)
    profile.set_shape(inputTensor.name, [min_batch], [opt_batch], [max_batch])

    inputTensor = network.get_input(2)
    profile.set_shape(inputTensor.name, [min_batch, 10, 64], [opt_batch, 10, 64], [max_batch, 10, 64])

    inputTensor = network.get_input(3)
    profile.set_shape(inputTensor.name, [min_batch, 10], [opt_batch, 10], [max_batch, 10])

    inputTensor = network.get_input(4)
    profile.set_shape(inputTensor.name, [min_batch, 10], [opt_batch, 10], [max_batch, 10])


    config.add_optimization_profile(profile)
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")

    with open(plan_name, "wb") as fout:
        fout.write(engineString)

if __name__ == '__main__':
    src_onnx = '/workspace/decoder.onnx'
    surgeon_onnx = 'decoder_surgeon.onnx'
    plan_name = 'decoder.plan'
    decoder_surgeon(src_onnx, surgeon_onnx)
    onnx2trt(surgeon_onnx, plan_name)
