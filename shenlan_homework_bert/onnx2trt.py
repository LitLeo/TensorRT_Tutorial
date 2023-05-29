import tensorrt as trt
import os

import numpy as np
import onnx

import torch
from torch.nn import functional as F
import numpy as np
import os
from transformers import BertTokenizer, BertForMaskedLM

# import onnxruntime as ort
import transformers

from trt_helper import *

def onnx2trt(onnxFile, plan_name):
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

    input_ids = network.get_input(0)
    token_type_ids = network.get_input(1)
    input_mask = network.get_input(2)
    profile.set_shape(input_ids.name, (1, 6), (1, 64), (1, 256))
    profile.set_shape(token_type_ids.name, (1, 6), (1, 64), (1, 256))
    profile.set_shape(input_mask.name, (1, 6), (1, 64), (1, 256))
    config.add_optimization_profile(profile)


    # engineString = builder.build_serialized_network(network, config)
    # if engineString == None:
        # print("Failed building engine!")
        # exit()
    # print("Succeeded building engine!")

    engine = builder.build_engine(network, config)
    if not engine:
        raise RuntimeError("build_engine failed")
    print("Succeeded building engine!")

    print("Serializing Engine...")
    serialized_engine = engine.serialize()
    if serialized_engine is None:
        raise RuntimeError("serialize failed")

    with open(plan_name, "wb") as fout:
        fout.write(serialized_engine)

def trt_infer(plan_name):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text = "The capital of France, " + tokenizer.mask_token + ", contains the Eiffel Tower."
    encoded_input = tokenizer.encode_plus(text, return_tensors = "pt")
    mask_index = torch.where(encoded_input["input_ids"][0] == tokenizer.mask_token_id)
    print(encoded_input)

    """
    TensorRT Initialization
    """
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

    # # TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    # handle = ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
    # if not handle:
        # raise RuntimeError("Could not load plugin library. Is `libnvinfer_plugin.so` on your LD_LIBRARY_PATH?")

    infer_helper = InferHelper(plan_name, TRT_LOGGER)
    # input_list = [encoded_input['input_ids'].detach().numpy(), encoded_input['attention_mask'].detach().numpy(), encoded_input['token_type_ids'].detach().numpy()]
    input_list = [encoded_input['input_ids'].detach().numpy(), encoded_input['token_type_ids'].detach().numpy(), encoded_input['attention_mask'].detach().numpy()]

    output = infer_helper.infer(input_list)
    print(output)

    logits = torch.from_numpy(output[0])
    softmax = F.softmax(logits, dim = -1)
    mask_word = softmax[0, mask_index, :]
    top_10 = torch.topk(mask_word, 10, dim = 1)[1][0]
    print(top_10)
    print("model test topk10 output:")
    for token in top_10:
        word = tokenizer.decode([token])
        new_sentence = text.replace(tokenizer.mask_token, word)
        print(new_sentence)

    print("hhhh, the result is wrong, debug yourself")

if __name__ == '__main__':
    src_onnx = 'bert-base-uncased/model-sim.onnx'
    plan_name = 'bert-base-uncased/bert.plan'
    onnx2trt(src_onnx, plan_name)
    trt_infer(plan_name)
