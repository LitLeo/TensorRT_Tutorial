import argparse
import ctypes
import json
import numpy as np
import os
import os.path
import re
import sys
import time
import onnx
import pycuda.autoinit

# TensorRT
import tensorrt as trt
from calibrator import BertCalibrator as BertCalibrator
from trt_helper import *

import torch
from torch.nn import functional as F
from transformers import BertTokenizer, BertForMaskedLM

# import onnxruntime as ort
import transformers

"""
TensorRT Initialization
"""
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
# TRT_LOGGER = trt.Logger(trt.Logger.INFO)

handle = ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
if not handle:
    raise RuntimeError("Could not load plugin library. Is `libnvinfer_plugin.so` on your LD_LIBRARY_PATH?")

handle = ctypes.CDLL("LayerNorm.so", mode=ctypes.RTLD_GLOBAL)
if not handle:
    raise RuntimeError("Could not load plugin library. Is `LayerNorm.so` on your LD_LIBRARY_PATH?")

trt.init_libnvinfer_plugins(TRT_LOGGER, "")
plg_registry = trt.get_plugin_registry()

class BertConfig:
    def __init__(self, bert_config_path, use_fp16, use_int8, use_strict):
        with open(bert_config_path, "r") as f:
            data = json.load(f)
            self.num_attention_heads = data["num_attention_heads"]
            self.hidden_size = data["hidden_size"]
            self.intermediate_size = data["intermediate_size"]
            self.num_hidden_layers = data["num_hidden_layers"]
            self.head_size = self.hidden_size // self.num_attention_heads
            self.use_fp16 = use_fp16
            self.use_int8 = use_int8
            self.use_strict = use_strict
            self.is_calib_mode = False

def set_tensor_name(tensor, prefix, name):
    tensor.name = prefix + name

def set_output_name(layer, prefix, name, out_idx = 0):
    set_tensor_name(layer.get_output(out_idx), prefix, name)

def set_output_range(layer, maxval, out_idx = 0):
    layer.get_output(out_idx).set_dynamic_range(-maxval, maxval)

def get_mha_dtype(config):
    dtype = trt.float32
    if config.use_fp16:
        dtype = trt.float16
    # Multi-head attention doesn't use INT8 inputs and output by default unless it is specified.
    if config.use_int8 and config.use_int8_multihead and not config.is_calib_mode:
        dtype = trt.int8
    return int(dtype)

def custom_fc(config, network, input_tensor, out_dims, W):
    pf_out_dims = trt.PluginField("out_dims", np.array([out_dims], dtype=np.int32), trt.PluginFieldType.INT32)
    pf_W = trt.PluginField("W", W.numpy(), trt.PluginFieldType.FLOAT32)
    pf_type = trt.PluginField("type_id", np.array([1 if config.use_fp16 else 0], np.int32), trt.PluginFieldType.INT32)
    pfc = trt.PluginFieldCollection([pf_out_dims, pf_W, pf_type])
    fc_plugin = fc_plg_creator.create_plugin("fcplugin", pfc)
    plug_inputs = [input_tensor]
    out_dense = network.add_plugin_v2(plug_inputs, fc_plugin)
    return out_dense

def self_attention_layer(network_helper, prefix, config, weights_dict, input_tensor, imask):
    num_heads = config.num_attention_heads
    head_size = config.head_size

    q_w = weights_dict[prefix + "attention_self_query_kernel"]
    q_b = weights_dict[prefix + "attention_self_query_bias"]
    q = network_helper.addLinear(input_tensor, q_w, q_b)
    q = network_helper.addShuffle(q, None, (0, -1, num_heads, head_size), (0, 2, 1, 3), "att_q_view_transpose")

    k_w = weights_dict[prefix + "attention_self_key_kernel"]
    k_b = weights_dict[prefix + "attention_self_key_bias"]
    k = network_helper.addLinear(input_tensor, k_w, k_b)
    k = network_helper.addShuffle(k, None, (0, -1, num_heads, head_size), (0, 2, 3, 1), "att_k_view_and transpose")
    # k = network_helper.addShuffle(k, None, (0, -1, self.h, self.d_k), (0, 2, 3, 1), "att_k_view_and transpose")

    v_w = weights_dict[prefix + "attention_self_value_kernel"]
    v_b = weights_dict[prefix + "attention_self_value_bias"]
    v = network_helper.addLinear(input_tensor, v_w, v_b)
    v = network_helper.addShuffle(v, None, (0, -1, num_heads, head_size), (0, 2, 1, 3), "att_v_view_and transpose")

    scores = network_helper.addMatMul(q, k, "q_mul_k")

    scores = network_helper.addScale(scores, 1/math.sqrt(head_size))

    attn = network_helper.addSoftmax(scores, dim=-1)

    attn = network_helper.addMatMul(attn, v, "matmul(p_attn, value)")

    attn = network_helper.addShuffle(attn, (0, 2, 1, 3), (0, -1, num_heads * head_size), None, "attn_transpose_and_reshape")

    return attn

def self_output_layer(network_helper, prefix, config, weights_dict, hidden_states, input_tensor):

    out_w = weights_dict[prefix + "attention_output_dense_kernel"]
    out_b = weights_dict[prefix + "attention_output_dense_bias"]
    out = network_helper.addLinear(hidden_states, out_w, out_b)

    out = network_helper.addAdd(out, input_tensor)

    gamma = weights_dict[prefix + "attention_output_layernorm_gamma"]
    beta = weights_dict[prefix + "attention_output_layernorm_beta"]
    out = network_helper.addLayerNorm(out, gamma, beta)

    return out

def attention_layer(network_helper, prefix, config, weights_dict, input_tensor, imask):
    attn = self_attention_layer(network_helper, prefix, config, weights_dict, input_tensor, imask)

    out = self_output_layer(network_helper, prefix, config, weights_dict, attn, input_tensor)

    return out


def transformer_layer(network_helper, prefix, config, weights_dict, input_tensor, imask):
    num_heads = config.num_attention_heads
    head_size = config.head_size

    attention_output = attention_layer(network_helper, prefix, config, weights_dict, input_tensor, imask)

    # BertIntermediate
    intermediate_w = weights_dict[prefix + "intermediate_dense_kernel"]
    intermediate_w = np.transpose(intermediate_w)
    intermediate_b = weights_dict[prefix + "intermediate_dense_bias"]
    intermediate_output = network_helper.addLinear(attention_output, intermediate_w, intermediate_b)

    intermediate_output = network_helper.addGELU(intermediate_output)

    # BertOutput
    output_w = weights_dict[prefix + "output_dense_kernel"]
    output_w = np.transpose(output_w)
    output_b = weights_dict[prefix + "output_dense_bias"]
    layer_output = network_helper.addLinear(intermediate_output, output_w, output_b)

    layer_output = network_helper.addAdd(layer_output, attention_output)

    gamma = weights_dict[prefix + "output_layernorm_gamma"]
    beta = weights_dict[prefix + "output_layernorm_beta"]
    layer_output = network_helper.addLayerNorm(layer_output, gamma, beta)

    return layer_output

def transformer_output_layer(network_helper, config, weights_dict, input_tensor):
    num_heads = config.num_attention_heads
    head_size = config.head_size

    # BertIntermediate
    dense_w = weights_dict["cls_predictions_transform_dense_kernel"]
    dense_w = np.transpose(dense_w)
    dense_b = weights_dict["cls_predictions_transform_dense_bias"]
    dense_output = network_helper.addLinear(input_tensor, dense_w, dense_b)

    dense_output = network_helper.addGELU(dense_output)

    gamma = weights_dict["cls_predictions_transform_layernorm_gamma"]
    beta = weights_dict["cls_predictions_transform_layernorm_beta"]
    layer_output = network_helper.addLayerNorm(dense_output, gamma, beta)

    # BertOutput
    output_w = weights_dict["embeddings_word_embeddings"]
    output_w = np.transpose(output_w)
    output_b = weights_dict["cls_predictions_bias"]
    layer_output = network_helper.addLinear(layer_output, output_w, output_b)

    return layer_output

def bert_model(network_helper, config, weights_dict, input_tensor, input_mask):
    """
    Create the bert model
    """
    prev_input = input_tensor
    for layer in range(0, config.num_hidden_layers):
        ss = "l{}_".format(layer)
        prev_input = transformer_layer(network_helper, ss, config,  weights_dict, prev_input, input_mask)

    return prev_input


def onnx_to_trt_name(onnx_name):
    """
    Converting variables in the onnx checkpoint to names corresponding to the naming convention used in the TF version, expected by the builder
    """
    onnx_name = onnx_name.lower()
    toks = [t.strip('_') for t in onnx_name.split('.')]

    if toks[0] == 'bert':
        if toks[1] == 'embeddings': #embedding
            if toks[-2] == 'layernorm': #bias->beta, weight->gamma
                toks[-1] = 'beta' if toks[-1] == 'bias' else 'gamma'
            else: #embeddings: drop "_weight" suffix
                toks = toks[:-1]
            toks = toks[1:]
        elif toks[1] == 'encoder': #transformer
            if toks[-2] == 'layernorm': #bias->beta, weight->gamma
                toks[-1] = 'beta' if toks[-1] == 'bias' else 'gamma'
            elif (toks[-2] == 'dense' or toks[-2] in {'key', 'value', 'query'}) and toks[-1] == 'weight':
                toks[-1] = 'kernel'
            elif (toks[-3] == 'dense' or toks[-3] in {'key', 'value', 'query'}) and toks[-1] == 'amax':
                if toks[-2] == 'weight_quantizer':
                    toks[-2] = 'kernel'
                elif toks[-2] == 'input_quantizer':
                    toks[-2] = 'input'
            toks = toks[3:]
            toks[0] = 'l{}'.format(int(toks[0]))
    elif 'cls' in onnx_name:
        if 'transform' in onnx_name:
            if toks[-2] == 'layernorm': #bias->beta, weight->gamma
                toks[-1] = 'beta' if toks[-1] == 'bias' else 'gamma'
            elif (toks[-2] == 'dense' and toks[-1] == 'weight'):
                toks[-1] = 'kernel'
        # else:
            # name = 'pooler_bias' if toks[-1] == 'bias' else 'pooler_kernel'
    else:
        print("Encountered unknown case:", onnx_name)
        assert(False)
    parsed = '_'.join(toks)
    return parsed

def load_onnx_weights_and_quant(path, config):
    """
    Load the weights from the onnx checkpoint
    """
    N = config.num_attention_heads
    H = config.head_size
    hidden_size = config.hidden_size

    model = onnx.load(path)
    weights = model.graph.initializer

    tensor_dict = {}
    for w in weights:
        if "position_ids" in w.name:
            continue

        a = onnx_to_trt_name(w.name)
        # print(w.name + " " + str(w.dims))
        print(a + " " + str(w.dims))
        b = np.frombuffer(w.raw_data, np.float32).reshape(w.dims)
        tensor_dict[a] = b

    weights_dict = tensor_dict

    TRT_LOGGER.log(TRT_LOGGER.INFO, "Found {:} entries in weight map".format(len(weights_dict)))
    return weights_dict

def emb_layernorm(network_helper, config, weights_dict, builder_config, sequence_lengths, batch_sizes):
    # int8 only support some of the sequence length, we dynamic on sequence length is not allowed.
    input_ids = network_helper.addInput(name="input_ids", dtype=trt.int32, shape=(1, -1))
    token_type_ids = network_helper.addInput(name="token_type_ids", dtype=trt.int32, shape=(1, -1))
    position_ids = network_helper.addInput(name="position_ids", dtype=trt.int32, shape=(1, -1))

    word_embeddings = weights_dict["embeddings_word_embeddings"]
    position_embeddings = weights_dict["embeddings_position_embeddings"]
    token_type_embeddings = weights_dict["embeddings_token_type_embeddings"]
    print(word_embeddings)

    input_embeds = network_helper.addEmbedding(input_ids, word_embeddings)
    token_type_embeds = network_helper.addEmbedding(token_type_ids, token_type_embeddings)
    position_embeds = network_helper.addEmbedding(position_ids, position_embeddings)

    embeddings = network_helper.addAdd(input_embeds, position_embeds)
    embeddings = network_helper.addAdd(embeddings, token_type_embeds)

    gamma = weights_dict["embeddings_layernorm_gamma"]
    beta = weights_dict["embeddings_layernorm_beta"]
    out = network_helper.addLayerNorm(embeddings, gamma, beta)

    return out

def build_engine(workspace_size, config, weights_dict, vocab_file, calibrationCacheFile, calib_num):
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    max_seq_length = 200
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch_flag) as network, builder.create_builder_config() as builder_config:
        builder_config.max_workspace_size = workspace_size * (1024 * 1024)
        if config.use_fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)

        if config.use_int8:
            builder_config.set_flag(trt.BuilderFlag.INT8)

            calibrator = BertCalibrator("calibrator_data.txt", "bert-base-uncased", calibrationCacheFile, 1, max_seq_length, 1000)
            builder_config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)
            builder_config.int8_calibrator = calibrator

        if config.use_strict:
            builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        # builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        # only use the largest sequence when in calibration mode
        if config.is_calib_mode:
            sequence_lengths = sequence_lengths[-1:]

        network_helper = TrtNetworkHelper(network, plg_registry, TRT_LOGGER)

        # Create the network
        embeddings = emb_layernorm(network_helper, config, weights_dict, builder_config, None, None)

        bert_out = bert_model(network_helper, config, weights_dict, embeddings, None)
        # network_helper.markOutput(bert_out)

        cls_output = transformer_output_layer(network_helper, config, weights_dict, bert_out)

        network_helper.markOutput(cls_output)

        profile = builder.create_optimization_profile()
        min_shape = (1, 1)
        opt_shape = (1, 50)
        max_shape = (1, max_seq_length)
        profile.set_shape("input_ids", min=min_shape, opt=opt_shape, max=max_shape)
        profile.set_shape("position_ids", min=min_shape, opt=opt_shape, max=max_shape)
        profile.set_shape("token_type_ids", min=min_shape, opt=opt_shape, max=max_shape)
        builder_config.add_optimization_profile(profile)

        build_start_time = time.time()
        engine = builder.build_engine(network, builder_config)
        build_time_elapsed = (time.time() - build_start_time)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "build engine in {:.3f} Sec".format(build_time_elapsed))
        if config.use_int8:
            calibrator.free()
        return engine

def generate_calibration_cache(sequence_lengths, workspace_size, config, weights_dict, squad_json, vocab_file, calibrationCacheFile, calib_num):
    """
    BERT demo needs a separate engine building path to generate calibration cache.
    This is because we need to configure SLN and MHA plugins in FP32 mode when
    generating calibration cache, and INT8 mode when building the actual engine.
    This cache could be generated by examining certain training data and can be
    reused across different configurations.
    """
    # dynamic shape not working with calibration, so we need generate a calibration cache first using fulldims network
    if not config.use_int8 or os.path.exists(calibrationCacheFile):
        return calibrationCacheFile

    # generate calibration cache
    saved_use_fp16 = config.use_fp16
    config.use_fp16 = False
    config.is_calib_mode = True

    with build_engine([1], workspace_size, sequence_lengths, config, weights_dict, squad_json, vocab_file, calibrationCacheFile, calib_num) as engine:
        TRT_LOGGER.log(TRT_LOGGER.INFO, "calibration cache generated in {:}".format(calibrationCacheFile))

    config.use_fp16 = saved_use_fp16
    config.is_calib_mode = False

def test_text(infer_helper, BERT_PATH):
    print("==============model test===================")
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    text = "The capital of France, " + tokenizer.mask_token + ", contains the Eiffel Tower."
    encoded_input = tokenizer.encode_plus(text, return_tensors = "pt")
    mask_index = torch.where(encoded_input["input_ids"][0] == tokenizer.mask_token_id)

    input_ids = encoded_input['input_ids'].int().detach().numpy()
    token_type_ids = encoded_input['token_type_ids'].int().detach().numpy()
    position_ids = torch.arange(0, encoded_input['input_ids'].shape[1]).int().view(1, -1).numpy()
    input_list = [input_ids, token_type_ids, position_ids]

    output = infer_helper.infer(input_list)
    print(output)

    logits = torch.from_numpy(output[0])

    softmax = F.softmax(logits, dim = -1)
    mask_word = softmax[0, mask_index, :]
    top_10 = torch.topk(mask_word, 10, dim = 1)[1][0]
    print("model test topk10 output:")
    for token in top_10:
        word = tokenizer.decode([token])
        new_sentence = text.replace(tokenizer.mask_token, word)
        print(new_sentence)

def test_case_data(infer_helper, case_data_path):
    print("==============test_case_data===================")
    case_data = np.load(case_data_path)

    input_ids = case_data['input_ids']
    token_type_ids = case_data['token_type_ids']
    position_ids = case_data['position_ids']
    print(input_ids)
    print(input_ids.shape)
    print(token_type_ids)
    print(position_ids)

    logits_output = case_data['logits']

    trt_outputs = infer_helper.infer([input_ids, token_type_ids, position_ids])
    trt_outputs = infer_helper.infer([input_ids, token_type_ids, position_ids])
    # infer_helper.infer([input_ids], [output_start])

    rtol = 1e-02
    atol = 1e-02

    # res = np.allclose(logits_output, trt_outputs[0], rtol, atol)
    # print ("Are the start outputs are equal within the tolerance:\t", res)
    print(logits_output.sum())
    print(logits_output)
    print(trt_outputs[0].sum())
    print(trt_outputs[0])

def main():
    parser = argparse.ArgumentParser(description="TensorRT BERT Sample", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-x", "--onnx", required=False, help="The ONNX model file path.")
    parser.add_argument("-o", "--output", required=True, default="bert_base_384.engine", help="The bert engine file, ex bert.engine")
    parser.add_argument("-c", "--config-dir", required=True,
                        help="The folder containing the bert_config.json, which can be downloaded e.g. from https://github.com/google-research/bert#pre-trained-models or by running download_models.py in dle/TensorFlow/LanguageModeling/BERT/data/pretrained_models_google")
    parser.add_argument("-f", "--fp16", action="store_true", help="Indicates that inference should be run in FP16 precision", required=False)
    parser.add_argument("-i", "--int8", action="store_true", help="Indicates that inference should be run in INT8 precision", required=False)
    parser.add_argument("-t", "--strict", action="store_true", help="Indicates that inference should be run in strict precision mode", required=False)
    parser.add_argument("-w", "--workspace-size", default=1000, help="Workspace size in MiB for building the BERT engine", type=int)
    parser.add_argument("-v", "--vocab-file", default="./pre-trained_model/uncased_L-24_H-1024_A-16/vocab.txt", help="Path to file containing entire understandable vocab", required=False)
    parser.add_argument("-n", "--calib-num", default=100, help="calibration batch numbers", type=int)
    parser.add_argument("-p", "--calib-path", help="calibration cache path", required=False)

    args, _ = parser.parse_known_args()

    # args.batch_size = args.batch_size or [1]
    # args.sequence_length = args.sequence_length or [128]

    # cc = pycuda.autoinit.device.compute_capability()
    # if cc[0] * 10 + cc[1] < 75 and args.force_int8_multihead:
        # raise RuntimeError("--force-int8-multihead option is only supported on Turing+ GPU.")
    # if cc[0] * 10 + cc[1] < 72 and args.force_int8_skipln:
        # raise RuntimeError("--force-int8-skipln option is only supported on Xavier+ GPU.")

    bert_config_path = os.path.join(args.config_dir, "config.json")
    TRT_LOGGER.log(TRT_LOGGER.INFO, "Using configuration file: {:}".format(bert_config_path))

    config = BertConfig(bert_config_path, args.fp16, args.int8, args.strict)

    if args.calib_path != None:
        calib_cache = args.calib_path
    else:
        calib_cache = "BertL{}H{}A{}CalibCache".format(config.num_hidden_layers, config.head_size, config.num_attention_heads)

    if args.onnx != None:
        weights_dict = load_onnx_weights_and_quant(args.onnx, config)
    else:
        raise RuntimeError("You need either specify ONNX using option --onnx to build TRT BERT model.")

    with build_engine(args.workspace_size, config, weights_dict, args.vocab_file, calib_cache, args.calib_num) as engine:
        TRT_LOGGER.log(TRT_LOGGER.VERBOSE, "Serializing Engine...")
        serialized_engine = engine.serialize()
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Saving Engine to {:}".format(args.output))
        with open(args.output, "wb") as fout:
            fout.write(serialized_engine)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Done.")

    infer_helper = InferHelper(args.output, TRT_LOGGER)

    test_case_data(infer_helper, args.config_dir + "/case_data.npz")

    test_text(infer_helper, args.config_dir)

if __name__ == "__main__":
    main()
