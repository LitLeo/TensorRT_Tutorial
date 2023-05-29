import torch
from torch.nn import functional as F
import numpy as np
import os
from transformers import BertTokenizer, BertForMaskedLM

# import onnxruntime as ort
import transformers

print("pytorch:", torch.__version__)
# print("onnxruntime version:", ort.__version__)
# print("onnxruntime device:", ort.get_device())
print("transformers:", transformers.__version__)

BERT_PATH = 'bert-base-uncased'


def model_test(model, tokenizer, text):
    print("==============model test===================")
    encoded_input = tokenizer.encode_plus(text, return_tensors = "pt")
    mask_index = torch.where(encoded_input["input_ids"][0] == tokenizer.mask_token_id)

    output = model(**encoded_input)
    print(output[0].shape)

    logits = output.logits
    softmax = F.softmax(logits, dim = -1)
    mask_word = softmax[0, mask_index, :]
    top_10 = torch.topk(mask_word, 10, dim = 1)[1][0]
    print("model test topk10 output:")
    for token in top_10:
        word = tokenizer.decode([token])
        new_sentence = text.replace(tokenizer.mask_token, word)
        print(new_sentence)

    # save inputs and output
    print("Saving inputs and output to case_data.npz ...")
    position_ids = torch.arange(0, encoded_input['input_ids'].shape[1]).int().view(1, -1)
    print(position_ids)
    input_ids=encoded_input['input_ids'].int().detach().numpy()
    token_type_ids=encoded_input['token_type_ids'].int().detach().numpy()
    print(input_ids.shape)

    # save data
    npz_file = BERT_PATH + '/case_data.npz'
    np.savez(npz_file,
             input_ids=input_ids,
             token_type_ids=token_type_ids,
             position_ids=position_ids,
             logits=output[0].detach().numpy())

    data = np.load(npz_file)
    print(data['input_ids'])

def model2onnx(model, tokenizer, text):
    print("===================model2onnx=======================")
    encoded_input = tokenizer.encode_plus(text, return_tensors = "pt")
    print(encoded_input)

    # convert model to onnx
    model.eval()
    export_model_path = BERT_PATH + "/model.onnx"
    opset_version = 12
    symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
    torch.onnx.export(model,                                            # model being run
                      args=tuple(encoded_input.values()),                      # model input (or a tuple for multiple inputs)
                      f=export_model_path,                              # where to save the model (can be a file or file-like object)
                      opset_version=opset_version,                      # the ONNX version to export the model to
                      do_constant_folding=False,                         # whether to execute constant folding for optimization
                      input_names=['input_ids',                         # the model's input names
                                   'attention_mask',
                                   'token_type_ids'],
                    output_names=['logits'],                    # the model's output names
                    dynamic_axes={'input_ids': symbolic_names,        # variable length axes
                                  'attention_mask' : symbolic_names,
                                  'token_type_ids' : symbolic_names,
                                  'logits' : symbolic_names})
    print("Model exported at ", export_model_path)

if __name__ == '__main__':

    if not os.path.exists(BERT_PATH):
        print(f"Download {BERT_PATH} model first!")
        assert(0)

    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    model = BertForMaskedLM.from_pretrained(BERT_PATH, return_dict = True)
    text = "The capital of France, " + tokenizer.mask_token + ", contains the Eiffel Tower."

    model_test(model, tokenizer, text)
    model2onnx(model, tokenizer, text)
