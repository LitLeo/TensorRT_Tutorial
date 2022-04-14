import torch
import torch.nn.functional as F

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask

feat = torch.zeros(1, 100, 80)
feat_len = torch.tensor([feat.size(1)]).int()

class BaseEncoder(torch.nn.Module):
    def forward(self, xs, xs_lens):
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)

        masks = masks[:, :, :-2:2][:, :, :-2:2]

        masks1 = ~masks
        masks2 = torch.unsqueeze(masks, 1).int()

        xs = F.log_softmax(xs, dim=-1)

        return xs, masks1, masks2


model = BaseEncoder()
# score, mask = model(feat, feat_len)

input_names, output_names = ["feat", "feat_len"], ["output", "masks1", "masks2"]
dynamic_axes= {'feat':{0:'batch_size', 1: 'seq_len'},
               'feat_len': {0: 'batch_size'},
               'output':{0:'batch_size', 1: 'seq_len'}}

torch.onnx.export(model, (feat, feat_len), "mask.onnx",
                 opset_version=13, verbose=True,
                 input_names=input_names, output_names=output_names,
                 dynamic_axes = dynamic_axes)
