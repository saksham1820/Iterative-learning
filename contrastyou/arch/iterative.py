import typing as t

import torch
import torch.nn as nn
from torch import Tensor

from contrastyou.arch import UNet

T = t.TypeVar("T")
single_or_tup2 = t.Union[T, t.Tuple[T, T]]


class CLSTM_Module(nn.Module):
    """This cell is used to unbound the error, without any normalization. """

    def __init__(self, shape: t.Tuple[int, int], filter_size: single_or_tup2[int], class_num: int,
                 num_features: int):
        super().__init__()
        self.shape = shape  # H, W
        self.filter_size = filter_size
        self.num_features = num_features
        self.class_num = class_num
        # in this way the output has the same size
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.num_features + self.class_num,
                      4 * self.num_features, self.filter_size, (1, 1),
                      self.padding),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features))
        self.squeeze = nn.Conv2d(self.num_features, class_num, (1, 1), )

    def forward(self, logits: Tensor, *, hidden_state: t.Tuple[Tensor, Tensor] = None, seq_len: int):
        assert seq_len > 0
        cur_device = next(self.parameters()).device

        B, c_pred, H, W = logits.shape

        if hidden_state is None:
            hx = torch.zeros(*(B, self.num_features, self.shape[0],
                               self.shape[1]), device=cur_device)
            cx = torch.zeros_like(hx)
        else:
            hx, cx = hidden_state
        errors = []
        refined_output = []
        for index in range(seq_len):
            if index == 0:
                cur_logits = logits
                norm_cur_logits = cur_logits
                state_input = norm_cur_logits
            else:
                cur_logits = logits + sum(errors)
                norm_cur_logits = cur_logits
                state_input = norm_cur_logits

            hy, (hx, cx) = self.iterate(state_input, (hx, cx))
            error = self.squeeze(hy)
            errors.append(error)
            refined_output.append(cur_logits + error)
        return torch.stack(errors, dim=1), torch.stack(refined_output, dim=1)

    def iterate(self, step_input: Tensor, hidden_state: t.Tuple[Tensor, Tensor]) -> \
        t.Tuple[Tensor, t.Tuple[Tensor, Tensor]]:
        hx, cx = hidden_state

        combined = torch.cat((step_input, hx), 1)
        gates = self.conv(combined)  # gates: S, num_features*4, H, W
        # it should return 4 tensors: i,f,g,o
        ingate, forgetgate, cellgate, outgate = torch.split(
            gates, self.num_features, dim=1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        hx = hy
        cx = cy
        return hy, (hx, cx)


class RefinementModel(nn.Module):

    def __init__(self, *, input_dim=1, num_classes=4, seq_len: int, detach=False, **kwargs):
        super().__init__()
        self._seq_len = seq_len
        self._unet = UNet(input_dim=input_dim, num_classes=num_classes)
        self._refinement = CLSTM_Module(shape=(224, 224), filter_size=3, class_num=4, num_features=16)
        self._detach = detach
        self.num_classes = num_classes

    def forward(self, image: Tensor):
        # d2 features
        logits, _, features = self._unet(image, return_features=True)
        features = features[-1]
        errors, refinements = self._refinement(
            logits if not self._detach else logits.detach(),
            hidden_state=(features, features), seq_len=self._seq_len)
        return logits, refinements, errors

    @property
    def num_iters(self):
        return self._seq_len


if __name__ == '__main__':
    image = torch.randn(10, 1, 224, 224)

    model = RefinementModel(input_dim=1, num_classes=4, seq_len=5)
    result = model(image)
