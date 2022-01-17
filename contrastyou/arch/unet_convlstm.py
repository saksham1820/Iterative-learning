import torch
from deepclustering2.loss import KL_div
from torch import nn, Tensor

__all__ = ["UNet"]

from contrastyou.arch import UNet
from contrastyou.arch.conv_rnn import CLSTM_cell3


class LSTM_Corrected_Unet(nn.Module):

    def __init__(self, *, layer_dim, codec, input_dim=3, num_classes=1, num_features,
                          seq_len=3, detach=False, iter_to_use, **kwargs):
        super().__init__()
        self._unet = UNet(input_dim=input_dim, num_classes=num_classes)
        self._correct_model = CLSTM_cell3(
            shape=(layer_dim, layer_dim,),
            input_channels=input_dim,
            filter_size=3,
            class_num=num_features,
            num_features=num_features
        )
        self._codec = codec
        self._layer_dim = layer_dim
        self._seq_len = seq_len
        self._detach = detach
        self._iter_to_use = iter_to_use

    def forward(self, x):
        seg1_out = self._unet.forward1(x, layer_dim = str(self._layer_dim), codec = self._codec)
        if len(seg1_out) != 1:
            last_logit = seg1_out[-1]
        else:
            last_logit = seg1_out

        seg1_out = seg1_out[:-1]

        batch_copy = x
        resized_x = torch.nn.functional.interpolate(batch_copy, size = self._layer_dim)
        #breakpoint()
        errors, corrected_logits = self._correct_model(
            resized_x,
            last_logit.detach() if self._detach else last_logit,
            seq_len=self._seq_len)


        bb_out = seg1_out + (corrected_logits[:, self._iter_to_use-1, :, :, :], )
        logits = self._unet.forward2(bb_out, layer_dim = str(self._layer_dim), codec = self._codec)

        return logits, corrected_logits, errors



    @property
    def num_classes(self):
        return self._unet.num_classes

    @property
    def num_iter(self):
        return self._seq_len

    @property
    def num_iters(self):
        return self._seq_len


class LSTMErrorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._kl_loss = KL_div(reduction="none")

    def forward(self, logits: Tensor, onehot: Tensor):
        b, t, c, h, w = logits.shape
        assert torch.Size([b, c, h, w]) == onehot.shape
        return self._kl_loss(logits.moveaxis(1, 2).softmax(1),
                             onehot.unsqueeze(1).repeat(1, t, 1, 1, 1).moveaxis(1, 2), disable_assert=True).mean(
            dim=[0, 2, 3],
        )
