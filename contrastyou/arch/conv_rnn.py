#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ConvRNN.py
@Time    :   2020/03/09
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   convrnn cell
'''
import typing as t
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

T = t.TypeVar("T")
single_or_tup2 = t.Union[T, t.Tuple[T, T]]


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0],
                                                 out_channels=v[1],
                                                 kernel_size=v[2],
                                                 stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0],
                               out_channels=v[1],
                               kernel_size=v[2],
                               stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError
    return nn.Sequential(OrderedDict(layers))


'''
class CGRU_cell(nn.Module):
    """
    ConvGRU Cell
    """

    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CGRU_cell, self).__init__()
        self.shape = shape
        self.input_channels = input_channels
        # kernel_size of input_to_state equals state_to_state
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      2 * self.num_features, self.filter_size, (1, 1),
                      self.padding),
            nn.GroupNorm(2 * self.num_features // 32, 2 * self.num_features))
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      self.num_features, self.filter_size, (1, 1), self.padding),
            nn.GroupNorm(self.num_features // 32, self.num_features))

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        # seq_len=10 for moving_mnist
        if hidden_state is None:
            htprev = torch.zeros(inputs.size(1), self.num_features,
                                 self.shape[0], self.shape[1]).cuda()
        else:
            htprev = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(htprev.size(0), self.input_channels,
                                self.shape[0], self.shape[1]).cuda()
            else:
                x = inputs[index, ...]

            combined_1 = torch.cat((x, htprev), 1)  # X_t + H_t-1
            gates = self.conv1(combined_1)  # W * (X_t + H_t-1)

            zgate, rgate = torch.split(gates, self.num_features, dim=1)
            # zgate, rgate = gates.chunk(2, 1)
            z = torch.sigmoid(zgate)
            r = torch.sigmoid(rgate)

            combined_2 = torch.cat((x, r * htprev),
                                   1)  # h' = tanh(W*(x+r*H_t-1))
            ht = self.conv2(combined_2)
            ht = torch.tanh(ht)
            htnext = (1 - z) * htprev + z * ht
            output_inner.append(htnext)
            htprev = htnext
        return torch.stack(output_inner), htnext
'''


class CLSTM_cell(nn.Module):
    """ConvLSTMCell
    """

    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        # in this way the output has the same size
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      4 * self.num_features, self.filter_size, (1, 1),
                      self.padding),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features))

    def forward(self, inputs: Tensor, hidden_state: t.Tuple[Tensor, Tensor] = None, seq_len=10):
        return self.forward_implement(inputs, hidden_state, seq_len=seq_len)
        # #  seq_len=10 for moving_mnist
        # if hidden_state:
        #     assert inputs.device == hidden_state[0].device, (input.device, hidden_state[0].device)
        # cur_device = inputs.device
        # if hidden_state is None:
        #     hx = torch.zeros(*(inputs.shape[1], self.num_features, self.shape[0],
        #                        self.shape[1]), device=cur_device)
        #     cx = torch.zeros_like(hx)
        # else:
        #     hx, cx = hidden_state
        # output_inner = []
        # for index in range(seq_len):
        #     if inputs is None:
        #         x = torch.zeros(hx.shape[0], self.input_channels, self.shape[0],
        #                         self.shape[1], device=cur_device)
        #     else:
        #         x = inputs[index, ...]
        #
        #     combined = torch.cat((x, hx), 1)
        #     gates = self.conv(combined)  # gates: S, num_features*4, H, W
        #     # it should return 4 tensors: i,f,g,o
        #     ingate, forgetgate, cellgate, outgate = torch.split(
        #         gates, self.num_features, dim=1)
        #     ingate = torch.sigmoid(ingate)
        #     forgetgate = torch.sigmoid(forgetgate)
        #     cellgate = torch.tanh(cellgate)
        #     outgate = torch.sigmoid(outgate)
        #
        #     cy = (forgetgate * cx) + (ingate * cellgate)
        #     hy = outgate * torch.tanh(cy)
        #     output_inner.append(hy)
        #     hx = hy
        #     cx = cy
        # return torch.stack(output_inner), (hy, cy)

    def iterate(self, step_input: Tensor, hidden_state: t.Tuple[Tensor, Tensor]):
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

    def forward_implement(self, inputs: Tensor, hidden_state: t.Tuple[Tensor, Tensor], *, seq_len: int):
        cur_device = next(self.parameters()).device

        if hidden_state is None:
            hx = torch.zeros(*(inputs.shape[1], self.num_features, self.shape[0],
                               self.shape[1]), device=cur_device)
            cx = torch.zeros_like(hx)
            hidden_state = (hx, cx)

        hx, cx = hidden_state
        outs = []

        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(hx.shape[0], self.input_channels, self.shape[0],
                                self.shape[1], device=cur_device)
            else:
                x = inputs[index]
            hy, hidden_state = self.iterate(x, hidden_state)
            outs.append(hy)
        return torch.stack(outs, dim=0), hidden_state


class CLSTM_cell2(nn.Module):

    def __init__(self, shape: t.Tuple[int, int], input_channels: int, filter_size: single_or_tup2[int], class_num: int,
                 num_features: int):
        super().__init__()
        self.shape = shape  # H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.class_num = class_num
        # in this way the output has the same size
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features + self.class_num,
                      4 * self.num_features, self.filter_size, (1, 1),
                      self.padding),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features))
        self.squeeze = nn.Sequential(nn.Conv2d(self.num_features, class_num, (1, 1), ),
                                     nn.Tanh())

    def forward(self, image: Tensor, logits: Tensor, *, seq_len: int):
        assert seq_len > 0
        cur_device = next(self.parameters()).device
        _, c_image, _, _ = image.shape

        B, c_pred, H, W = logits.shape

        hx = torch.zeros(*(B, self.num_features, self.shape[0],
                           self.shape[1]), device=cur_device)
        cx = torch.zeros_like(hx)

        errors = []
        refined_output = []
        for index in range(seq_len):
            if index == 0:
                cur_logits = logits
                norm_cur_logits = self.zero_one_normalize(cur_logits)
                state_input = torch.cat([image, norm_cur_logits], dim=1)
            else:
                cur_logits = logits + sum(errors)
                norm_cur_logits = self.zero_one_normalize(cur_logits)
                state_input = torch.cat([image, norm_cur_logits], dim=1)

            hy, (hx, cx) = self.iterate(state_input, (hx, cx))
            error = self.squeeze(hy)
            errors.append(error)
            refined_output.append(cur_logits + error)
        return torch.stack(errors, dim=1), torch.stack(refined_output, dim=1)

    @staticmethod
    def zero_one_normalize(logits: Tensor):
        logits = logits - logits.min().detach()
        logits = logits / (logits.max().detach() - logits.min().detach() + 1e-6)
        return logits

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


class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            # index sign from 1
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)

    def forward_by_stage(self, inputs, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        outputs_stage, state_stage = rnn(inputs, None)
        return outputs_stage, state_stage

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)  # to S,B,1,64,64
        hidden_states = []
        logger.debug(inputs.size())
        for i in range(1, self.blocks + 1):
            inputs, state_stage = self.forward_by_stage(
                inputs, getattr(self, 'stage' + str(i)),
                getattr(self, 'rnn' + str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)


class Decoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index),
                    make_layers(params))

    def forward_by_stage(self, inputs, state, subnet, rnn):
        inputs, state_stage = rnn(inputs, state, seq_len=10)
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        return inputs

        # input: 5D S*B*C*H*W

    def forward(self, hidden_states):
        inputs = self.forward_by_stage(None, hidden_states[-1],
                                       getattr(self, 'stage3'),
                                       getattr(self, 'rnn3'))
        for i in list(range(1, self.blocks))[::-1]:
            inputs = self.forward_by_stage(inputs, hidden_states[i - 1],
                                           getattr(self, 'stage' + str(i)),
                                           getattr(self, 'rnn' + str(i)))
        inputs = inputs.transpose(0, 1)  # to B,S,1,64,64
        return inputs


class activation:

    def __init__(self, act_type, negative_slope=0.2, inplace=True):
        super().__init__()
        self._act_type = act_type
        self.negative_slope = negative_slope
        self.inplace = inplace

    def __call__(self, input):
        if self._act_type == 'leaky':
            return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)
        elif self._act_type == 'relu':
            return F.relu(input, inplace=self.inplace)
        elif self._act_type == 'sigmoid':
            return torch.sigmoid(input)
        else:
            raise NotImplementedError


class ED(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        state = self.encoder(input)
        output = self.decoder(state)
        return output


if __name__ == '__main__':
    image = torch.randn(10, 1, 224, 224)
    logits = torch.randn(10, 4, 224, 224)
    model = CLSTM_cell2(shape=(224, 224), input_channels=1, filter_size=3, class_num=4, num_features=16)
    errors, corrected_logits = model(image, logits, seq_len=1)
    pass
