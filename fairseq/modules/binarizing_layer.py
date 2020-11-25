# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

# import dynamicconv_cuda
from fairseq import utils
# from fairseq.modules.unfold import unfold1d
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout


# TODO: See if elegant way to treat 0s
class binarizingFunction(Function):

    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)


class BinarizingLayer(nn.Module):
    def __init__(self):
        super(BinarizingLayer, self).__init__()

    def forward(self, x, incremental_state=None, query=None, unfold=None):
        return binarizingFunction.apply(x)

class BinarizingLinear(nn.Module):
    def __init__(self, innerlinear):
        super(BinarizingLinear, self).__init__()
        self.innerlinear=innerlinear
        self.weight = self.innerlinear.weight
        self.bias = self.innerlinear.bias
        # Learnable alpha
        self.alpha = Parameter(torch.rand((1, 1, self.weight.size()[0]))*0.5, requires_grad=True)

    def forward(self, input):
        linout = F.linear(binarizingFunction.apply(input), binarizingFunction.apply(self.innerlinear.weight)) 
        # print(linout.size())
        # print(self.alpha.size())
        # print(linout.size())
        # print(self.innerlinear.bias.size())
        # print(linout[:2,:2,:2])
        # print(self.alpha.expand_as(linout).size())
        linout = linout * self.alpha.expand_as(linout) + torch.reshape(self.innerlinear.bias,(1,1,-1))
        # print(linout[:2,:2,:2])
        return linout  