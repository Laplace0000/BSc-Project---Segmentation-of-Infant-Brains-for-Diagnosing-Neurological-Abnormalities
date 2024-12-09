
# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# IMPORTS
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


class DiceLoss(_Loss):
    """
    Dice Loss
    """

    def forward(self, output, target, weights=None, ignore_index=None):
        """
        :param output: N x C x H x W Variable
        :param target: N x H x W LongTensor with starting class at 0
        :param weights: C FloatTensor with class wise weights
        :param int ignore_index: ignore label with index x in the loss calculation
        :return:
        """
        eps = 0.001

        encoded_target = output.detach() * 0

        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0

        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        intersection = output * encoded_target
        # First dimensions H x W x D, then batch = dim 0, just leave channel = dim 1
        numerator = 2 * intersection.sum(tuple(range(2, intersection.ndim))).sum(0)
        union = output + encoded_target

        if ignore_index is not None:
            union[mask] = 0

        # First dimensions H x W x D, then batch = dim 0, just leave channel = dim 1
        denominator = union.sum(tuple(range(2, union.ndim))).sum(0) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))  # Channel-wise weights

        return loss_per_channel.sum() / output.size(1)


class DiceLoss2(_Loss):
    """
    Dice Loss
    """
    def __init__(self, **kwargs):
        #self.idc = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, output, target):
        """
        :param Tensor output: sN x C x H x W Variable
        :param Tensor target: N x H x W LongTensor with starting class at 0
        :return Tensor:
        """
        # Make target one hot
        encoded_target = output.detach() * 0
        encoded_target.scatter_(1, target.unsqueeze(1), 1)

        # Select indices of interest (can be used for masking)
        output = output[:, 1:, ...].type(torch.float32)
        encoded_target = encoded_target[:, 1:, ...].type(torch.float32)

        # Calculate DSC Loss (1-2*intersection/union)
        intersection = torch.einsum("bcwh, bcwh->bc", output, encoded_target)
        union = (torch.einsum("bkwh->bk", output) + torch.einsum("bkhw->bk", encoded_target))

        loss_per_channel = torch.ones_like(intersection) - (2 * intersection + 1e10) / (union + 1e10)
        loss = loss_per_channel.mean()
        # Return loss averaged over channels
        return loss


class CrossEntropy2D(nn.Module):
    """
    2D Cross-entropy loss implemented as negative log likelihood
    """

    def __init__(self, weight=None, reduction='none'):
        super(CrossEntropy2D, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        print(f"Initialized {self.__class__.__name__} with weight: {weight} and reduction: {reduction}")

    def forward(self, inputs, targets):
        return self.nll_loss(inputs, targets)


class CombinedLoss(nn.Module):
    """
    For CrossEntropy the input has to be a long tensor
    Args:
        -- inputx N x C x H x W
        -- target - N x H x W - int type
        -- weight - N x H x W - float
    """

    def __init__(self, weight_dice=1, weight_ce=1):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropy2D()
        self.dice_loss = DiceLoss()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(self, inputx, target, weight):
        # Typecast to long tensor --> labels are bytes initially (uint8), index operations requiere LongTensor in pytorch
        # Due to typecasting above, target needs to be shifted to gpu again --> no done in train.py
        input_soft = F.softmax(inputx, dim=1)  # Along Class Dimension
        dice_val = torch.mean(self.dice_loss(input_soft, target))
        ce_val = torch.mean(torch.mul(self.cross_entropy_loss.forward(inputx, target), weight))
        total_loss = torch.add(torch.mul(dice_val, self.weight_dice), torch.mul(ce_val, self.weight_ce))

        return total_loss, dice_val, ce_val


class CombinedLossNoWeight(nn.Module):
    """
    For CrossEntropy the input has to be a long tensor
    Args:
        -- inputx N x C x H x W
        -- target - N x H x W - int type
        -- weight - N x H x W - float
    """

    def __init__(self, weight_dice=1, weight_ce=1):
        super(CombinedLossNoWeight, self).__init__()
        self.cross_entropy_loss = CrossEntropy2D()
        self.dice_loss = DiceLoss()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(self, inputx, target, weight):
        # Typecast to long tensor --> labels are bytes initially (uint8), index operations requiere LongTensor in pytorch
        target = target.type(torch.LongTensor)
        # Due to typecasting above, target needs to be shifted to gpu again
        if inputx.is_cuda:
            target = target.cuda()

        input_soft = F.softmax(inputx, dim=1)  # Along Class Dimension
        dice_val = torch.mean(self.dice_loss(input_soft, target))
        ce_val = torch.mean(self.cross_entropy_loss.forward(inputx, target))
        total_loss = torch.add(torch.mul(dice_val, self.weight_dice), torch.mul(ce_val, self.weight_ce))

        return total_loss, dice_val, ce_val

class CrossEntropy(nn.Module):
    def __init__(self, **kwargs):
        super(CrossEntropy, self).__init__()
        self.cross_entropy_loss = CrossEntropy2D()
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def forward(self, inputx, target, weight):
        # Typecast to long tensor --> labels are bytes initially (uint8), index operations requiere LongTensor in pytorch
        target = target.type(torch.LongTensor)
        # Due to typecasting above, target needs to be shifted to gpu again
        if inputx.is_cuda:
            target = target.cuda()

        ce_val = torch.mean(torch.mul(self.cross_entropy_loss.forward(inputx, target), weight))

        return ce_val, ce_val, ce_val


class SurfaceLoss():
    """
    Boundary loss implementation from LIVIAETS (github.com/LIVIAETS/boundary-loss/blob/master/losses.py#L76)
    """
    def __init__(self, **kwargs):
        self.idc_prob = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs, dist_maps):
        """
        Surface loss forward pass
        :param Tensor probs: NxCxHxW logits-probabilities (after Softmax)
        :param Tensor dist_maps: NxCxHxW distance maps (one-hot encoded?)
        :return: Tensor
        """
        pc = probs[:, self.idc_prob, ...].type(torch.float32)
        dc = dist_maps[:, self.idc_prob, ...].type(torch.float32)

        multiplied = torch.einsum("bkwh,bkwh->bkwh", pc, dc)
        loss = multiplied.mean()
        return loss


class CombinedLoss2(nn.Module):
    """
    For CrossEntropy the input has to be a long tensor
    Args:
        -- inputx N x C x H x W
        -- target - N x H x W - int type
        -- weight - N x H x W - float
    """

    def __init__(self, weight_dice=1, weight_ce=1, idc=list(range(1, 72))):
        super(CombinedLoss2, self).__init__()
        self.boundary_loss = SurfaceLoss(idc=idc)
        self.dice_loss = DiceLoss2(idc=idc)
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(self, inputx, target, dist_map_label):
        # Typecast to long tensor --> labels are bytes initially (uint8), index operations requiere LongTensor in pytorch
        target = target.type(torch.LongTensor)  # Typecast to long tensor
        # Due to typecasting above, target needs to be shifted to gpu again
        if inputx.is_cuda:
            target = target.cuda()

        input_soft = F.softmax(inputx, dim=1)  # Along Class Dimension
        dice_val = torch.mean(self.dice_loss(input_soft, target))
        boundary_val = self.boundary_loss(input_soft, dist_map_label)
        total_loss = torch.add(torch.mul(dice_val, self.weight_dice), torch.mul(boundary_val, self.weight_ce))

        return total_loss, dice_val, boundary_val


def get_loss_func(cfg):
    if cfg.MODEL.LOSS_FUNC == 'combined':
        return CombinedLoss()
    elif cfg.MODEL.LOSS_FUNC == 'ce':
        return CrossEntropy()
    elif cfg.MODEL.LOSS_FUNC == "combinedNoW":
        return CombinedLossNoWeight()
    elif cfg.MODEL.LOSS_FUNC == 'combined2':
        return CombinedLoss2(idc=cfg.DATA.BOUNDARY_CLASSES)
