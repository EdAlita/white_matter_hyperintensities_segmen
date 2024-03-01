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
import yacs.config

from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from numbers import Real
from typing import Optional, Tuple, Union

class DiceLoss(_Loss):
    """
    Calculate Dice Loss.

    Methods
    -------
    forward
        Calulate the DiceLoss.
    """

    def forward(
        self,
        output: Tensor,
        target: Tensor
    ) -> torch.Tensor:
        """
        Calulate the DiceLoss.

        Parameters
        ----------
        output : Tensor
            N x C x H x W Variable.
        target : Tensor
            N x C x W LongTensor with starting class at 0.
        weights : int, optional
            C FloatTensor with class wise weights(Default value = None).
        ignore_index : int, optional
            Ignore label with index x in the loss calculation (Default value = None).

        Returns
        -------
        torch.Tensor
            Calculated Diceloss.
        """
        eps = 0.001

        encoded_target = target.unsqueeze(1)


        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + encoded_target

        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = 1 - (numerator / denominator)

        return loss_per_channel.sum() / output.size(1)


class CrossEntropy2D(nn.Module):
    """
    2D Cross-entropy loss implemented as negative log likelihood.

    Attributes
    ----------
    nll_loss
        Calculated cross-entropy loss.

    Methods
    -------
    forward
        Returns calculated cross entropy.
    """

    def __init__(self, weight: Optional[Tensor] = None, reduction: str = "none"):
        """
        Construct CrossEntropy2D object.

        Parameters
        ----------
        weight : Tensor, optional
            A manual rescaling weight given to each class. If given, has to be a Tensor of size `C`. Defaults to None.
        reduction : str
            Specifies the reduction to apply to the output, as in nn.CrossEntropyLoss. Defaults to 'None'.
        """
        super(CrossEntropy2D, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        print(
            f"Initialized {self.__class__.__name__} with weight: {weight} and reduction: {reduction}"
        )

    def forward(self, inputs, targets):
        """
        Feedforward.
        """
        return self.nll_loss(inputs, targets)


class CombinedLoss(nn.Module):
    """
    For CrossEntropy the input has to be a long tensor.

    Attributes
    ----------
    cross_entropy_loss
        Results of cross entropy loss.
    dice_loss
        Results of dice loss.
    weight_dice
        Weight for dice loss.
    weight_ce
        Weight for float.
    """

    def __init__(self, weight_dice: Real = 1, weight_ce: Real = 1):
        """
        Construct CobinedLoss object.

        Parameters
        ----------
        weight_dice : Real
            Weight for dice loss. Defaults to 1.
        weight_ce : Real
            Weight for cross entropy loss. Defaults to 1.
        """
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropy2D()
        self.dice_loss = DiceLoss()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(
        self, inputx: Tensor, target: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        [MISSING].

        Parameters
        ----------
        inputx : Tensor
            A Tensor of shape N x C x H x W containing  the input x values.
        target : Tensor
            A Tensor of shape N x H x W of integers containing the target.
        weight : Tensor
            A Tensor of shape N x H x W of floats containg the weights.

        Returns
        -------
        Tensor
            Total loss.
        Tensor
            Dice loss.
        Tensor
            Cross entropy value.
        """
        # Typecast to long tensor --> labels are bytes initially (uint8),
        # index operations require LongTensor in pytorch
        target = target.type(torch.LongTensor)
        # Due to typecasting above, target needs to be shifted to gpu again
        if inputx.is_cuda:
            target = target.cuda()

        input_soft = F.softmax(inputx, dim=1)  # Along Class Dimension
        dice_val = torch.mean(self.dice_loss(input_soft, target))
        ce_val = torch.mean(self.cross_entropy_loss.forward(inputx, target))
        total_loss = torch.add(dice_val,ce_val)

        return total_loss, dice_val, ce_val


def get_loss_func(
    cfg: yacs.config.CfgNode,
) -> Union[CombinedLoss, CrossEntropy2D, DiceLoss]:
    """
    Give a default object of the loss function.

    Parameters
    ----------
    cfg : yacs.config.CfgNode
        Configuration node, containing searched loss function.
        The model loss function can either be 'combined', 'ce' or 'dice'.

    Returns
    -------
    CombinedLoss
        Total loss.
    CrossEntropy2D
        Cross entropy value.
    DiceLoss
        Dice loss.

    Raises
    ------
    NotImplementedError
        Requested loss function is not implemented.
    """
    if cfg.MODEL.LOSS_FUNC == "combined":
        return CombinedLoss()
    elif cfg.MODEL.LOSS_FUNC == "ce":
        return CrossEntropy2D()
    elif cfg.MODEL.LOSS_FUNC == "dice":
        return DiceLoss()
    else:
        raise NotImplementedError(
            f"{cfg.MODEL.LOSS_FUNC}" f" loss function is not supported"
        )