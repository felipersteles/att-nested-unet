import torch
import torch.nn.functional as F
from torch import nn

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, output, target, use_sigmoid=True):
        if use_sigmoid:
            output = torch.sigmoid(output)

        # Ensure output and target have the same number of channels
        if output.shape[1] != target.shape[1]:
            # Assuming output has 1 channel and target has 3 channels
            output = output.repeat(1, target.shape[1], 1, 1)  # Repeat output channels

        dim0 = output.shape[0]
        output = output.contiguous().view(dim0, -1).float()
        target = target.contiguous().view(dim0, -1).float()

        num = 2 * torch.sum(torch.mul(output, target), dim=1) + self.smooth
        den = torch.sum(output.abs() + target.abs(), dim=1) + self.smooth

        dice = num / den
        loss = 1 - torch.mean(dice)
        return loss

class BinaryFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=3, gamma=2, ignore_index=None, reduction='mean', **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6  # set '1e-4' when train with FP16
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()
        if valid_mask is not None:
            pos_mask = pos_mask * valid_mask
            neg_mask = neg_mask * valid_mask

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)  # / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * neg_weight * F.logsigmoid(-output)  # / (torch.sum(neg_weight) + 1e-4)
        loss = pos_loss + neg_loss
        loss = loss.mean()
        return loss

class BinaryDiceBCELoss(nn.Module):
    def __init__(self, epsilon=1e-6, bce_weight=0.5, dice_weight=0.5):
        super(BinaryDiceBCELoss, self).__init__()
        self.epsilon = epsilon
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss with logits

    def forward(self, outputs, labels):
        # BCE Loss
        # BCEWithLogitsLoss internally applies sigmoid, so we don't need to apply it manually
        bce_loss = self.bce_loss(outputs, labels)

        # Dice Loss
        # Apply sigmoid to the outputs to bring them in the [0, 1] range for Dice loss
        outputs = torch.sigmoid(outputs)

        # Flatten the tensors along the batch and channel dimensions
        outputs = outputs.contiguous().view(outputs.size(0), -1)  # [batch_size, num_pixels]
        labels = labels.contiguous().view(labels.size(0), -1)  # Same for labels

        # Calculate the intersection and sums
        intersection = (outputs * labels).sum(dim=1)  # Sum over pixels for each image in batch
        output_sum = outputs.sum(dim=1)
        label_sum = labels.sum(dim=1)

        # Compute Dice score for each image in batch
        dice_score = (2. * intersection + self.epsilon) / (output_sum + label_sum + self.epsilon)

        # Dice loss is 1 - Dice score (averaged over the batch)
        dice_loss = 1 - dice_score.mean()

        # Combined loss (weighted sum of BCE and Dice loss)
        combined_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        return combined_loss