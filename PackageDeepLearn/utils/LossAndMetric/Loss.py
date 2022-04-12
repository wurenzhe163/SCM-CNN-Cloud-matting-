import torch.nn as nn
import torch.nn.functional as F
import torch

def MSE(gt,pre,eps=0.0001):

    return torch.sqrt(torch.pow(pre - gt, 2) + eps).mean()
    #均方误差

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def calc_loss(prediction, target, bce_weight=0.5):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    prediction = F.sigmoid(prediction)
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss


class TverskyLoss(nn.Module):
    '''
    二分类，适用于Onehot数据
    '''
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.2, beta=0.8,softmax=False):

        #comment out if your model contains a sigmoid or equivalent activation layer
        ###-----------------------------------------------------###
        ###  如果有了激活函数这里如果加入Softmax会导致loss函数指向错误   ###
        ###-----------------------------------------------------###
        if softmax:
            inputs = torch.softmax(inputs,dim=1)

        #flatten label and prediction tensors
        inputsN = inputs[:,0,:,:].reshape(-1)
        targetsN = targets[:,0,:,:].reshape(-1)
        inputsP = inputs[:, 1,:,:].reshape(-1)
        targetsP = targets[:,1,:,:].reshape(-1)

        #True Positives, False Positives & False Negatives
#         TP = (inputs * targets).sum()
#         FP = ((1-targets) * inputs).sum()
#         FN = (targets * (1-inputs)).sum()
        TP = (inputsP * targetsP ).sum()
        FP = (inputsP * targetsN ).sum()
        FN = (targetsP * inputsN ).sum()

        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)

        return 1 - Tversky


def TverskyLoss_i(gt, pre, alpha=0.2, layerAttention=[2, 3], eps=1e-6, softmax=False, argmax=False):
    """

    Args:
        gt:  Ground - truth
        pre: Prediction
        alpha: prediction of Alpha-image
        layerAttention:  select one channel to attention
        eps:  in case the denominator is zero

    Returns:
        TverskyLoss of channel
    """

    shape = gt.shape
    # gt = torch.split(gt, 1, 1)
    # pre = torch.split(pre, 1, 1)
    if softmax:
        pre = torch.nn.Softmax(dim=1)(pre)
    if argmax:
        pre = torch.argmax(pre, )

    layerNotAttention = lambda shape, layer: [i for i in range(shape[1]) if i != layer]
    fp, fn = 0, 0;
    for i in layerAttention:
        fp += (gt[:, layerNotAttention(shape, 2), :, :].sum(axis=1) * pre[:, i, :, :]).sum()
        fn += (gt[:, i, :, :] * pre[:, layerNotAttention(shape, 2), :, :].sum(axis=1)).sum()
    fp = fp / len(layerAttention)
    fn = fn / len(layerAttention)

    tp = torch.Tensor([(gt[:, i, :, :] * pre[:, i, :, :]).sum() for i in layerAttention]).sum()

    # tp = (gt[layerAttention] * pre[layerAttention]).sum()
    # fp = ((gt[0] + gt[1]) * pre[layerAttention]).sum()
    # fn = (gt[layerAttention] * (pre[0] + pre[1])).sum()

    if tp > 0:
        return (1 - (tp + eps) / (tp + alpha * fp + (1 - alpha) * fn + eps)) / shape[0]
    if tp == 0:
        return (fp + fn) / (shape[0] * shape[2] * shape[3])

# 超分损失函数https://www.cnblogs.com/jiangnanyanyuchen/p/11884912.html
class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss