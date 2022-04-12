
import torch,math
import numpy as np
__all__ = ['SegmentationMetric']

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反,需要输入是连续值
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    """
    输入的标签和预测，必须为连续值，从0开始
    """
    def __init__(self, numClass):
        '''
        numClass : 分类数
        '''
        self.numClass = numClass
        self.confusionMatrix = torch.zeros((self.numClass,) * 2)  # 混淆矩阵（空）

    def pixelAccuracy(self):
        # return all class overall pixel accuracy 正确的像素占总像素的比例
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # precision = (TP) / TP + FP
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        """
        Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return:
        """
        classAcc = self.classPixelAccuracy()
        meanAcc = classAcc[classAcc < float('inf')].mean() # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = torch.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = torch.sum(self.confusionMatrix, axis=1) + torch.sum(self.confusionMatrix, axis=0) - torch.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def meanIntersectionOverUnion(self):
        IoU = self.IntersectionOverUnion()
        mIoU = IoU[IoU<float('inf')].mean()# 求各类别IoU的平均
        return mIoU

    def Recall(self):
        recall = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        return recall

    def genConfusionMatrix(self, imgPredict, imgLabel, ignore_labels):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :param ignore_labels: list as [x1,x2,x3]
        :return: 混淆矩阵
        """
        # 筛选指定numClass的，并且去除ignore_labels
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        for IgLabel in ignore_labels:
            mask &= (imgLabel != IgLabel)

        label = self.numClass * imgLabel[mask] + imgPredict[mask]   # *numClass确保每个值相加都是唯一值
        count = torch.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.view(self.numClass, self.numClass)
        # print(confusionMatrix)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        FWIoU，频权交并比:为MIoU的一种提升，这种方法根据每个类出现的频率为其设置权重。
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = torch.sum(self.confusion_matrix, axis=1) / torch.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                torch.sum(self.confusion_matrix, axis=1) + torch.sum(self.confusion_matrix, axis=0) -
                torch.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel, ignore_labels):
        """
        :param imgPredict:
        :param imgLabel:
        :param ignore_labels: list as [x1,x2,x3]
        :return: 混淆矩阵
        """
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel, ignore_labels)  # 得到混淆矩阵
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass))

# 测试内容
if __name__ == '__main__':
    # 注意，使用该方法需要值连续
    imgPredict = torch.tensor([[0,1,2],[2,1,1]]).long()  # 可直接换成预测图片
    imgLabel = torch.tensor([[0,1,255],[1,1,2]]).long() # 可直接换成标注图片
    ignore_labels = [255]
    metric = SegmentationMetric(3) # 3表示有3个分类，有几个分类就填几, 0也是1个分类
    hist = metric.addBatch(imgPredict, imgLabel,ignore_labels)
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    IoU = metric.IntersectionOverUnion()
    mIoU = metric.meanIntersectionOverUnion()
    print('hist is :\n', hist)
    print('PA is : %f' % pa)
    print('cPA is :', cpa)  # 列表
    print('mPA is : %f' % mpa)
    print('IoU is : ', IoU)
    print('mIoU is : ', mIoU)

##output
# hist is :
# tensor([[1., 0., 0.],
#        [0., 2., 1.],
#        [0., 1., 0.]])
# PA is : 0.600000
# cPA is : tensor([1.0000, 0.6667, 0.0000])
# mPA is : 0.555556
# IoU is :  tensor([1.0000, 0.5000, 0.0000])
# mIoU is :  tensor(0.5000)


def MSE(pre,gt):

    return np.power(pre - gt, 2).mean()

def RMS(pre,gt):
    return np.sqrt(np.power(pre - gt, 2).mean())

def PSNR(pre,gt,pixel_max=10000):
    mse = MSE(pre,gt)
    if mse < 1.0e-10:
        return 100
    else:
        return 10 * math.log10(pixel_max ** 2 / mse)

def SSIM(pre,gt,data_range=255,channel=3):
    import PackageDeepLearn.utils.LossAndMetric.Ms_ssim as Ms_ssim
    SSIM = Ms_ssim.SSIM(
        data_range=data_range,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=channel,
        spatial_dims=2,
        K=(0.01, 0.03),
        nonnegative_ssim=False,)
    return SSIM(pre,gt)

def MSSSIM(pre,gt,data_range=255,channel=3):
    import PackageDeepLearn.utils.LossAndMetric.Ms_ssim as Ms_ssim
    MS_SSIM = Ms_ssim.MS_SSIM(
        data_range=data_range,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=channel,
        spatial_dims=2,
        weights=None,
        K=(0.01, 0.03),)
    return MS_SSIM(pre, gt)


