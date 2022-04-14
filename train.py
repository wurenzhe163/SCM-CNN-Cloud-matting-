"""
Detail: 修改为两个输出，并加入绘图保存功能，loss函数融合了重构误差
Ref:
Project:my_python_script
Time:2022/3/2 9:46
Author:WRZ
"""
# %%
import Resnet,u2net,DataseatIO,Log_weights
from loss import losses, LearningRate, Ms_ssim
from tqdm import tqdm
from PackageDeepLearn.utils.Visualize import visualize
from PackageDeepLearn.utils import OtherTools, DataIOTrans
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore")
# %% Argse
def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='SCM-CNN,A Cloud Matting and Cloud Removal Method!')
    parser.add_argument('--trainDir', default='data/train_dataset', help='train dataset directory')
    parser.add_argument('--testDir', default='data/test_dataset', help='test dataset directory')
    parser.add_argument('--saveDir', default='./modelSave', help='model save dir')
    parser.add_argument('--ckpt', default=False, help='Ckpt path')
    parser.add_argument('--nEpochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--startEpoch', type=int, default=1, help='start epoch to train')
    parser.add_argument('--save_epoch', type=int, default=20, help='number of epochs to save model')
    parser.add_argument('--nThreads', type=int, default=4, help='number of threads for data loading')
    parser.add_argument('--batchSize', type=int, default=4, help='input batch size for train')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lrDecay', type=int, default=5,  help='learning rate')
    parser.add_argument('--lrdecayType', default='poly')
    parser.add_argument('--train_phase', default= 'U2NET', help='train phase U2NET or RESNET50')
    parser.add_argument("--port", default=52162)
    args = parser.parse_args()
    print(args)
    return args

# %%Loss function
def loss_function(trainimg, origin_img, alpha_gt, alpha_pre, cloudMaxDN, cloudMaxDN_pre):
    loss = 0
    M = Ms_ssim.MS_SSIM(data_range=1,
                        size_average=True,
                        win_size=11,
                        win_sigma=1.5,
                        channel=3,
                        spatial_dims=2,
                        weights=None,
                        K=(0.01, 0.03))

    if type(alpha_pre) == list:
        for each, DN in zip(alpha_pre, cloudMaxDN_pre):
            L_alpha = losses.MSE(alpha_gt, each)
            L_DN = losses.MSE(cloudMaxDN, DN)
            ComposeImg = each * \
                DN.reshape(len(each), 1, 1, 1) + origin_img * (1 - each)
            Loss_compose = 1 - M(trainimg, ComposeImg)
#            loss += (0.5*L_alpha + 2.0*L_DN + 5*Loss_compose)
            loss += (3 * L_alpha + 3 * L_DN + 4 * Loss_compose)
    else:
        L_alpha = losses.MSE(alpha_gt, alpha_pre)
        L_DN = losses.MSE(cloudMaxDN, cloudMaxDN_pre)
        ComposeImg = alpha_pre * \
            cloudMaxDN_pre.reshape(len(alpha_pre), 1, 1, 1) + \
            origin_img * (1 - alpha_pre)
        Loss_compose = 1 - M(trainimg, ComposeImg)

        loss = 0.5 * L_alpha + 2.0 * L_DN + Loss_compose * 5

    return loss, L_alpha, 2.0 * L_DN, Loss_compose * 5

# %%Train_model
class trainModel(object):
    def __init__(self, train_phase, lr,
                 train_dir, test_dir, saveDir, batch,
                 nThreads, lrdecayType, start_epoch, train_epoch,
                 save_epoch, finetuning=False):
        print("============> Environment init")
        self.train_phase = train_phase
        self.lr = lr
        self.dir = train_dir
        self.test_dir = test_dir
        self.saveDir = saveDir
        self.batch = batch
        self.nThreads = nThreads
        self.start_epoch = start_epoch
        self.train_epoch = train_epoch
        self.save_epoch = save_epoch
        self.lrdecayType = lrdecayType

        self.device = OtherTools.DEVICE_SLECT()

        print("============> Building model ...")
        # train_phase

        if train_phase == 'U2NET':
            self.model = u2net.U2NET_2Out()
        if train_phase == 'RESNET50':
            self.model = Resnet.Resnet50_rebuild()
            for i in self.model.classifer_out[0].parameters():
                i.requires_grad = False

        self.model.to(self.device)

        self.train_data = getattr(DataseatIO, 'AlphaCloudDataset')(dir=self.dir,
                                                                    Numclass=3,
                                                                    augmentation=DataIOTrans.DataTrans.data_augmentation(
                                                                        ToTensor=True))


        self.trainloader = DataLoader(self.train_data,
                                      batch_size=self.batch,
                                      drop_last=True,
                                      shuffle=True,
                                      num_workers=self.nThreads,
                                      pin_memory=True)

        self.test_data = getattr(DataseatIO, 'AlphaCloudDataset')(dir=self.test_dir,
                                                                   Numclass=3,
                                                                   augmentation=DataIOTrans.DataTrans.data_augmentation(
                                                                       ToTensor=True))
        self.testloader = DataLoader(self.test_data)

        self.trainlog = Log_weights.Train_Log(self.saveDir)

        if finetuning:
            self.start_epoch, self.model = self.trainlog.load_model(self.model,
                                                                    lastest_out_path=finetuning)
            self.start_epoch = self.start_epoch + 1
            self.train_epoch = self.train_epoch + self.start_epoch + 1

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    def train(self, epoch, tqdm_leaveLen):
        # 切换模型为训练模式
        self.model.train()
        if self.lrdecayType != 'keep':
            self.lr = LearningRate.set_lr(self.lrdecayType, self.lr, self.lrDecay, epoch, 220,
                                          self.optimizer)
        loss_ = 0
        L_alpha_ = 0
        L_DN_ = 0
        Loss_compose_ = 0

        pbar = tqdm(self.trainloader, leave=bool(epoch == tqdm_leaveLen))
        for i, [sample_batched, AlphaMAXDN] in enumerate(pbar):

            self.origin_img, self.img, self.trimap_gt, self.alpha_gt = \
                sample_batched['original_image'].to(self.device), \
                sample_batched['train_image'].to(self.device), \
                sample_batched['OnehotTrimap'].to(self.device), \
                sample_batched['mattinglabel'].to(self.device)

            self.cloudMaxDN = AlphaMAXDN.to(self.device)
            # ----------------------------pre-----------------------------
            self.alpha_pre, self.cloudMaxDN_pre = self.model(self.img)
            # ------------------------------------------------------------
            loss, L_alpha, L_DN, Loss_compose = loss_function(
                self.img, self.origin_img, self.alpha_gt, self.alpha_pre, self.cloudMaxDN, self.cloudMaxDN_pre)

            pbar.set_description(
                'Epoch : %d/%d, Iter : %d/%d,lr : %.6f  Loss: %.4f , L_alpha: %.4f , L_DN:%.4f,Loss_compose:%.4f' %
                (epoch, self.train_epoch, i + 1, len(self.train_data) // self.batch,
                 self.lr, loss.data.item(), L_alpha.data.item(
                ), L_DN.data.item(), Loss_compose.data.item()
                ))

            if (i + 1) % 1000 == 0:
                if type(self.alpha_pre) == list:
                    alpha_pre = torch.squeeze(
                        self.alpha_pre[0][0, :, :, :].detach().cpu())
                else:
                    alpha_pre = torch.squeeze(
                        self.alpha_pre[0, :, :, :].detach().cpu())

                visualize(savepath=self.saveDir + f'/train/epoch{epoch:03d}' + f'_iter{i:04d}',
                          original_img=self.origin_img[0, :, :, :].cpu().permute(
                              1, 2, 0),
                          train_img=self.img[0, :, :,
                                             :].cpu().permute(1, 2, 0),
                          trimap_gt=DataIOTrans.DataTrans.OneHotDecode(
                              self.trimap_gt[0, :, :, :].permute(1, 2, 0).cpu()),
                          alpha_gt=torch.squeeze(
                              self.alpha_gt[0, :, :, :].detach().cpu()),
                          alpha_pre=alpha_pre
                          )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # 将损失拆成数字表达
            loss_ += loss.item()
            L_alpha_ += L_alpha.item()
            L_DN_ += L_DN.item()
            Loss_compose_ += Loss_compose.item()

        loss_ = loss_ / (i + 1)
        L_alpha_ = L_alpha_ / (i + 1)
        L_DN_ = L_DN_ / (i + 1)
        Loss_compose_ = Loss_compose_ / (i + 1)

        log = pd.DataFrame({"epoch": epoch,
                            "train_epoch": self.train_epoch,
                            'tain_loss': loss_,
                            "lr": self.lr,
                            "train_L_alpha": L_alpha_,
                            "train_L_DN": L_DN_,
                            "Loss_compose_": Loss_compose_,
                            }, index=[epoch])
        self.log = log
        self.trainlog.save_log(self.log)

        return loss_
    # -----------------------------------------------------------------------------------
    # ==================================test=============================================

    def validate(self, epoch, tqdm_leaveLen):

        self.model.eval()
        testLossALL, testL_alphaALL, testL_DNALL = [], [], []

        with torch.no_grad():
            for ll, [test_batched, TestAlphaMAXDN] in enumerate(self.testloader):
                self.testorigin_img, self.testimg, self.testtrimap_gt, self.testalpha_gt = \
                    test_batched['original_image'].to(self.device), \
                    test_batched['train_image'].to(self.device), \
                    test_batched['OnehotTrimap'].to(self.device), \
                    test_batched['mattinglabel'].to(self.device)

                self.TestAlphaMAXDN = TestAlphaMAXDN.to(self.device)
                # ----------------------------pre-----------------------------
                self.testalpha_pre, self.TestAlphaMAXDN_pre = self.model(
                    self.testimg)
                # ------------------------------------------------------------
                testloss, testL_alpha, testL_DN, _ = loss_function(
                    self.testimg, self.testorigin_img, self.testalpha_gt, self.testalpha_pre, self.TestAlphaMAXDN, self.TestAlphaMAXDN_pre)

                self.testloss = testloss
                testLossALL.append(testloss.item())
                testL_alphaALL.append(testL_alpha.item())
                testL_DNALL.append(testL_DN.item())
            if type(self.testalpha_pre) == list:
                testalpha_pre = torch.squeeze(
                    self.testalpha_pre[0][0, :, :, :].detach().cpu())
            else:
                testalpha_pre = torch.squeeze(
                    self.testalpha_pre[0, :, :, :].detach().cpu())
            visualize(savepath=self.saveDir + f'/val/epoch{epoch:03d}' + f'_iter{ll:04d}.png',
                      testoriginal_img=self.testorigin_img[0, :, :, :].cpu().permute(
                          1, 2, 0),
                      testtrain_img=self.testimg[0, :,
                                                 :, :].cpu().permute(1, 2, 0),
                      testtrimap_gt=DataIOTrans.DataTrans.OneHotDecode(
                          self.testtrimap_gt[0, :, :, :].permute(1, 2, 0).cpu()),
                      testalpha_gt=torch.squeeze(
                          self.testalpha_gt[0, :, :, :].detach().cpu()),
                      testalpha_pre=testalpha_pre, )

            tqdm.write('Epoch : %d/%d, lr : %.6f  TestLoss: %.4f ,Test_alpha: %.4f ,Test_DN: %.4f' % (
                epoch, self.train_epoch, self.lr, np.mean(testLossALL), np.mean(testL_alphaALL), np.mean(testL_DNALL)))

            log = pd.DataFrame({
                'test_loss': np.mean(testLossALL),
                "test_L_alpha": np.mean(testL_alphaALL),
                "test_L_DN": np.mean(testL_DNALL)
            }, index=[epoch])
            self.testlog = log
            self.trainlog.save_log(self.testlog, logname='/logtest.csv')
        return np.mean(testLossALL)

    def execute(self, lrDecay):
        inint_train_loss, inint_val_loss = 10, 10
        self.lrDecay = lrDecay
        self.tqdm_leaveLen = self.train_epoch
        for epoch in tqdm(range(self.start_epoch, self.train_epoch + 1)):
            train_loss = self.train(epoch, self.tqdm_leaveLen)
            val_loss = self.validate(epoch, self.tqdm_leaveLen)

            if train_loss < inint_train_loss:
                inint_train_loss = train_loss
                self.trainlog.save_model(
                    self.model, epoch, self.lr, train_loss=True, val_loss=False)
            if val_loss < inint_val_loss:
                inint_val_loss = val_loss
                self.trainlog.save_model(
                    self.model, epoch, self.lr, train_loss=False, val_loss=True)

            # 固定
            if epoch % self.save_epoch == 0:
                self.trainlog.save_model(
                    self.model, epoch, self.lr, train_loss=False, val_loss=False)




# %%
if __name__ == '__main__':

    args = get_args()
    train_dir = args.trainDir
    test_dir = args.testDir
    saveDir = args.saveDir
    ckpt = args.ckpt
    train_epoch = args.nEpochs
    start_epoch = args.startEpoch
    save_epoch = args.save_epoch
    batch = args.batchSize
    nThreads = args.nThreads
    lr = args.lr
    lrDecay = args.lrDecay
    lrdecayType = args.lrdecayType
    train_phase = args.train_phase  # 'U2NET'  'RESNET50'

    Model = trainModel(train_phase, lr, train_dir, test_dir,
                       saveDir,batch, nThreads, lrdecayType,
                       start_epoch, train_epoch, save_epoch,
                       finetuning=ckpt)

    print('train_datalen={},test_datalen={}'.format(
        Model.train_data.__len__(), Model.test_data.__len__()))

    Model.execute(lrDecay=lrDecay)
