import torch
import os,cv2,sys
sys.path += ['..']
import numpy as np
from PackageDeepLearn.utils import OtherTools,DataIOTrans,Visualize
from PackageDeepLearn import ImageAfterTreatment
import u2net,Resnet
from tqdm import tqdm


class PreModel(object):
    def __init__(self,lastest_out_path,saveDir,pre_phase='Just_alpha'):
        '''
        mnetModel : 'parallel' or 'cascade'
        pre_phase: Just_alpha or Alpha_bright
        '''
        self.saveDir = saveDir
        self.pre_phase = pre_phase
        self.device = OtherTools.DEVICE_SLECT()

        print("============> Building model ...")
        # build model
        if self.pre_phase == 'Just_alpha':
            self.model = u2net.U2NET().to(self.device)
        if self.pre_phase == 'Alpha_bright':
            self.model = u2net.U2NET_2Out().to(self.device)
        if self.pre_phase == 'Resnet_bright':
            self.model = Resnet.Resnet50_rebuild().to(self.device)

        # lode_model
        if self.device.type == 'cpu':
            ckpt = torch.load(lastest_out_path,map_location=lambda storage, loc: storage)
        else:
            ckpt = torch.load(lastest_out_path)

        self.epoch = ckpt['epoch']
        self.lr = ckpt['lr']
        self.model.load_state_dict(ckpt['state_dict'])

        print("=> loaded checkpoint '{}' (epoch {})".format(lastest_out_path, ckpt['epoch']))

    def __call__(self,pre_img_dir=False,pre_img=False,kernel = [],stride = []):
        """

        Args:
            pre_img_dir:
            pre_img: 是否读入整景影像进行处理
            kernel:
            stride:

        Returns:

        """
        self.kernel = kernel
        self.stride = stride
        self.model.eval()
        search_files = lambda path : sorted([os.path.join(path,f) for f in os.listdir(path) if f.endswith(".tif")])
        outList = []
        if pre_img_dir:
            imgs = search_files(pre_img_dir)
            for i,eachimg in enumerate(imgs):
                I = DataIOTrans.DataIO.read_IMG(eachimg).astype(np.float32)
                img = torch.from_numpy(I[np.newaxis,...] / 10000).to(self.device)
                img = img.permute(0,3,1,2)

                # #-----------------------------------------------this
                if self.pre_phase=='Just_alpha':
                    self.alpha_pre = self.model(img)
                    alpha_pre_decode = torch.squeeze(self.alpha_pre[0]).detach().cpu().numpy()

                    Visualize.visualize(
                        savepath=r'D:\train_data\cloud_matting_dataset\test\pre\Just_Alpha\fig' + '\\' + f'alpha{i:04d}.jpg',
                        alpha=alpha_pre_decode,
                        img=torch.squeeze(img).permute(1, 2, 0).detach().numpy())

                    cv2.imwrite(self.saveDir + '\\' + f'alpha{i:04d}.tif', alpha_pre_decode)
                if self.pre_phase=='Alpha_bright':
                    self.alpha_pre , self.cloudDN = self.model(img)
                    a = torch.squeeze(self.alpha_pre[0]).detach().cpu().numpy()
                    F = self.cloudDN[0].detach().cpu().numpy()[0][0]
                    S0 = I - a[...,np.newaxis]*F*10000
                    S0[S0 < 0] = 0
                    S1 =  1-a[...,np.newaxis]
                    S1[S1 <= 0.1] = 0.1
                    B = S0 / S1
                    Visualize.save_img(path=self.saveDir,
                                   index=i,Alpha=a,dehaze_img = B)
                if self.pre_phase == 'Resnet_bright':
                    self.alpha_pre, self.cloudDN = self.model(img)
                    a = torch.squeeze(self.alpha_pre).detach().cpu().numpy()
                    F = self.cloudDN.detach().cpu().numpy()[0][0]
                    S0 = I - a[...,np.newaxis]*F*10000
                    S0[S0 < 0] = 0
                    S1 =  1-a[...,np.newaxis]
                    S1[S1 <= 0.1] = 0.1
                    B = S0 / S1
                    Visualize.save_img(path=self.saveDir,
                                   index=i,Alpha=a,dehaze_img = B)
        elif pre_img:

            Img_Post = ImageAfterTreatment.Img_Post()
            data = Img_Post.read_IMG(pre_img).astype(np.float32)
            Shape = data.shape
            data = Img_Post.expand_image(data, self.stride, self.kernel)
            data_list, H, W = Img_Post.cut_image(data, self.kernel, self.stride)

            for i,img in enumerate(tqdm(data_list, ncols=80)):

                img_ = torch.from_numpy(img[np.newaxis,...] / 10000).to(self.device)
                img_ = img_.permute(0,3,1,2)
                self.alpha_pre = self.model(img_)

                alpha_pre_decode = torch.squeeze(self.alpha_pre[0][0]).detach().cpu().numpy()
                img_decoed = torch.squeeze(img_).permute(1, 2, 0).detach().numpy()
                MAXDN = self.alpha_pre[1][0].detach().numpy()[0][0]
                # Dehaze = (img_decoed - alpha_pre_decode[..., np.newaxis] * MAXDN) / (1 - alpha_pre_decode[..., np.newaxis])
                Visualize.visualize(img_decoed=img_decoed,alpha_pre_decode = alpha_pre_decode,Deahze=Dehaze)
                outList.append(alpha_pre_decode)
            import copy
            img = copy.deepcopy(outList)

            outPut =Img_Post.join_image2(img = img, kernel=self.kernel, stride=self.stride, H=H, W=W, S=Shape[-1])
            savename = 'Pre' + Image_path.split('\\')[-1]
            cv2.imwrite(self.saveDir + '\\' + savename, outPut[0:Shape[0],0:Shape[1],:])

        else:
            print('Input Wrong')



if __name__ == '__main__':
    # 工作路径
    work_dir = r'C:\Users\SAR\Desktop\cloud_matting\work_dir'

    make_dir = DataIOTrans.make_dir

    # 相对路径
    pre_img_dir = r'D:\train_data\cloud_matting_dataset\test\train_image\cloudy_image'
    ckpt = r'D:\train_data\cloud_matting_dataset\test\pre\cloud_matting\Resnet\0040model_obj.pth'
    output = r'D:\train_data\cloud_matting_dataset\test\pre\cloud_matting\Resnet'

    Image_path = r'E:\HLG+GB_Sentinal2\Raw\S2B_MSIL1C_20180510T034539_N0206_R104_T47RQN_20180510T080714.SAFE\GRANULE\L1C_T47RQN_A006136_20180510T035318\IMG_DATA\icecloud.tif'
    Image_path2 = r'E:\HLG+GB_Sentinal2\Raw\S2B_MSIL1C_20180510T034539_N0206_R104_T47RQN_20180510T080714.SAFE\GRANULE\L1C_T47RQN_A006136_20180510T035318\IMG_DATA\noice.tif'
    kernel = [512, 512]
    stride = 512

    Model = PreModel(lastest_out_path=ckpt,
                     saveDir=output,pre_phase='Resnet_bright')(pre_img_dir = pre_img_dir)

    Model = PreModel(lastest_out_path=ckpt,
                     saveDir=output, pre_phase='Alpha_bright')(pre_img=Image_path)



    self=Model
    # 测试
    search_files = lambda path: sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".tif")])
    imgs = search_files(pre_img_dir)
    eachimg = imgs[2]
    img0 = DataIOTrans.DataIO.read_IMG(eachimg).astype(np.float32)
    img0 = torch.from_numpy(img0[np.newaxis, ...] / 10000).to(self.device)
    img0 = img0.permute(0, 3, 1, 2)
    self.alpha_pre = self.model(img0)
    alpha_pre_decode = torch.squeeze(self.alpha_pre[0]).detach().cpu().numpy()
    Visualize.visualize(
        alpha=alpha_pre_decode,
        img=torch.squeeze(img0).permute(1, 2, 0).detach().numpy())

