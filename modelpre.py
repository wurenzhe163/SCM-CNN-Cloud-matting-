import torch
import os,cv2
import numpy as np
from PackageDeepLearn.utils import OtherTools,DataIOTrans,Visualize
from PackageDeepLearn import ImageAfterTreatment
import u2net,Resnet
from tqdm import tqdm
import copy
import argparse

# %%Argparse
def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='SCM-CNN,A Cloud Matting and Cloud Removal Method!')
    parser.add_argument('--preDir', default=False, help='train dataset directory')
    parser.add_argument('--ckpt', default=False, help='Ckpt path')
    parser.add_argument('--saveDir', default='./Pre', help='model pre save dir')
    parser.add_argument('--pre_phase', default='U2NET', help='pre phase U2NET or RESNET50')
    parser.add_argument('--Image_path', default=False, help='full image')
    parser.add_argument('--kernel', type=int, default=[512, 512], nargs='+', metavar='', help='Kernel Size')
    parser.add_argument('--stride', type=int, default=512, metavar='', help='Stride')
    parser.add_argument("--port", default=52162)
    args = parser.parse_args()
    print(args)
    return args

# %% Model
class PreModel(object):
    def __init__(self,lastest_out_path,saveDir,pre_phase='Just_alpha'):
        '''
        mnetModel : 'parallel' or 'cascade'
        pre_phase: U2NET or RESNET50
        '''
        self.saveDir = saveDir
        self.pre_phase = pre_phase
        self.device = OtherTools.DEVICE_SLECT()

        print("============> Building model ...")
        # build model
        if self.pre_phase == 'U2NET':
            self.model = u2net.U2NET_2Out().to(self.device)
        if self.pre_phase == 'RESNET50':
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
                if self.pre_phase=='U2NET':
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
                if self.pre_phase == 'RESNET50':
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
                img_decode = torch.squeeze(img_).permute(1, 2, 0).detach().numpy()
                MAXDN = self.alpha_pre[1][0].detach().numpy()[0][0]
                Dehaze = (img_decode - alpha_pre_decode[..., np.newaxis] * MAXDN) / (1 - alpha_pre_decode[..., np.newaxis])
                Visualize.visualize(img_decoed=img_decode,alpha_pre_decode = alpha_pre_decode,Deahze=Dehaze)
                outList.append(alpha_pre_decode)

            img = copy.deepcopy(outList)
            outPut =Img_Post.join_image2(img = img, kernel=self.kernel, stride=self.stride, H=H, W=W, S=Shape[-1])
            savename = 'Pre' + Image_path.split('\\')[-1]
            cv2.imwrite(self.saveDir + '\\' + savename, outPut[0:Shape[0],0:Shape[1],:])

        else:
            print('Input Wrong')



if __name__ == '__main__':

    # 相对路径
    args = get_args()
    pre_img_dir = args.preDir
    ckpt = args.ckpt
    output = args.saveDir
    pre_phase = args.pre_phase
    Image_path = args.Image_path
    kernel = args.kernel
    stride = args.stride
    if pre_img_dir:
        assert Image_path == False, 'make Image_path=False'
        Model = PreModel(lastest_out_path=ckpt,
                         saveDir=output,pre_phase='Resnet_bright')(pre_img_dir = pre_img_dir)
    if Image_path:
        assert pre_img_dir==False ,'make pre_img_dir=False'
        Model = PreModel(lastest_out_path=ckpt,
                         saveDir=output, pre_phase='Alpha_bright')(pre_img=Image_path)




