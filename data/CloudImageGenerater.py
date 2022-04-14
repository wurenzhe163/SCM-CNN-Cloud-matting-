import numpy as np
import cv2,os
import pandas as pd
from osgeo import gdal
import save_img_arr_csv as sc

def make_dir(path):
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        os.makedirs(path)
        print(path + 'Successful folder creation')
        return path
    else:
        return path

def method_gather():
    # 搜索文件
    search_files = lambda path: sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".tif")])
    # 合成有阴影的图像
    generate_cloudshadow_image = lambda image,CloudAlpha,Cloudshadow,CloudMaxDN: \
                                CloudAlpha*CloudMaxDN + (1-CloudAlpha)*Cloudshadow*image
    # 合成无阴影的图像
    generate_image = lambda image,CloudAlpha,CloudMaxDN:\
                                CloudAlpha*CloudMaxDN + (1-CloudAlpha)*image
    return search_files,generate_cloudshadow_image,generate_image

def readtif_GDAL(path, dtype=float):

    dataset = gdal.Open(path)
    nXSize = dataset.RasterXSize
    nYSize = dataset.RasterYSize
    bands = dataset.RasterCount

    data = np.zeros([nYSize, nXSize, bands], dtype=dtype)
    for i in range(bands):
        band = dataset.GetRasterBand(i + 1)
        data[:, :, i] = band.ReadAsArray(0, 0, nXSize, nYSize)
    return data

class Generate_label(object):

    def __init__(self,path):
        self.B2_alpha = (cv2.imread(path[0],cv2.IMREAD_ANYDEPTH)/ 65535.).round(2)
        self.Cloudshadow = cv2.imread(path[1],cv2.IMREAD_ANYDEPTH)
        self.trimap = cv2.imread(path[2],cv2.IMREAD_ANYDEPTH)

    def random(self):
        # 随机裁剪图片,这里随机裁剪的是透明度信息alpha
        self.random_crop()
        # 反转
        self.random_flip()
        #
        self.random_DN()

    def random_crop(self,randseed=(4,5)):
        '''
        Randomly cropping the image, randseed=[1,5] then one of the three possible fetches here 128*128, 256*256, 512*512
        Note that this code can only handle two-dimensional images
        '''
        # 确定裁剪图像大小
        image = self.B2_alpha
        CloudSize= np.random.randint(low=randseed[0], high=randseed[1]) * 128
        image_shape = np.shape(image)
        assert len(np.shape(image)) == 2

        while True:
            # 取左下角点
            w = np.random.randint(low=0,high=image_shape[0]-CloudSize)
            h = np.random.randint(low=0,high=image_shape[1]-CloudSize)
            self.Alpha_crop = image[w:w + CloudSize,h:h + CloudSize]

            if np.sum(self.Alpha_crop==0) / CloudSize ** 2 <= 0.85:
                self.Cloudshadow_crop = self.Cloudshadow[w:w + CloudSize, h:h + CloudSize]
                self.Trimap_crop = self.trimap[w:w + CloudSize, h:h + CloudSize]
                print('Clip Alpha/Cloudshadow/Trimap，crop size: w={},h={}'.format(w, h))
                break
            else:
                continue

    def random_flip(self):
        '''
        Random Inverted Pictures, Single img
        '''
        img = self.Alpha_crop
        axis = np.random.randint(low=-1, high=3)
        if axis != 2:
            self.Alpha_flip = cv2.flip(img, axis)
            self.Cloudshadow_flip = cv2.flip(self.Cloudshadow_crop, axis)
            self.Trimap_flip = cv2.flip(self.Trimap_crop, axis)
            print('Rotational transformation, axis={}'.format(axis))
        else:
            self.Alpha_flip = self.Alpha_crop
            self.Cloudshadow_flip = self.Cloudshadow_crop
            self.Trimap_flip = self.Trimap_crop
            print('Keep')

    def random_DN(self):
        self.CloudMaxDN = np.random.randint(low=4500, high=6000)

if __name__ == '__main__':
    search_files, generate_cloudshadow_image, generate_image = method_gather()
    path = [1,2,3]
    path[0] = r'D:\train_data\cloud_matting_dataset\cloud_clip.tif'
    path[1] = r'D:\train_data\cloud_matting_dataset\cloudshadow.tif'
    path[2] = r'D:\train_data\cloud_matting_dataset\trimap_0.8.tif'

    Savepath = [1,2,3,4,5,6]
    Savepath[0] = r'D:\train_data\cloud_matting_dataset\train\train_image'
    Savepath[1] = r'D:\train_data\cloud_matting_dataset\train\mattingLabel'
    Savepath[2] = r'D:\train_data\cloud_matting_dataset\train\trimap'
    Savepath[3] = r'D:\train_data\cloud_matting_dataset\train\shadow'
    Savepath[4] = r'D:\train_data\cloud_matting_dataset\train\npz'
    Savepath[5] = r'D:\train_data\cloud_matting_dataset\train\Information.csv'
    list(map(make_dir,Savepath[0:5]))

    image_path =r'D:\train_data\cloud_matting_dataset\train\after_cut'
    imgs = search_files(image_path)
    Generate_label = Generate_label(path)

    mark = 'Without_cloudshadow'
    DF = pd.DataFrame()

    for i,each in enumerate(imgs):
        # each_image = cv2.imread(each, cv2.IMREAD_ANYDEPTH)  # 读取训练图像
        each_image = readtif_GDAL(each)
        Generate_label.random()

        Alpha,Cloudshadow,Trimap,CloudMaxDN = Generate_label.Alpha_flip,Generate_label.Cloudshadow_flip,\
        Generate_label.Trimap_flip,Generate_label.CloudMaxDN,
        # 提出全纯前景图像
        if np.sum(Trimap !=  255) == 0:
            continue
        if len(each_image.shape)==3:
            Alpha,Cloudshadow = Alpha[:, :, None],Cloudshadow[:,:,None]
        if mark == 'Without_cloudshadow':
            Gimage = generate_image(each_image,Alpha,CloudMaxDN)
        if mark == 'With_cloudshadow':
            Gimage = generate_image(each_image, Alpha,Cloudshadow, CloudMaxDN)
            cv2.imwrite(Savepath[3] + f'Cloudshadow{i+7000:04d}.tif', Cloudshadow)
        Gimage = Gimage.astype(np.float32)
        Alpha = np.squeeze(Alpha).astype(np.float32)
        # cv2.imwrite(Savepath[0] +'\\'+ f'train{i:04d}.tif', Gimage.astype(np.float32))
        sc.save_tif(Gimage,Savepath[0] ,f'train{i+7000:05d}.tif')
        cv2.imwrite(Savepath[1] + '\\' + f'alpha{i+7000:05d}.tif', Alpha)
        cv2.imwrite(Savepath[2]+ '\\'+f'Trimap{i+7000:05d}.tif', Trimap)
        def save_npz(**dict):
            return np.save(Savepath[4] +'\\' +f'npz{i+7000:05d}.npy', dict)#OriginImage=each_image,Gimage=Gimage, Alpha=Alpha, Trimap=Trimap)
        save_npz(OriginImage=each_image,Gimage=Gimage, Alpha=Alpha, Trimap=Trimap, CloudMaxDN=CloudMaxDN)
        df = pd.DataFrame({
            'Origin':'after_cut'+'\\'+each.split('\\')[-1],
            'Train':'train_image'+'\\'+f'train{i+7000:05d}.tif',
            'Alpha':'mattingLabel'+'\\'+f'alpha{i+7000:05d}.tif',
            'Trimap':'trimap'+'\\'+f'Trimap{i+7000:05d}.tif',
            'Cloudshadow':'shadow'+'\\'+f'Cloudshadow{i+7000:05d}.tif',
            'npz':'npz'+'\\'+f'npz{i+7000:05d}.npy',
            'CloudMaxDN':CloudMaxDN
        },index=[i])
        DF = DF.append(df)

    DF.to_csv(Savepath[5])




# 生成合成图像的image_label , 是不是对label应该进行一样的操作
# def generate_image_label(image,CloudAlpha,Cloudshadow,CloudDrect=''):
#     imageOut = copy.deepcopy(image)
#     imageOut[CloudAlpha>0.8] = 0
#     imageOut[Cloudshadow<0.1] = 0
#     imageOut[(1-CloudAlpha)*Cloudshadow < 1/9*CloudAlpha] = 0
#     if type(CloudDrect) != str:
#         imageOut[CloudDrect==1] = 0
#     return imageOut





# Visualize.visualize(        image=each_image0,
#                             flip=B2_alpha.flip,
#                             shadow=B2_alpha.g,
#                             # CloudDrect=B2_alpha.CloudDrect,
#                             gimage=Gimage,
#                             Glabel=Glabel)
