import numpy as np
import os,torch
from torchvision import transforms
from osgeo import gdal

search_files = lambda path : sorted([os.path.join(path,f) for f in os.listdir(path)])

def make_dir(path):
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    return path

class Denormalize(object):
    '''
    return : 反标准化
    '''
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return transforms.functional.normalize(tensor, self._mean, self._std)

class DataTrans(object):
    @staticmethod
    def OneHotEncode(LabelImage,NumClass):
        '''
        Onehot Encoder  2021/03/23 by Mr.w
        -------------------------------------------------------------
        LabelImage ： Ndarry   |   NumClass ： Int
        -------------------------------------------------------------
        return： Ndarry
        '''
        one_hot_codes = np.eye(NumClass).astype(np.uint8)
        try:
            one_hot_label = one_hot_codes[LabelImage]
        except IndexError:
            # pre treat cast 不连续值 到 连续值
            Unique_code = np.unique(LabelImage)
            Objectives_code = np.arange(len(Unique_code))
            for i, j in zip(Unique_code, Objectives_code):
                LabelImage[LabelImage == i] = j
                # print('影像编码从{},转换为{}'.format(i, j))

            one_hot_label = one_hot_codes[LabelImage]
        # except IndexError:
        #     # print('标签存在不连续值，最大值为{}--->已默认将该值进行连续化,仅适用于二分类'.format(np.max(LabelImage)))
        #     LabelImage[LabelImage == np.max(LabelImage)] = 1
        #     one_hot_label = one_hot_codes[LabelImage]
        return one_hot_label

    @staticmethod
    def OneHotDecode(OneHotImage):
        '''
        OneHotDecode 2021/03/23 by Mr.w
        -------------------------------------------------------------
        OneHotImage : ndarray -->(512,512,x)
        -------------------------------------------------------------
        return : image --> (512,512)
        '''
        return np.argmax(OneHotImage,axis=-1)

    @staticmethod
    def data_augmentation(ToTensor=False,Resize=None,Contrast=None,Equalize=None,HFlip=None,Invert=None,VFlip=None,
                          Rotation=None,Grayscale=None,Perspective=None,Erasing=None,Crop=None,
                          ): # dHFlip=None
        """

        DataAgumentation 2021/03/23 by Mr.w
        -------------------------------------------------------------
        ToTensor : False/True , 注意转为Tensor，通道会放在第一维
        Resize : tuple-->(500,500)
        Contrast : 0-1 -->图像被自动对比度的可能,支持维度1-3
        Equalize : 0-1 -->图像均衡可能性 , 仅支持uint8
        HFlip : 0-1 --> 图像水平翻转
        Invert : 0-1--> 随机翻转
        VFlip : 0-1 --> 图像垂直翻转
        Rotation : 0-360 --> 随机旋转度数范围, as : 90 , [-90,90]
        Grayscale : 0-1 --> 随机转换为灰度图像
        Perspective : 0-1 --> 随机扭曲图像
        Erasing : 0-1 --> 随机擦除
        Crop : tuple --> (500,500)
        -------------------------------------------------------------
        return : transforms.Compose(train_transform) --> 方法汇总
        """
        #列表导入Compose
        train_transform = []
        if ToTensor == True:
            trans_totensor = transforms.ToTensor()
            train_transform.append(trans_totensor)

        if Resize != None:
            trans_Rsize = transforms.Resize(Resize)  # Resize=(500,500)
            train_transform.append(trans_Rsize)
        if Contrast != None:
            trans_Rcontrast = transforms.RandomAutocontrast(p=Contrast)
            train_transform.append(trans_Rcontrast)
        if Equalize != None:
            trans_REqualize = transforms.RandomEqualize(p=Equalize)
            train_transform.append(trans_REqualize)
        if HFlip != None:
            train_transform.append(transforms.RandomHorizontalFlip(p=HFlip))
        if Invert != None:
            train_transform.append(transforms.RandomInvert(p=Invert))
        if VFlip != None:
            train_transform.append(transforms.RandomVerticalFlip(p=VFlip))
        if Rotation != None:
            train_transform.append(transforms.RandomRotation(Rotation,expand=False,center=None,fill=0,resample=None))
        if Grayscale != None:
            train_transform.append(transforms.RandomGrayscale(p=Grayscale))
        if Perspective != None:
            train_transform.append(transforms.RandomPerspective(distortion_scale=0.5,p=Perspective,fill=0))
        if Erasing != None:
            train_transform.append(transforms.RandomErasing(p=Erasing,scale=(0.02, 0.33),ratio=(0.3, 3.3),value=0,inplace=False))
        if Crop != None:
            train_transform.append(transforms.RandomCrop(Crop,padding=None,pad_if_needed=False,fill=0,padding_mode='constant'))

        # class Detect_RandomHorizontalFlip(torch.nn.Module):
        #     """随机水平翻转图像以及bboxes
        #         用于目标检测
        #     """
        #
        #     def __init__(self, p=0.5):
        #         self.prob = p
        #
        #     def forward(self, image, target):
        #         if torch.rand(1).item() < self.prob:
        #             height, width = image.shape[-2:]
        #             image = image.flip(-1)  # 水平翻转图片
        #             bbox = target["boxes"]
        #             # bbox: xmin, ymin, xmax, ymax
        #             bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
        #             target["boxes"] = bbox
        #         return image, target
        #     def __repr__(self):
        #         return self.__class__.__name__ + '(p={})'.format(self.prob)
        # if dHFlip != 0 :
        #     train_transform.append(Detect_RandomHorizontalFlip(p=dHFlip))

        return transforms.Compose(train_transform)
    @staticmethod
    def copy_geoCoordSys(img_pos_path, img_none_path):
        '''
        获取img_pos坐标，并赋值给img_none
        :param img_pos_path: 带有坐标的图像
        :param img_none_path: 不带坐标的图像
        '''

        def def_geoCoordSys(read_path, img_transf, img_proj):
            array_dataset = gdal.Open(read_path)
            img_array = array_dataset.ReadAsArray(0, 0, array_dataset.RasterXSize, array_dataset.RasterYSize)
            if 'int8' in img_array.dtype.name:
                datatype = gdal.GDT_Byte
            elif 'int16' in img_array.dtype.name:
                datatype = gdal.GDT_UInt16
            else:
                datatype = gdal.GDT_Float32

            if len(img_array.shape) == 3:
                img_bands, im_height, im_width = img_array.shape
            else:
                img_bands, (im_height, im_width) = 1, img_array.shape

            filename = read_path[:-4] + '_proj' + read_path[-4:]
            driver = gdal.GetDriverByName("GTiff")  # 创建文件驱动
            dataset = driver.Create(filename, im_width, im_height, img_bands, datatype)
            dataset.SetGeoTransform(img_transf)  # 写入仿射变换参数
            dataset.SetProjection(img_proj)  # 写入投影

            # 写入影像数据
            if img_bands == 1:
                dataset.GetRasterBand(1).WriteArray(img_array)
            else:
                for i in range(img_bands):
                    dataset.GetRasterBand(i + 1).WriteArray(img_array[i])
            print(read_path, 'geoCoordSys get!')

        dataset = gdal.Open(img_pos_path)  # 打开文件
        img_pos_transf = dataset.GetGeoTransform()  # 仿射矩阵
        img_pos_proj = dataset.GetProjection()  # 地图投影信息
        def_geoCoordSys(img_none_path, img_pos_transf, img_pos_proj)

class DataIO(object):
    @staticmethod
    def _get_dir(*path,DATA_DIR=r''):
        """
        as : a = ['train', 'val', 'test'] ; _get_dir(*a,DATA_DIR = 'D:\\deep_road\\tiff')
        :return list path
        """
        return [os.path.join(DATA_DIR, each) for each in path]

    @staticmethod
    def read_IMG(path):
        """
        读为一个numpy数组,读取所有波段
        对于RGB图像仍然是RGB通道，cv2.imread读取的是BGR通道
        path : img_path as:c:/xx/xx.tif
        """
        dataset = gdal.Open(path)
        if dataset == None:
            raise Exception("Unable to read the data file")

        nXSize = dataset.RasterXSize  # 列数
        nYSize = dataset.RasterYSize  # 行数
        bands = dataset.RasterCount  # 波段
        Raster1 = dataset.GetRasterBand(1)

        if Raster1.DataType == 1 :
            datatype = np.uint8
        elif Raster1.DataType == 2:
            datatype = np.uint16
        else:
            datatype = float

        data = np.zeros([nYSize, nXSize, bands], dtype=datatype)
        for i in range(bands):
            band = dataset.GetRasterBand(i + 1)
            data[:, :, i] = band.ReadAsArray(0, 0, nXSize, nYSize)  # .astype(np.complex)
        return data



    @staticmethod
    def save_Gdal(img_array, SavePath, img_transf=False, coordnates=None, img_proj=None):
        """

        Args:
            img_array:  [H,W,C] , RGB色彩，不限通道深度
            SavePath:
            img_transf: 是否进行投影
            coordnates: 仿射变换
            img_proj: 投影信息

        Returns: 0

        """

        # 判断数据类型
        if 'int8' in img_array.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in img_array.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 判断数据维度，仅接受shape=3或shape=2
        if len(img_array.shape) == 3:
            im_height, im_width, img_bands = img_array.shape
        else:
            img_bands, (im_height, im_width) = 1, img_array.shape

        driver = gdal.GetDriverByName("GTiff")  # 创建文件驱动
        dataset = driver.Create(SavePath, im_width, im_height, img_bands, datatype)

        if img_transf:
            dataset.SetGeoTransform(coordnates)  # 写入仿射变换参数
            dataset.SetProjection(img_proj)  # 写入投影

        # 写入影像数据
        if len(img_array.shape) == 2:
            dataset.GetRasterBand(1).WriteArray(img_array)
        else:
            for i in range(img_bands):
                dataset.GetRasterBand(i + 1).WriteArray(img_array[:, :, i])

        dataset = None


# 待调整 后续请参照https://github.com/wurenzhe163/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/utils/train.py
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb
class SegmentDataset(torch.utils.data.Dataset):
    '''
    该数据集建立方法服务于图像语义分割模型
    input
        :images_dir    输入图像路径
        :masks_dir     输入标签路径
        :Numclass=2    图像分类标签数
        :augmentation=None   图像扩充方式DataTrans.data_augmentation
    output
        :return         PrefetchDataset(Image, Mask)
    '''

    def __init__(
            self,
            images_dir,
            masks_dir,
            Numclass=None,  # 分类数
            augmentation=None,
    ):
        self.image_paths = search_files(images_dir)
        self.mask_paths = search_files(masks_dir)
        self.Numclass = Numclass
        self.augmentation = augmentation


    def __getitem__(self, i):
        # read images and masks
        image = DataIO.read_IMG(self.image_paths[i])
        mask  = DataIO.read_IMG(self.mask_paths[i])
        if i==1 :
            print('img={},label={}'.format(image.shape,mask.shape))

        # one-hot-encode the mask  # ， tgs的损失函数内部计算loss这里重复
        mask1 = DataTrans.OneHotEncode(mask, self.Numclass)
        # apply augmentations
        if self.augmentation:
            ImageMask = np.concatenate([image, mask1], axis=2)  # 图像与Lable一同变换
            sample = self.augmentation(ImageMask)
            image2, mask2 = sample[0:image.shape[2], :, :], sample[image.shape[2]:, :, :].type(torch.int64)
            # 注意,经过augmentation,数据dtype=float64,需要转换数据类型才能够正常显示
        self.image = image
        self.image2 = image2
        self.mask2 = mask2
        return image2, mask2

    def visu(self):
        from PackageDeepLearn.utils import Visualize
        Visualize.visualize(
            Befor_Argu=self.image,
            After_Argu=self.image2.permute(1, 2, 0).numpy().astype(np.uint8),
            Label=DataTrans.OneHotDecode(self.mask2.permute(1, 2, 0).numpy().astype(np.uint8))
        )

    def __len__(self):
        # return length of
        return len(self.image_paths)



