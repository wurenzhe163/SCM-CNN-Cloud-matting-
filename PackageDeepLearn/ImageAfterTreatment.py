# -*- coding: utf-8 -*-
"""
Created on 2021/04/15 
@author Mr.w
注意： 路径不可包含中文
避免转义字符
python ImageAfterTreatment.py --path D:/deep_road/test/val --ImgPrePath D:/deep_road/test/test --kernel 256 256 --stride 256 --ImgUnionPath D:/deep_road/test/union
"""
import os,argparse
import numpy as np
from osgeo import gdal




class Img_Post(object):
    """
    预测影像合并并赋予坐标系
    """

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
    # 原始文件的长宽高，以及扩充后的长宽高
    @staticmethod
    def expand_image(img,stride,kernel):

        """
        填充，在右方，下方填充0
        img: 输入图像,二维或三维矩阵int or float, as : img.shape=(512,512,3)
        kernel: 卷积核大小,矩阵int
        stride: 步长,int
        """
        expand_H = stride - (img.shape[0] - kernel[0]) % stride
        expand_W = stride - (img.shape[1] - kernel[1]) % stride
        if len(img.shape) == 3:
            img2 = img.shape[2]
            H_x = np.zeros((expand_H, img.shape[1], img2), dtype=img.dtype)
            W_x = np.zeros((img.shape[0] + expand_H, expand_W, img2), dtype=img.dtype)
        else:
            H_x = np.zeros((expand_H, img.shape[1]), dtype=img.dtype)
            W_x = np.zeros((img.shape[0] + expand_H, expand_W), dtype=img.dtype)
        img = np.r_[img, H_x]  # 行
        # img = np.c_[img,W_x]#列
        img = np.concatenate([img, W_x], axis=1)
        return img
    @staticmethod
    def cut_image(img,kernel,stride):      #需要输入expand_image

        """"
        切片，将影像分成固定大小的块
        img     ---->输入图像,二维或三维矩阵int or float
        """
        a_append = []
        # global total_number_H,total_number_W
        total_number_H = int((img.shape[0] - kernel[0]) / stride + 1)
        total_number_W = int((img.shape[1] - kernel[1]) / stride + 1)
        if len(img.shape) == 3:
            for H in range(total_number_H):  # H为高度方向切片数
                Hmin = H * stride
                Hmax = H * stride + kernel[0]

                for W in range(total_number_W):  # W为宽度方向切片数
                    Wmin = W * stride
                    Wmax = W * stride + kernel[1]
                    imgd = img[Hmin:Hmax, Wmin:Wmax, :]
                    a_append.append(imgd)
        else:
            for H in range(total_number_H):
                Hmin = H * stride
                Hmax = H * stride + kernel[0]

                for W in range(total_number_W):
                    Wmin = W * stride
                    Wmax = W * stride + kernel[1]
                    imgd = img[Hmin:Hmax, Wmin:Wmax]
                    a_append.append(imgd)
        if total_number_H * total_number_W == len(a_append):
            print('cut right')
        else:
            print('cut wrong')
        return a_append,total_number_H,total_number_W

    @classmethod
    def read_shape(cls, path, kernel=[256, 256], stride=256):
        img = cls.read_IMG(path)
        r0, s0, w0 = img.shape
        print('img_shape = {} {} {}and dtype={}'.format(img.shape[0], img.shape[1], img.shape[2], img.dtype))
        img = cls.expand_image(img,stride,kernel)
        r1, s1, w1 = img.shape
        r1 = int((img.shape[0] - kernel[0]) / stride + 1)
        s1 = int((img.shape[1] - kernel[1]) / stride + 1)
        print('expand_img_shape = {} {} {}and dtype={}'.format(img.shape[0], img.shape[1], img.shape[2], img.dtype))

        return [r0, s0, w0, r1, s1, w1]

    @staticmethod
    def join_image(img, kernel=[512, 512], stride=512, H=17, W=10, S=0):
        """
        重叠区域用后面的影像直接叠加
        Parameters
        ----------
        img : 矩阵[img1,img2,img3]
        kernel : TYPE, optional
            DESCRIPTION. The default is [512,512].
        stride : TYPE, optional
            DESCRIPTION. The default is 512.
        H : 行方向数量
        W : 列方向数量
        S : 通道数

        Returns
        -------
        zeros_np : 合成后的单个矩阵
            DESCRIPTION.

        """
        zeros_np = np.zeros(
            (H * kernel[0] - (H - 1) * (kernel[0] - stride), W * kernel[1] - (W - 1) * (kernel[1] - stride), S),
            dtype=img[0].dtype)
        num_index = 0
        if len(img[0].shape) == 3:
            for h in range(H):  # H为高度方向切片数
                Hmin = h * stride
                Hmax = h * stride + kernel[0]
                for w in range(W):  # W为宽度方向切片数
                    Wmin = w * stride
                    Wmax = w * stride + kernel[1]
                    zeros_np[Hmin:Hmax, Wmin:Wmax, :] = img[num_index]
                    print('Hmin:{},Hmax:{},Wmin:{},Wmax:{},合并目标：{}'.format(Hmin, Hmax, Wmin, Wmax, zeros_np.shape))
                    num_index += 1
        else:
            for h in range(H):  # H为高度方向切片数
                Hmin = h * stride
                Hmax = h * stride + kernel[0]
                for w in range(W):  # W为宽度方向切片数
                    Wmin = w * stride
                    Wmax = w * stride + kernel[1]
                    zeros_np[Hmin:Hmax, Wmin:Wmax] = img[num_index]
                    num_index += 1

        return zeros_np

    @staticmethod
    def join_image2(img, kernel=[512, 512], stride=512, H=17, W=10, S=0):
        """
        重叠区域用后面的影像对半，注意stride是偶数
        Parameters
        ----------
        img : 矩阵[img1,img2,img3]
        kernel : TYPE, optional
            DESCRIPTION. The default is [512,512].
        stride : TYPE, optional
            DESCRIPTION. The default is 512.
        H : 行方向数量
        W : 列方向数量
        S : 通道数

        Returns
        -------
        zeros_np : 合成后的单个矩阵
            DESCRIPTION.

        """
        zeros_np = np.zeros(
            (H * kernel[0] - (H - 1) * (kernel[0] - stride), W * kernel[1] - (W - 1) * (kernel[1] - stride), S),
            dtype=img[0].dtype)
        imgShape = img[0].shape
        num_index = 0
        # 对半补偿
        Compense = [int((kernel[0] - stride)/2), int((kernel[1] - stride)/2)]
        if len(imgShape) == 2:
            img = [each[...,np.newaxis] for each in img]

        for h in range(H):  # H为高度方向切片数
            if h == 0:
                Hmin = h * stride
            else:
                Hmin = h * stride + Compense[0]
            if h == H-1:
                Hmax = h * stride + kernel[0]
            else:
                Hmax = h * stride + kernel[0] - Compense[0]
            for w in range(W):  # W为宽度方向切片数
                if w==0:
                    Wmin = w * stride
                else:
                    Wmin = w * stride + Compense[1]
                if w==W-1:
                    Wmax = w * stride + kernel[1]
                else:
                    Wmax = w * stride + kernel[1] - Compense[1]


                if h == 0 :
                    img[num_index] = img[num_index][0:(kernel[0]-Compense[0]),:,:]
                    if w != 0 and w != W-1 :
                        img[num_index] = img[num_index][:, Compense[1]:kernel[1]-Compense[1], :]

                if h == H-1:
                    img[num_index] = img[num_index][Compense[0]:kernel[0], :, :]
                    if w != 0 and w != W-1 :
                        img[num_index] = img[num_index][:, Compense[1]:kernel[1]-Compense[1], :]

                if w == 0 :
                    img[num_index] = img[num_index][:,0:kernel[1]-Compense[1],:]
                    if h != 0 and h != H-1:
                        img[num_index] = img[num_index][Compense[0]:kernel[0]-Compense[0],:,:]
                if w == W-1:
                    img[num_index] = img[num_index][ :, Compense[1]:kernel[1],:]
                    if h != 0 and h != H - 1:
                        img[num_index] = img[num_index][Compense[0]:kernel[0] - Compense[0], :, :]
                if h != 0 and w != 0  and h != H-1 and w != W-1 :
                    img[num_index] = img[num_index][Compense[0]:kernel[0]-Compense[0],Compense[1]:kernel[1]-Compense[1],:]

                zeros_np[Hmin:Hmax, Wmin:Wmax, :] = img[num_index]

                # print('Hmin:{},Hmax:{},Wmin:{},Wmax:{},合并目标：{}'.format(Hmin, Hmax, Wmin, Wmax, zeros_np.shape))
                num_index += 1

        return zeros_np

    @staticmethod
    def make_dir(path):
        isExists = os.path.exists(path)
        # 判断结果
        if not isExists:
            os.makedirs(path)
            print(path + ' 创建成功')
            return True

    @classmethod
    def save_img(cls,img_array,pathname,name):

        cls.make_dir(pathname)

        #判断数据类型
        if 'int8' in img_array.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in img_array.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        #判断数据维度，仅接受shape=3或shape=2
        if len(img_array.shape) == 3:
            im_height, im_width, img_bands = img_array.shape
        else:
            img_bands, (im_height, im_width) = 1, img_array.shape

        SavePath = pathname + '\\' + name + '.tif'
        driver = gdal.GetDriverByName("GTiff")  # 创建文件驱动
        dataset = driver.Create(SavePath, im_width, im_height, img_bands, datatype)


        # 写入影像数据
        if len(img_array.shape) == 2:
            dataset.GetRasterBand(1).WriteArray(img_array)
        else:
            for i in range(img_bands):
                dataset.GetRasterBand(i + 1).WriteArray(img_array[:, :, i])

        dataset = None


        # 投影坐标
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

    @classmethod
    def mosaic_img(cls,img_path, rsw, endwith='', img_size=[256, 256, 1], stride=256):
        '''
        :param img_path:一组图像的地址
        :param rsw: 参数
        :param endwith: 后缀名
        :param img_size: 图像大小
        :param stride: 步长
        :return:
        '''
        img = []
        if endwith != '':
            if endwith == '.npy':
                for each_img in img_path:
                    img.append(np.load(each_img))
            else : #endwith == '.png' or endwith == '.jpg' or endwith == '.tif':
                #读取图像
                for each_img in img_path:
                    image = cls.read_IMG(each_img)
                    img.append(image.reshape(img_size))
            joint_img = cls.join_image(img, kernel=img_size[0:2], stride=stride, H=rsw[3], W=rsw[4],
                                             S=img_size[-1])  # rsw[5]
            joint_img = joint_img[0:rsw[0], 0:rsw[1], :]  # z左上角切片，扩充零在右下角
            return joint_img
        else:
            print('False/n')
            return 0

    @staticmethod
    def erode_image(bin_image,kernel_size):
        """
        erode bin image
        Args:
            bin_image: image with 0,1 pixel value
        Returns:
            erode image
        """
        # kernel = np.ones(shape=(kernel_size, kernel_size))

        if ((kernel_size % 2) == 0) or (kernel_size < 1):
            raise ValueError("kernel size must be odd and bigger than 1")
        # if (bin_image.max() != 1) or (bin_image.min() != 0):
        #     raise ValueError("input image's pixel value must be 0 or 1")
        d_image = np.zeros(shape=bin_image.shape)
        center_move = int((kernel_size - 1) / 2)
        for i in range(center_move, bin_image.shape[0] - kernel_size + 1):
            for j in range(center_move, bin_image.shape[1] - kernel_size + 1):
                d_image[i, j] = np.min(bin_image[i - center_move:i + center_move,
                                       j - center_move:j + center_move])
        return d_image

    @staticmethod
    def dilate_image(bin_image,kernel_size):
        """
        dilate bin image
        Args:
            bin_image: image as label
        Returns:
            dilate image
        """
        if (kernel_size % 2 == 0) or kernel_size < 1:
            raise ValueError("kernel size must be odd and bigger than 1")
        # if (bin_image.max() != 1) or (bin_image.min() != 0):
        #     raise ValueError("input image's pixel value must be 0 or 1")
        d_image = np.zeros(shape=bin_image.shape)
        center_move = int((kernel_size - 1) / 2)
        for i in range(center_move, bin_image.shape[0] - kernel_size + 1):
            for j in range(center_move, bin_image.shape[1] - kernel_size + 1):
                d_image[i, j] = np.max(bin_image[i - center_move:i + center_move, j - center_move:j + center_move])
        return d_image

    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='图像合并、开闭运算、重投影')
    parser.add_argument('--path', type=str, metavar='', help='原始图像路径')
    parser.add_argument('-s', '--ErodDilateSize', type=str, metavar='', default=0, help='图像开闭运算大小，0不进行操作')
    parser.add_argument('--ImgPrePath', type=str, metavar='', help='预测图像路径')
    parser.add_argument('--kernel', type=int, nargs='+', metavar='', help='裁剪图像大小')
    parser.add_argument('--stride', type=int, metavar='', help='裁剪步长')
    parser.add_argument('--ImgUnionPath', type=str, metavar='', help='合并预测图像路径')
    args = parser.parse_args()

    path = args.path  # r'D:\deep_road\test\val'
    erod_dilate_size = args.ErodDilateSize  # 1
    imgprePath = args.ImgPrePath  # r'D:\deep_road\test\test'
    kernel = args.kernel  # [256,256]
    stride = args.stride  # 256
    img_pre_union = args.ImgUnionPath  # r'D:\deep_road\test\uion'

    search_files = lambda path: sorted([os.path.join(path, f) for f in os.listdir(path)])
    path_pre = search_files(imgprePath)
    if erod_dilate_size != 0:
        print('执行影像腐蚀膨胀算法')
        # 腐蚀膨胀
        files_num = len(path_pre)
        num = 0
        for each_image in path_pre:
            image = Img_Post.read_IMG(each_image)
            num += 1
            print('执行到第{}个文件，共{}个文件'.format(num, files_num))
            if np.max(image) != 0:
                #######连接
                # 膨胀
                image = Img_Post.dilate_image(image, erod_dilate_size)
                # 腐蚀
                image = Img_Post.erode_image(image, erod_dilate_size)

                #######削减碎斑
                # 腐蚀
                image = Img_Post.erode_image(image, erod_dilate_size)
                # 膨胀
                image = Img_Post.dilate_image(image, erod_dilate_size)
                A = Img_Post.save_img(image,imgprePath,'AfterED')
            else:
                continue
    else:
        print('跳过影像腐蚀膨胀运算')
    # 加载分割前的图片
    img_path_ = search_files(path)
    for i, tif_path in enumerate(img_path_):
        # 拼接时所需参数
        rsw = Img_Post.read_shape(path=tif_path, kernel=kernel, stride=stride)
        img_path_2 = path_pre[0:rsw[3] * rsw[4]]
        del path_pre[0:rsw[3] * rsw[4]]
        img_size = kernel.copy()
        img_size.append(rsw[2])
        mask = Img_Post.mosaic_img(img_path_2, rsw, endwith='.tif', stride=stride, img_size=img_size)
        name0 = 'pre_all' + f'{(i + 1):04d}'

        Img_Post.save_img(img_array=np.squeeze(mask), pathname=img_pre_union, name=name0)

    non_ = search_files(img_pre_union)

    print('********执行重投影*********')
    pos = img_path_
    for each_non, each_pos in zip(non_, pos):
        Img_Post.copy_geoCoordSys(each_pos, each_non)
