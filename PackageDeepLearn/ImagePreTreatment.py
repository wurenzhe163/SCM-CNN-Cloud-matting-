import os,argparse
import numpy as np
from osgeo import gdal
'''
Created on 2021/04/15
@author Mr.w
注意： 路径不可包含中文，输入文件夹内只有图像，默认令nodata=0,如需修改查看read_IMG
python ImagePreTreatment.py --path D:\\deep_road\\test\\val --kernel 256 256 --stride 256 --sifer 0.5 --save_path D:\\deep_road\\test\\test --save_name xx
'''


# search_files = lambda path, endwith='.tif': [os.path.join(path, f) for f in os.listdir(path) if f.endswith(endwith)]
class Img_pre(object):
    def __init__(self,path=[],kernel=[],stride=[],sifer=[],save_path=[],save_name=[],
                 nodata = []
                 ): #,Projection=None
        """
        第一次写面向对象，贼烂 …-_-
                功能:根据输入影像文件夹，将文件夹中的影像批量预处理

        path: 文件所在文件夹路径
        kernel - --->卷积核大小, 矩阵int
        stride - --->步长, int
        save_path: 文件保存文件夹路径
        save_name: 文件保存名称
        sifer: 过滤器，将0值占比大于一定阈值的图像去除
        """
        self.path = path
        self.kernel = kernel
        self.stride = stride
        self.save_path = save_path
        self.save_name = save_name
        self.sifer = sifer
        self.nodata = nodata
        # self.Projection = Projection
        #中间变量
        self.var1 = '  '
        self.var2 = '  '

    def search_files(self):
        path = self.path
        """
        返回当前文件夹下文件
        path : 路径
        Returns: s ,   列表
        """
        s = []
        for f in os.listdir(path):
            s.append(os.path.join(path, f))
        self.var1 = s
        return s

    def read_IMG(self,path):
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


        self.GeoTransform = list(dataset.GetGeoTransform())
        self.img_proj = dataset.GetProjection()

        Raster1 = dataset.GetRasterBand(1)
        # type = gdal.GetDataTypeName(Raster1.DataType)
        # Raster1.GetNoDataValue()

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
        data[data == self.nodata] = 0  # 替换nodata------------------------------------------------------------------------------------
        return data

    def expand_image(self,img):
        stride = self.stride
        kernel = self.kernel
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

    def cut_image(self,img):      #需要输入expand_image
        stride = self.stride
        kernel = self.kernel
        """"
        切片，将影像分成固定大小的块
        img     ---->输入图像,二维或三维矩阵int or float
        """
        a_append = []
        self.coordinate=[]
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
                    self.coordinate.append([self.GeoTransform[0]+ Wmin*self.GeoTransform[1],self.GeoTransform[1],
                                            self.GeoTransform[2],self.GeoTransform[3]+Hmin*self.GeoTransform[5],
                                            self.GeoTransform[4],self.GeoTransform[5]])
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
        return a_append

    def make_dir(self,path):
        isExists = os.path.exists(path)
        # 判断结果
        if not isExists:
            os.makedirs(path)
            print(path + ' 创建成功')
            return True

    # def save_img(self,img,name):      #输入cut_image的列表
    #     pathname = self.save_path
    #     """
    #     img: cut_image的列表
    #     """
    #     self.make_dir(pathname)
    #
    #     # for i in range(len(a_append)):
    #     A = PIL.Image.fromarray(img)
    #     s = pathname + '\\' + name + '.png'
    #     A.save(s)#, quality=95)

    def save_img(self,img_array,name,coordnates=None,img_transf=True):
        pathname = self.save_path
        self.make_dir(pathname)

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


        if img_transf:
            dataset.SetGeoTransform(coordnates)  # 写入仿射变换参数
            dataset.SetProjection(self.img_proj)  # 写入投影


        # 写入影像数据
        if len(img_array.shape) == 2:
            dataset.GetRasterBand(1).WriteArray(img_array)
        else:
            for i in range(img_bands):
                dataset.GetRasterBand(i + 1).WriteArray(img_array[:, :, i])

        dataset = None



    def main(self):
        self.search_files()
        count = 0
        k = 0
        for each_line in self.var1:
            self.var2 = self.read_IMG(each_line)
            self.var2 = self.cut_image(self.expand_image(self.var2))
            if self.sifer != [] : # 加一个过滤器，将背景占比大于一定阈值的图像去除
                self.var2 = [(img,coor) for img,coor in zip(self.var2,self.coordinate) if np.sum(img == 0)/(img.shape[0] * img.shape[1]) < self.sifer]

            print('{}'.format(each_line))
            if k != 0:
                count = count + k + 1
            for k in range(len(self.var2)):
                num = count + k
                name0 = self.save_name + f'{num:05d}'
                # if self.Projection:
                #     AdGeoTransform = [i for i in range(6)]
                #     img_transf = []
                #     for H in range(total_number_H):
                #         # 维度变换
                #         AdGeoTransform[3] = GeoTransform[3] + H * self.stride * GeoTransform[5]
                #         for W in range(total_number_W):
                #             AdGeoTransform[0] = GeoTransform[0] + W * self.stride * GeoTransform[1]
                #         img_transf.append(AdGeoTransform)
                #     self.save_img(np.squeeze(self.var2[k]), name=name0, img_transf=img_transf[k])
                # else:
                self.save_img(np.squeeze(self.var2[k][0]), name=name0,coordnates=self.var2[k][1],img_transf=True)
                print(name0)

def parse_opt():
    parser = argparse.ArgumentParser(description='图像扩充与裁剪')
    parser.add_argument('--path', type=str,default=r'C:\Users\SAR\Desktop\tif', metavar='', help='图像路径')
    parser.add_argument('--kernel', type=int, default=[512,512],nargs='+', metavar='', help='裁剪图像大小')
    parser.add_argument('--stride', type=int,default=412, metavar='', help='裁剪步长')
    parser.add_argument('--nodata', type=int, default=255, metavar='', help='nodata')
    parser.add_argument('--sifer', type=float,default=0.5, metavar='', required=False,help='过滤背景占比大于一定阈值的图像')
    parser.add_argument('--save_path', type=str,default=r'C:\Users\SAR\Desktop\cut', metavar='', help='图像保存路径')
    parser.add_argument('--save_name', type=str, default='',metavar='', help='图像保存名称')
    args = parser.parse_args()
    return args
# test
if __name__ == '__main__':
    args=parse_opt()

    # 尝试保留坐标，繁琐，弃
    Img_pre(path=args.path, kernel=args.kernel,
            stride=args.stride,sifer=args.sifer,
            save_path=args.save_path,
            save_name=args.save_name,
            nodata=args.nodata).main()
