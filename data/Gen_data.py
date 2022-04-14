import cv2,os,copy
import numpy as np
from img_byme.PackageDeepLearn.utils import Visualize

# 读入alpha图像以及图像save路径
def read_data(Cloud_matting_datasetPath):
    B2_alpha_path = os.path.join(Cloud_matting_datasetPath, 'cloud_clip.tif')
    Save_trimap = os.path.join(Cloud_matting_datasetPath, 'trimap_0.8.tif')
    Save_cloudshadow = os.path.join(Cloud_matting_datasetPath, 'cloudshadow.tif')
    B2_alpha = (cv2.imread(B2_alpha_path,cv2.IMREAD_ANYDEPTH) /65535).round(2)
    return (B2_alpha_path,Save_trimap,Save_cloudshadow,B2_alpha)

# 生成trimap图像
def gen_trimap(B2_alpha,size=(5,5)):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    # 膨胀
    dilated = cv2.dilate(B2_alpha, kernel, iterations=1)
    # 腐蚀
    erode   = cv2.erode(dilated,kernel,iterations=1)

    Visualize.visualize(dilated=dilated, alpha=B2_alpha, erode=erode)

    # 三值化
    erode[erode >= 0.8] = 255
    erode[(erode > 0) & (erode < 0.8)] = 125
    erode = erode.astype(np.uint8)
    return erode

# 生成cloud_shadow
def generate_cloudShadow(B2_alpha):
    h,w = B2_alpha.shape
    h_ = np.random.randint(low=100, high=200)* np.random.choice([-1, 1])
    w_ = np.random.randint(low=100, high=200)* np.random.choice([-1,1])
    print('offset=({},{})'.format(h_, w_))

    img_pad = np.pad(np.ones_like(B2_alpha), pad_width=((600,600),(600,600)), mode='constant', constant_values=(1, 1))

    # 在偏移的位置计算遮挡度,这里使用600纯粹是为了后面的减法运算不出现负号
    img_pad[600+h_:600+h+h_,600+w_:600+w+w_] = 1 - B2_alpha
    cloud_shadow = img_pad[600 + h_:600 + h + h_, 600 + w_:600 + w + w_]
    return cloud_shadow

# 直接运行
def main(Cloud_matting_datasetPath):
    B2_alpha_path,Save_trimap,Save_cloudshadow,B2_alpha = read_data(Cloud_matting_datasetPath)

    cv2.imwrite(Save_trimap, gen_trimap(B2_alpha,size=(5, 5)))   # save_trimap
    cv2.imwrite(Save_cloudshadow, generate_cloudShadow(B2_alpha))  # save_cloud shadow
if __name__ == "__main__":
    Cloud_matting_datasetPath = r'D:\train_data\cloud_matting_dataset'
    main(Cloud_matting_datasetPath)
