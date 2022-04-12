import cv2,os,copy
import numpy as np
import argparse
from img_byme.PackageDeepLearn.utils import Visualize,DataIOTrans
"""
>>> path = r'D:\cloud5000\matting_label'
>>> savepath = r'D:\cloud5000\trimap'
>>> gen_trimap(path,savepath,size = (2,2))
"""

# 根据前景像素为1，背景像素为0的图像产生 trimap，可能对云来说不可行
def get_args():
    parser = argparse.ArgumentParser(description='Trimap')
    parser.add_argument('--mskDir', type=str, required=True, help="Alpha folder")
    parser.add_argument('--saveDir', type=str, required=True, help="where trimap result save to")
    parser.add_argument('--classes', type=int, required=True, help="Onehot classes")
    parser.add_argument('--size', type=int, nargs='+', required=True, help="dilated kernel size")
    args = parser.parse_args()
    print(args)
    return args
# 搜索文件
search_files = lambda path: sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.tif')])
# 调试
def gen_trimap(path,savepath,classes=3,size=(2,2)):

    names = search_files(path)
    for i,msk_name in enumerate(names):
        msk = cv2.imread(msk_name, cv2.IMREAD_ANYDEPTH)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
        # 膨胀
        dilated = cv2.dilate(msk, kernel, iterations=1)
        dila = copy.deepcopy(dilated)
        # 三值化
        dilated[dilated == 1] = 2
        dilated[(dilated > 0) & (dilated < 1)] = 1
        dilated = dilated.astype(np.uint8)

        onehot_dilated = DataIOTrans.DataTrans.OneHotEncode(dilated,classes)
        # onehot_dilated = onehot_dilated.transpose(2,0,1)  #一定不能用reshape，transpose不改变坐标值，从(h,w,c)-->(c,h,w)
        cv2.imwrite(savepath + '/dilated' + f'{i:04d}' + '.tif', onehot_dilated)
        if i % 1000 == 0:
            Visualize.visualize(mask=msk,dilate=dila,Trimap=dilated,
                                Onehot=onehot_dilated.astype(np.float32)) #plt onehot无法正常显示，需要转化为浮点数

def main():
    args = get_args()
    gen_trimap(args.mskDir,args.saveDir,size=args.size)


if __name__ == "__main__":
    main()
