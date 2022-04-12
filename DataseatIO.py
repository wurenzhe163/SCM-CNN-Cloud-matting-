import torch
import os
import numpy as np
from PackageDeepLearn.utils import Visualize, DataIOTrans


search_files = lambda path: sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".npy")])

class AlphaCloudDataset(torch.utils.data.Dataset):
    '''
    This dataset establishment method serves the image semantic segmentation model
    input
        :images_dir    input image path
        :masks_dir     input label path
        :Numclass    Number of image classification labels
        :augmentation=None   DataIOTrans.DataTrans.data_augmentation
    output
        :return         PrefetchDataset(Image, Mask)
    '''

    def __init__(
            self,
            dir,
            Numclass=None,
            augmentation=None,
    ):
        self.npyPath = search_files(os.path.join(dir, "npz"))
        self.Numclass = Numclass
        self.augmentation = augmentation

    def __getitem__(self, i):
        # read images and masks
        Npy = np.load(self.npyPath[i], allow_pickle=True).item()
        MAXDN = 10000
        self.original_image = Npy['OriginImage']/MAXDN
        self.train_image = Npy['Gimage']/MAXDN
        self.mattinglabel = Npy['Alpha'][..., np.newaxis]
        self.CloudMaxDN = Npy['CloudMaxDN']/MAXDN
        self.trimap = Npy['Trimap']
        self.onehotTrimap = DataIOTrans.DataTrans.OneHotEncode(self.trimap, self.Numclass)

        if self.augmentation:
            # 使用合成图像最大值对原始图像以及合成图像归一化
            ImageMask = np.concatenate([self.original_image, self.train_image, self.onehotTrimap, self.mattinglabel], axis=2)
            sample = self.augmentation(ImageMask)
            self.original_image, self.train_image, self.OnehotTrimap, self.mattinglabel = \
            sample[0:3, :, :], sample[3:6, :,:], sample[6:9, :, :], sample[9:10, :,:]

            sample = {"original_image": self.original_image,
                      'train_image': self.train_image,
                      'OnehotTrimap': self.OnehotTrimap,
                      'mattinglabel': self.mattinglabel}

            for i, j in sample.items():
                sample[i] = j.type(torch.FloatTensor)

            return sample, self.CloudMaxDN

        else:
            print("必须使用agumatation")

    def visu(self):
        Visualize.visualize(
            original_image=self.original_image.permute(1,2,0),
            Befor_Argu=self.train_image.permute(1,2,0),
            onehotTrimap=DataIOTrans.DataTrans.OneHotDecode(self.OnehotTrimap.permute(1,2,0)),
            mattinglabel=self.mattinglabel.permute(1,2,0),
        )

    def __len__(self):
        # return length of
        return len(self.npyPath)


if __name__ == '__main__':
    dir = r'D:\train_data\cloud_matting_dataset\train'
    Numclass = 3
    augmentation = DataIOTrans.DataTrans.data_augmentation(ToTensor=True)
    test = AlphaCloudDataset(dir, Numclass=Numclass, augmentation=augmentation)
    print(test.__getitem__(0))
    test.visu()
    print(test.__len__())
