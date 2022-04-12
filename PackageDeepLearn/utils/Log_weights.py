import torch,os
import pandas as pd

class Train_Log():
    '''
    创建保存路径、文件夹
    '''

    def __init__(self, saveDir):

        self.save_dir = saveDir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # create dir name=model
        self.save_dir_model = os.path.join(self.save_dir, 'model')
        if not os.path.exists(self.save_dir_model):
            os.makedirs(self.save_dir_model)

    def save_model(self, model, epoch, lr, train_loss=False, val_loss=False):
        """
        save_model 加上监测功能，并且尽量取消覆盖
        """
        if train_loss:
            lastest_out_path = "{}/ckpt_train.pth".format(self.save_dir_model)
            torch.save({
                'epoch': epoch,
                'lr':lr,
                'state_dict': model.state_dict(),
            }, lastest_out_path)
        elif val_loss:
            lastest_out_path = "{}/ckpt_val.pth".format(self.save_dir_model)
            torch.save({
                'epoch': epoch,
                'lr':lr,
                'state_dict': model.state_dict(),
            }, lastest_out_path)
        else:
            model_out_path = "{}/{}model_obj.pth".format(self.save_dir_model, f'{epoch:04d}')
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'state_dict': model.state_dict(),
            }, model_out_path)


    def load_model(self, model, lastest_out_path):
        #         lastest_out_path = "{}/ckpt_lastest.pth".format(self.save_dir_model)
        ckpt = torch.load(lastest_out_path)
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(lastest_out_path, ckpt['epoch']))

        return start_epoch, model

    #     def namekeys(self,ckpt,headstr):
    #         '''
    #         修改有序字典的键
    #         '''
    #         oldkeys=[key for key in ckpt['state_dict'].keys()]
    #         for key in oldkeys:
    #             newkey = headstr + key
    #             ckpt['state_dict'][newkey] = ckpt['state_dict'][key]
    #             del ckpt['state_dict'][key]

    def load_model_end(self, model, lastest_out_path, headstr='t_net.'):
        """
        强制替换model前置层的权重
        Args:
            model:  输入模型
            lastest_out_path: 载入权重的模型
            headstr: layer前置字符

        Returns: 替换权重后的模型

        """
        ckpt = torch.load(lastest_out_path)
        #         self.namekeys(ckpt,headstr)
        Model_dict = model.state_dict()
        for key1, key2 in zip(Model_dict.keys(), ckpt['state_dict'].keys()):
            Model_dict[key1] = ckpt['state_dict'][key2]
        model.load_state_dict(Model_dict)
        start_epoch = ckpt['epoch']
        print("=> loaded checkpoint '{}' (epoch {})".format(lastest_out_path, ckpt['epoch']))
        return start_epoch, model

    def save_log(self, log, logname='/log.csv'):

        if os.path.exists(self.save_dir + logname):
            # df = pd.read_csv(self.save_dir + logname)
            # df = pd.concat([df, log], axis=0, ignore_index=True)  # ignore_index
            log.to_csv(self.save_dir + logname,mode='a', index=False,header=0)
        else:
            log.to_csv(self.save_dir + logname, mode='w', index=False)