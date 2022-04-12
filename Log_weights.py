import torch
import os
import pandas as pd


class Train_Log():
    def __init__(self, saveDir):

        self.save_dir = saveDir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # create dir name=model
        self.save_dir_model = os.path.join(self.save_dir, 'model')
        if not os.path.exists(self.save_dir_model):
            os.makedirs(self.save_dir_model)

    def save_model(self, model, epoch, lr, train_loss=False, val_loss=False):
        if train_loss:
            lastest_out_path = "{}/ckpt_train.pth".format(self.save_dir_model)
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'state_dict': model.state_dict(),
            }, lastest_out_path)
        elif val_loss:
            lastest_out_path = "{}/ckpt_val.pth".format(self.save_dir_model)
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'state_dict': model.state_dict(),
            }, lastest_out_path)
        else:
            model_out_path = "{}/{}model_obj.pth".format(
                self.save_dir_model, f'{epoch:04d}')
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'state_dict': model.state_dict(),
            }, model_out_path)

    def load_model(self, model, lastest_out_path):

        ckpt = torch.load(lastest_out_path)
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(
            lastest_out_path, ckpt['epoch']))

        return start_epoch, model

    def load_model_end(self, model, lastest_out_path, headstr='t_net.'):

        ckpt = torch.load(lastest_out_path)
        #         self.namekeys(ckpt,headstr)
        Model_dict = model.state_dict()
        for key1, key2 in zip(Model_dict.keys(), ckpt['state_dict'].keys()):
            Model_dict[key1] = ckpt['state_dict'][key2]
        model.load_state_dict(Model_dict)
        start_epoch = ckpt['epoch']
        print("=> loaded checkpoint '{}' (epoch {})".format(
            lastest_out_path, ckpt['epoch']))

        return start_epoch, model

    def save_log(self, log, logname='/log.csv'):

        if os.path.exists(self.save_dir + logname):

            log.to_csv(self.save_dir + logname,
                       mode='a', index=False, header=0)
        else:
            log.to_csv(self.save_dir + logname, mode='w', index=False)
