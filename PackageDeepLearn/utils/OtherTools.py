import torch

def DEVICE_SLECT():
    '''
    :return cpu || cuda
    '''
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Will use {}'.format(DEVICE))
    return DEVICE

