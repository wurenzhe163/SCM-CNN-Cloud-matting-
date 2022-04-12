import math

def set_lr(lrdecayType, lr, lrDecay, epoch, train_epoch, optimizer):
    '''
    lrdecayType : lr Change mode
    lr : 
    lrDecay : How many epoch lr changes
    epoch ： Current epoch
    train_epoch : all epoch
    optomizer :
    '''
    print('inputlr={},lrdecayType={},epoch={},train_epoch={}'.format(lr, lrdecayType, epoch, train_epoch))
    decayType = lrdecayType
    if decayType == 'keep':
        lr = lr
    elif decayType == 'step':
        epoch_iter = (epoch + 1) // lrDecay
        lr = lr / 2 ** epoch_iter
    elif decayType == 'exp':
        k = math.log(2) / lrDecay
        lr = lr * math.exp(-k * epoch)
    elif decayType == 'poly':
        lr = lr * math.pow((1 - epoch / train_epoch), 0.9)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print('outputlr={}'.format(lr))
    return lr
