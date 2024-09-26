import math

def adjust_learning_rate(comargs, optimizer, epoch):
    """Sets the learning rate"""
    if not comargs.cos_lr:
        if epoch in comargs.milestones:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= comargs.lr_decay_rate
        print('lr:')
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

    else:
        for param_group in optimizer.param_groups:
            if epoch <= 10:
                param_group['lr'] = 0.5 * comargs.lr \
                                * (1 + math.cos(math.pi * epoch / comargs.epochs)) * (epoch - 1) / 10 + 0.01 * (11 - epoch) / 10
            else:
                param_group['lr'] = 0.5 * comargs.lr\
                                    * (1 + math.cos(math.pi * epoch / comargs.epochs))
        print('lr:')
        for param_group in optimizer.param_groups:
            print(param_group['lr'])