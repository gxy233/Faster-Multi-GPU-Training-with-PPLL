import torch.optim as optim

def get_optimizer(pg,optimizer,lr,momentum,weight_decay):
    if optimizer == 'SGD':
        opti= optim.SGD(pg, lr=lr,momentum=momentum,weight_decay=weight_decay)
        
        
    return opti