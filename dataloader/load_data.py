# loaddata.py

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import torchvision

def get_dataloader(dataset,batch_size):
    """
    加载训练验证数据的函数。
    
    参数:

    返回:
    DataLoader: 封装了训练数据的DataLoader对象
    """
    
    if dataset == 'CIFAR10':
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

    # 同样的方法应用于测试集
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
        
        val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        
    elif dataset == 'STL-10':
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        trainset = torchvision.datasets.STL10(root='./data', split="train", transform=transform, download=True)
        testset = torchvision.datasets.STL10(root='./data', split="test", transform=transform, download=True)

    # 同样的方法应用于测试集
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
        
        val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
    
    
    
    elif dataset == 'SVHN':
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        
        trainset = torchvision.datasets.SVHN(root='./data', split="train", transform=transform, download=True)
        testset = torchvision.datasets.SVHN(root='./data', split="test", transform=transform, download=True)
        
    

    # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
        
        val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
    

    return train_loader, val_loader

