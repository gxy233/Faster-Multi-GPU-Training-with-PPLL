import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml
import threading
from models.models import getmodel, NetworkA, NetworkB
from dataloader.load_data import load_data
from trainingFw import train_front, train_back


def train_A(netA, deviceA, optimizerA, criterion, dataloader, cache):
    """
    在GPU上训练网络A的线程函数。

    参数:
    netA (nn.Module): 网络A模型
    deviceA (str): 网络A所在的设备（如 'cuda:0'）
    optimizerA (optim.Optimizer): 网络A的优化器
    criterion (nn.Module): 损失函数
    dataloader (DataLoader): 数据加载器
    cache (list): 用于缓存网络A的输出数据
    """
    for inputs, labels in dataloader:
        inputs = inputs.to(deviceA)
        optimizerA.zero_grad()
        outputsA = netA(inputs)
        
        lossA = criterion(outputsA, labels)
        lossA.backward()
        optimizerA.step()
        
        cache.append(outputsA.detach().cpu(),labels.detach().cpu())
        
    ###### Append 一个停止信号
    cache.append('END')
    

def train_B(netB, deviceB, optimizerB, criterion, cache):
    """
    在GPU上训练网络B的线程函数。

    参数:
    netB (nn.Module): 网络B模型
    deviceB (str): 网络B所在的设备（如 'cuda:1'）
    optimizerB (optim.Optimizer): 网络B的优化器
    criterion (nn.Module): 损失函数
    cache (list): 用于缓存网络A的输出数据
    """
    while True:
        if cache:
            data = cache.pop(0)
            #### 退出线程
            if data == 'END':
                return 
            inputsB, labels = data.to(deviceB)
            optimizerB.zero_grad()
            outputsB = netB(inputsB)
            lossB = criterion(outputsB, labels)
            lossB.backward()
            optimizerB.step()
        

def training(config, criterion, optimizerA, optimizerB):
    """
    训练函数，初始化模型、优化器并启动训练线程。

    参数:
    config (dict): 配置字典，包括设备、训练参数等
    criterion (nn.Module): 损失函数
    optimizerA (optim.Optimizer): 网络A的优化器
    optimizerB (optim.Optimizer): 网络B的优化器
    """
    # 读取配置参数
    deviceA = config['deviceA']
    deviceB = config['deviceB']
    epochs = config['epochs']
    batch_size = config['batch_size']
    nwA_name = config['NetworkA']
    nwB_name = config['NetworkB']
    nwA_para = config['paraA']
    nwB_para = config['paraB']  # 修正为paraB
    lr = config['lr']
    
    # 初始化模型
    netA = getmodel(nwA_name, **nwA_para).to(deviceA)
    netB = getmodel(nwB_name, **nwB_para).to(deviceB)
    
    # 加载数据
    dataloader = load_data(batch_size)
    
    # 创建缓存
    ## 元组形式[(outputB1,labelB1), (outputB2,labelB2),...]
    cache = []

    for epoch in range(epochs):  # 训练多个epoch
        # 创建线程来并行训练
        thread_A = threading.Thread(target=train_A, args=(netA, deviceA, optimizerA, criterion, dataloader, cache))
        thread_B = threading.Thread(target=train_B, args=(netB, deviceB, optimizerB, criterion, cache))
        
        # 启动线程
        thread_A.start()
        thread_B.start()
        
        # 等待线程完成
        thread_A.join()
        thread_B.join()

        print(f'Epoch {epoch+1}/{epochs} completed.')

    print('Training completed.')

# 读取配置文件
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# 开始训练
training(config)
