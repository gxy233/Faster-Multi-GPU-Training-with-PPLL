
import os
import sys
if os.path.join(os.path.dirname(__file__),r'../..') not in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__),r'../..'))
    
    
sys.path.append('/home/chengqixu/gxy/temp_name1')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml
import threading
# from models.models import getmodel, NetworkA, NetworkB
# from dataloader.load_data import load_data
from train_unit import train_front, train_back, train_mid
from util.log_gpu_memory import log_gpu_memory
import argparse
import multiprocessing
from multiprocessing import Manager
from multiprocessing import Queue

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')


# def training(cfg):
#     """
#     训练函数，初始化模型、优化器并启动训练线程。

#     参数:
#     config (dict): 配置字典，包括设备、训练参数等
#     criterion (nn.Module): 损失函数
#     optimizerA (optim.Optimizer): 网络A的优化器
#     optimizerB (optim.Optimizer): 网络B的优化器
#     """
#     # 读取配置文件
#     with open(cfg, 'r') as file1:
#         cfg_data=  yaml.load(file1,Loader=yaml.FullLoader)
    
#     # 划分块数
#     k = cfg_data['k']
    
#     # k=2 则没有mid层
#     if k==2:
#         front_args = cfg_data['front']
#         back_args = cfg_data['back']
#         comargs = cfg_data['common']

            
#         # 创建缓存
#         ## 元组形式[(outputB1,labelB1), (outputB2,labelB2),...]
 
#         manager = Manager()

#         # 创建可以共享的 cache
#         cache = manager.list()
#         evacache = manager.list()
    
        
        
#         front_args = argparse.Namespace(**front_args)
#         back_args = argparse.Namespace(**back_args)
#         comargs = argparse.Namespace(**comargs)
        
#         rd_txt=f'rd/{comargs.exp_name}.txt'
        

#         # 创建进程来并行训练
#         process_front = multiprocessing.Process(target=train_front, args=(front_args, comargs, cache, evacache))
#         process_back = multiprocessing.Process(target=train_back, args=(back_args, comargs, cache, evacache))
#         process_rd = multiprocessing.Process(target=log_gpu_memory, args=(rd_txt,))
#         process_rd.daemon = True  # 设置为守护进程

#         # 启动进程
#         process_front.start()
#         process_back.start()
#         process_rd.start()

#         # 等待进程完成
#         process_front.join()
#         process_back.join()

 
    
#     # 网络由front，mid，back组成
#     else:

#         front_args = cfg_data['front']
#         mid_args = cfg_data['mid']
#         back_args = cfg_data['back']
#         comargs = cfg_data['common']

            
#         # 创建缓存
#         ## 元组形式[(outputB1,labelB1), (outputB2,labelB2),...]


#         manager = Manager()


#         # 共享的缓存结构
#         startcache = manager.list()
#         mid_cache = manager.list([manager.list() for _ in range(k-2)])  # 嵌套共享列表
#         startevacache = manager.list()
#         midevacache = manager.list([manager.list() for _ in range(k-2)])
        
#         front_args = argparse.Namespace(**front_args)
#         back_args = argparse.Namespace(**back_args)
#         comargs = argparse.Namespace(**comargs)
        
#         rd_txt=f'rd/{comargs.exp_name}.txt'
        
        
    

#     # 创建进程来并行训练
#     process_front = multiprocessing.Process(target=train_front, args=(front_args, comargs, startcache, startevacache))
#     process_mid_pool = []

#     for i in range(k - 2):
#         mid_args_ = argparse.Namespace(**mid_args[f'part{i+1}'])

#         if i == 0:
#             process_mid = multiprocessing.Process(target=train_mid, args=(mid_args_, comargs, startcache, mid_cache[i], startevacache, midevacache[i]))
#             process_mid_pool.append(process_mid)
#         else:
#             process_mid = multiprocessing.Process(target=train_mid, args=(mid_args_, comargs, mid_cache[i-1], mid_cache[i], midevacache[i-1], midevacache[i]))
#             process_mid_pool.append(process_mid)

#     process_back = multiprocessing.Process(target=train_back, args=(back_args, comargs, mid_cache[-1], midevacache[-1]))
#     process_rd = multiprocessing.Process(target=log_gpu_memory, args=(rd_txt,))
#     process_rd.daemon = True  # 设置为守护进程

#     # 启动进程
#     process_front.start()
#     for process_mid in process_mid_pool:
#         process_mid.start()
#     process_back.start()
#     process_rd.start()

#     # 等待进程完成
#     process_front.join()
#     for process_mid in process_mid_pool:
#         process_mid.join()
#     process_back.join()

#     print('Training completed.')






def training(cfg):
    """
    训练函数，初始化模型、优化器并启动训练线程。

    参数:
    config (dict): 配置字典，包括设备、训练参数等
    criterion (nn.Module): 损失函数
    optimizerA (optim.Optimizer): 网络A的优化器
    optimizerB (optim.Optimizer): 网络B的优化器
    """
    # 读取配置文件
    with open(cfg, 'r') as file1:
        cfg_data = yaml.load(file1, Loader=yaml.FullLoader)
    
    # 划分块数
    k = cfg_data['k']
    
    # k=2 则没有mid层
    if k == 2:
        front_args = cfg_data['front']
        back_args = cfg_data['back']
        comargs = cfg_data['common']

        # 创建队列用于进程间通信
        cache = multiprocessing.Queue()
        evacache = multiprocessing.Queue()
        
        front_args = argparse.Namespace(**front_args)
        back_args = argparse.Namespace(**back_args)
        comargs = argparse.Namespace(**comargs)
        
        rd_txt = f'rd/{comargs.exp_name}.txt'

        # 创建进程来并行训练
        process_front = multiprocessing.Process(target=train_front, args=(front_args, comargs, cache, evacache))
        process_back = multiprocessing.Process(target=train_back, args=(back_args, comargs, cache, evacache))
        process_rd = multiprocessing.Process(target=log_gpu_memory, args=(rd_txt,))
        process_rd.daemon = True  # 设置为守护进程

        # 启动进程
        process_front.start()
        process_back.start()
        process_rd.start()

        # 等待进程完成
        process_front.join()
        process_back.join()

    # 网络由front，mid，back组成
    else:
        front_args = cfg_data['front']
        mid_args = cfg_data['mid']
        back_args = cfg_data['back']
        comargs = cfg_data['common']

        # 创建队列用于进程间通信
        startcache = multiprocessing.Queue()
        mid_cache = [multiprocessing.Queue() for _ in range(k - 2)]  # 为每个mid创建独立的队列
        startevacache = multiprocessing.Queue()
        midevacache = [multiprocessing.Queue() for _ in range(k - 2)]
        
        front_args = argparse.Namespace(**front_args)
        back_args = argparse.Namespace(**back_args)
        comargs = argparse.Namespace(**comargs)
        
        rd_txt = f'rd/{comargs.exp_name}.txt'

        # 创建进程来并行训练
        process_front = multiprocessing.Process(target=train_front, args=(front_args, comargs, startcache, startevacache))
        process_mid_pool = []

        for i in range(k - 2):
            mid_args_ = argparse.Namespace(**mid_args[f'part{i+1}'])

            if i == 0:
                process_mid = multiprocessing.Process(target=train_mid, args=(mid_args_, comargs, startcache, mid_cache[i], startevacache, midevacache[i]))
                process_mid_pool.append(process_mid)
            else:
                process_mid = multiprocessing.Process(target=train_mid, args=(mid_args_, comargs, mid_cache[i-1], mid_cache[i], midevacache[i-1], midevacache[i]))
                process_mid_pool.append(process_mid)

        process_back = multiprocessing.Process(target=train_back, args=(back_args, comargs, mid_cache[-1], midevacache[-1]))
        process_rd = multiprocessing.Process(target=log_gpu_memory, args=(rd_txt,))
        process_rd.daemon = True  # 设置为守护进程

        # 启动进程
        process_front.start()
        for process_mid in process_mid_pool:
            process_mid.start()
        process_back.start()
        process_rd.start()

        # 等待进程完成
        process_front.join()
        for process_mid in process_mid_pool:
            process_mid.join()
        process_back.join()

    print('Training completed.')





import argparse
if __name__ == "__main__":
    
 
        
        
  
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", help="实验", default="k2_resnet32")
    args = parser.parse_args()
    
    exp = args.exp
    cfg = f'configs/{exp}.yaml'
    # if exp == 'k2_vit_224_p16':
    #     cfg = 'configs/k2_vit_224_p16.yaml'
       
    # elif exp =='k4_vit_224_p16':
    #     cfg = 'configs/k4_vit_224_p16.yaml'

    
    # if exp=='k2_resnet32':
    #     cfg = 'configs/k2_resnet32.yaml'
        
        
    # if exp=='k2_resnet32_t':
    #     cfg = 'configs/k2_resnet32_t.yaml'
    

    # 开始训练
    training(cfg)