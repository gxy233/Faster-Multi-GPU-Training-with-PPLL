
import os
import sys
if os.path.join(os.path.dirname(__file__),r'../..') not in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__),r'../..'))
    
    
sys.path.append('/data/share/torch_projects/guoxiuyuan/temp_name')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml
import threading
# from models.models import getmodel, NetworkA, NetworkB
# from dataloader.load_data import load_data
from train_unit import train_front, train_back, train_mid


        

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
        cfg_data=  yaml.load(file1,Loader=yaml.FullLoader)
    
    
    k = cfg_data['k']
    
    if k==2:
        front_args = cfg_data['front']
        back_args = cfg_data['back']
        comargs = cfg_data['common']

            
        # 创建缓存
        ## 元组形式[(outputB1,labelB1), (outputB2,labelB2),...]
        cache = []
        evacache = []

        
        front_args = argparse.Namespace(**front_args)
        back_args = argparse.Namespace(**back_args)
        comargs = argparse.Namespace(**comargs)
        
        
        # 创建线程来并行训练
        thread_front = threading.Thread(target=train_front, args=(front_args, comargs, cache,evacache))
        thread_back = threading.Thread(target=train_back, args=(back_args, comargs, cache, evacache))
        
        # 启动线程
        thread_front.start()
        thread_back.start()
        
        # 等待线程完成
        thread_front.join()
        thread_back.join()
        
    else:
        front_args = cfg_data['front']
        mid_args = cfg_data['mid']
        back_args = cfg_data['back']
        comargs = cfg_data['common']

            
        # 创建缓存
        ## 元组形式[(outputB1,labelB1), (outputB2,labelB2),...]
        startcache = []
        startevacache= []
        mid_cache = [[] for _ in range(k-2)]
        midevacache = [[] for _ in range(k-2)]

        
        front_args = argparse.Namespace(**front_args)
        back_args = argparse.Namespace(**back_args)
        comargs = argparse.Namespace(**comargs)
        
        
        # 创建线程来并行训练
        thread_front = threading.Thread(target=train_front, args=(front_args, comargs, startcache, startevacache))
        thread_mid_pool = []
        for i in range(k-2):
            mid_args_ = argparse.Namespace(**mid_args[f'part{i+1}'])
            
            if i==0:
                thread_mid = threading.Thread(target=train_mid, args=(mid_args_, comargs, startcache, mid_cache[i], startevacache, midevacache[i]))
                thread_mid_pool.append(thread_mid)
                
            else:
                thread_mid = threading.Thread(target=train_mid, args=(mid_args_, comargs, mid_cache[i-1], mid_cache[i],midevacache[i-1], midevacache[i]))
                thread_mid_pool.append(thread_mid)
                
            
        thread_back = threading.Thread(target=train_back, args=(back_args, comargs, mid_cache[-1], midevacache[-1]))
        
        # 启动线程
        thread_front.start()
        for thread_mid in thread_mid_pool:
            thread_mid.start()
        thread_back.start()
        
        # 等待线程完成
        thread_front.join()
        for thread_mid in thread_mid_pool:
            thread_mid.join()
        thread_back.join()
        
    print('Training completed.')

import argparse
if __name__ == "__main__":
    
 
        
        
  
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", help="实验", default="k2_vit_224_p16")
    args = parser.parse_args()
    
    exp = args.exp
    cfg = None
    if exp == 'k2_vit_224_p16':
        cfg = 'configs/k2_vit_224_p16.yaml'
       
    elif exp =='k4_vit_224_p16':
        cfg = 'configs/k4_vit_224_p16.yaml'

    # 开始训练
    training(cfg)