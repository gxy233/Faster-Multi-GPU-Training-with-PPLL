import os
import sys
if os.path.join(os.path.dirname(__file__),r'../..') not in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__),r'../..'))


import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision
import torch.nn.functional as F

from util.utils_dp import  train_one_epoch_front,train_one_epoch_back,train_one_epoch_mid, evaluate_front, evaluate_mid, evaluate_back
from dataloader.load_data import get_dataloader
from util.get_optimizer import get_optimizer
from util.create_model import create_model
from util.adjustlr import adjust_learning_rate
#### 第一个模块
def train_front(args, comargs, cache, evacache):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    os.makedirs(f"./weights/{comargs.exp_name}/{args.partname}",exist_ok=True)
   

    train_loader,val_loader = get_dataloader(comargs.dataset,batch_size=comargs.batch_size)
    
    
#### 初始化front模型


    # print(f'args.aux_depth_list -1 {args.aux_depth_list}')
    model = create_model(model_name=comargs.model_name, part='front', args=args, comargs=comargs)

    
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=args.device)
            
        # 加载模型时忽略不匹配的权重 
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        print("front Missing keys:", missing_keys)
        print("front Unexpected keys:", unexpected_keys)
        # exit(0)
    

            
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(pg,comargs.optimizer,comargs.lr,comargs.momentum,comargs.weight_decay)
    
    # # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    
    milestones = comargs.milestones  # 在第 80 和 120 个 epoch 时衰减学习率
    lr_decay_rate = comargs.lr_decay_rate
    
# 创建学习率调度器
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=lr_decay_rate)
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(comargs.epochs):
        
        
        if comargs.adjustlr:
            adjust_learning_rate(comargs=comargs, optimizer=optimizer, epoch=epoch + 1)
        
       
        # train
        pred = train_one_epoch_front(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    cache=cache,
                                    split_size=comargs.split_size,
                                    inc_input=comargs.inc_input # resnet 实验需要原始img计算contrast loss， vit不需要
                                    )
        if comargs.scheduler:
            scheduler.step()
        print(f'train_one_epoch_front finish')
        # exit(0)
          # validate
        pred = evaluate_front(model=model,
                            data_loader=val_loader,
                            device=args.device,
                            evacache=evacache,
                            inc_input=comargs.inc_input
                            )
        print(f'evaluate_front finish')
        # exit(0)
       
        # torch.save(model.state_dict(), f"./weights/{comargs.exp_name}/{args.partname}/model-{epoch}.pth")




#### 中间模块
def train_mid(args, comargs, in_cache, out_cache, in_evacache, out_evacache):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    os.makedirs(f"./weights/{comargs.exp_name}/{args.partname}",exist_ok=True)


    
#### 初始化mid模型
    model = create_model(model_name=comargs.model_name, part='mid', args=args, comargs=comargs)





    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=args.device)


        
        # 加载模型时忽略不匹配的权重 
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        print("mid Missing keys:", missing_keys)
        print("mid Unexpected keys:", unexpected_keys)

   

   
                
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(pg,comargs.optimizer,comargs.lr,comargs.momentum,comargs.weight_decay)

    milestones = comargs.milestones  # 在第 80 和 120 个 epoch 时衰减学习率
    lr_decay_rate = comargs.lr_decay_rate
    
# 创建学习率调度器
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=lr_decay_rate)
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(comargs.epochs):
        # train
        pred = train_one_epoch_mid(model=model,
                                    optimizer=optimizer,
                                    device=device,
                                    in_cache=in_cache,
                                    out_cache=out_cache)

        scheduler.step()

          # validate
        pred = evaluate_mid(model=model,
                            device=device,
                            in_evacache=in_evacache,
                            out_evacache=out_evacache)

        # torch.save(model.state_dict(), f"./weights/{comargs.exp_name}/{args.partname}/model-{epoch}.pth")






#### 最后一个模块
def train_back(args, comargs, cache, evacache):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    os.makedirs(f"./weights/{comargs.exp_name}/{args.partname}",exist_ok=True)

    log_dir=f'./runs/{comargs.exp_name}'
    os.makedirs(log_dir,exist_ok=True)
    
    tb_writer = SummaryWriter(log_dir=log_dir)



#### 初始化back模型
    model = create_model(model_name=comargs.model_name, part='back', args=args, comargs=comargs)


    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=args.device)

            
        # 加载模型时忽略不匹配的权重 
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        print("back Missing keys:", missing_keys)
        print("back Unexpected keys:", unexpected_keys)
        


    pg = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = get_optimizer(pg,comargs.optimizer,comargs.lr,comargs.momentum,comargs.weight_decay)
    
    # # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine

    # 创建学习率调度器
    
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=comargs.milestones, gamma=comargs.lr_decay_rate)
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(comargs.epochs):
        
        # print(f'train_back cache len: {len(cache)}')
        if comargs.adjustlr:
            adjust_learning_rate(comargs=comargs, optimizer=optimizer, epoch=epoch + 1)

            
        # train
        train_loss, train_acc = train_one_epoch_back(model=model,
                                                optimizer=optimizer,
                                                device=device,
                                                epoch=epoch,
                                                cache=cache,
                                                inc_input=comargs.inc_input # resnet 实验需要原始img计算contrast loss， vit不需要
                                                )
        if comargs.scheduler:
            scheduler.step()
        print(f'train_one_epoch_back finish')
        # exit(0)
        # validate
        val_loss, val_acc = evaluate_back(model=model,
                                     device=device,
                                     epoch=epoch,
                                     evacache=evacache,
                                     inc_input=comargs.inc_input
                                     )
        
        print(f'evaluate_back finish')
        # exit(0)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        print(f'Train:\nloss:{train_loss}\nacc:{train_acc}\n\nVal:\nloss:{val_loss}\nacc:{val_acc}')

        # torch.save(model.state_dict(), f"./weights/{comargs.exp_name}/{args.partname}/model-{epoch}.pth")


