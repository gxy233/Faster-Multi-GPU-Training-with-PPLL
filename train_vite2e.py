import os
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import torch.distributed as dist
from vit_model import vit_base_patch16_224_base_back, vit_base_patch16_224_base_front,vit_base_patch16_224_base
from utils import train_one_epoch, evaluate
import torchvision
import time

from tqdm import tqdm
import sys

    
from datetime import datetime
    
    
def main(args):
    device = torch.device("cuda:6")
    img_size = args.img_size
    patch_size= args.patch_size
    
    if os.path.exists("./whole/weights") is False:
        os.makedirs("./whole/weights")


    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# 创建数据加载器
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                            shuffle=True, num_workers=2)

    # 同样的方法应用于测试集
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=128,
                                            shuffle=False, num_workers=2)

    
    

    # device_ids = [0,1,2,3]
    # device_ids = [0,1]
    
    model = vit_base_patch16_224_base(img_size=img_size,patch_size=patch_size,num_classes=args.num_classes).to(device)
    # model = nn.DataParallel(model, device_ids=device_ids)
    
    # model = CombinedModel(front, back, device1,device2)

    
    
    pg = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # optimizer = optim.AdamW(pg, lr=0.01, weight_decay=1e-4)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    milestones = [80, 120]  # 在第 80 和 120 个 epoch 时衰减学习率
    lr_decay_rate = 0.1

# 创建学习率调度器
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=lr_decay_rate)
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    loss_function = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        # train
        start_time = time.time()
        model.train()
        accu_loss = torch.zeros(1).to(device)  # 累计损失
        accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
        optimizer.zero_grad()

        sample_num = 0
        train_loader = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_loader):
            
            
            images, labels = data
            
            # print(f'images.shape: {images.shape}')
            sample_num += images.shape[0]

            
            
            pred = model(images.to(device),labels.to(device))
            
            # current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            # # 打印带毫秒时间戳的消息
            # print(f'[start bp {current_time}] in a batch')
            
            
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()

            loss = loss_function(pred, labels.to(device))
            loss.backward()
            accu_loss += loss.detach()
            
            
            # current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            # # 打印带毫秒时间戳的消息
            # print(f'[end bp {current_time}] in a batch')
            

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)

            optimizer.step()
            optimizer.zero_grad()
            
            
            # current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            # # 打印带毫秒时间戳的消息
            # print(f'[end updatepara {current_time}] in a batch')
            # return accu_loss.item() / (step + 1), accu_num.item() / sample_num
        # train_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
        #                                                                                 accu_loss.item() / (step + 1),
        #                                                                                 accu_num.item() / sample_num)
        print(f'Epoch{epoch} time: {time.time() - start_time:.2f}')
        print(f"[train epoch {epoch}] loss: {accu_loss.item() / (step + 1):.3f}, acc: {accu_num.item() / sample_num:.3f}")
        # exit(0)
        
        sample_num_val=0
        accu_loss_val = torch.zeros(1).to(device)  # 累计损失
        accu_num_val = torch.zeros(1).to(device)   # 累计预测正确的样本数
        for step, data in enumerate(val_loader):
            
            
            images, labels = data
            
            # print(f'images.shape: {images.shape}')
            sample_num_val += images.shape[0]

            
            
            pred_val = model(images.to(device),labels.to(device))
            
            # current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            # # 打印带毫秒时间戳的消息
            # print(f'[start bp {current_time}] in a batch')
            
            
            pred_classes_val = torch.max(pred_val, dim=1)[1]
            accu_num_val += torch.eq(pred_classes_val, labels.to(device)).sum()

            

  
        # train_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
        #                                                                                 accu_loss.item() / (step + 1),
        #                                                                                 accu_num.item() / sample_num)
        print(f"[val epoch {epoch}] acc: {accu_num_val.item() / sample_num_val:.3f}")
            
        
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--img_size', type=float, default=32)
    parser.add_argument('--patch_size', type=float, default=16)
 

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="/data/flower_photos")
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    
    
    
    main(opt)