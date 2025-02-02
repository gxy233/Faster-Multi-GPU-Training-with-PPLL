import os
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from vit_model import vit_base_patch16_224_base
from tqdm import tqdm
from datetime import datetime
import os
import sys
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'

def main(local_rank, args):
    # 设置环境变量
    os.environ['RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())

    # 初始化分布式进程组
    dist.init_process_group(backend='nccl', init_method='env://')

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if os.path.exists("./whole/weights") is False:
        os.makedirs("./whole/weights")

    if local_rank == 0:  # 只在主进程记录日志
        tb_writer = SummaryWriter(log_dir='./whole/runs')

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # 使用 DistributedSampler 来确保每个进程处理不同的子集
    train_sampler = DistributedSampler(trainset)
    val_sampler = DistributedSampler(testset)

    train_loader = DataLoader(trainset, batch_size=128, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(testset, batch_size=128, sampler=val_sampler, num_workers=2)

    model = vit_base_patch16_224_base(img_size=32, patch_size=4, num_classes=args.num_classes).to(device)
    # 将模型转换为 DDP 模式
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)

    milestones = [80, 120]
    lr_decay_rate = 0.1
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=lr_decay_rate)

    loss_function = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        # 设置每个 epoch 的采样器
        train_sampler.set_epoch(epoch)
        model.train()

        accu_loss = torch.zeros(1).to(device)
        accu_num = torch.zeros(1).to(device)
        optimizer.zero_grad()
        sample_num = 0

        train_loader = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_loader):
            images, labels = data
            sample_num += images.shape[0]
            pred = model(images.to(device), labels.to(device))

            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()

            loss = loss_function(pred, labels.to(device))
            loss.backward()
            accu_loss += loss.detach()

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)

            optimizer.step()
            optimizer.zero_grad()

        print(f"[train epoch {epoch}] loss: {accu_loss.item() / (step + 1):.3f}, acc: {accu_num.item() / sample_num:.3f}")

        scheduler.step()

    # 清理进程组
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--data-path', type=str, default="/data/flower_photos")
    parser.add_argument('--model-name', default='', help='create model name')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    # 启动分布式训练
    torch.multiprocessing.spawn(main, args=(opt,), nprocs=torch.cuda.device_count(), join=True)
