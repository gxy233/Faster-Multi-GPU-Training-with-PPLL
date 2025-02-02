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
from vit_model import vit_base_patch16_224_base_back, vit_base_patch16_224_base_front, vit_base_patch16_224_base_mid
from utils import train_one_epoch, evaluate
import torchvision
from torchsummary import summary

from tqdm import tqdm
import sys
class CombinedModel(nn.Module):
    def __init__(self, layer_list, device_list):
        super(CombinedModel, self).__init__()
        if len(layer_list) != len(device_list):
            raise NotImplementedError('Layer list must have the same length as the device list')
        self.layers = nn.ModuleList()
        for i in range(len(layer_list)):
            self.layers.append(layer_list[i].to(device_list[i]))
        self.device_list = device_list
        

    def forward(self, x, label):
        for i in range(len(self.layers)-1):
            x = self.layers[i].forward_features(x,label)
            x = x.to(self.device_list[i+1])
        x = self.layers[-1](x,label)   # 再通过 back 模型
        return x
    
    
def count_folders(path):
    items = os.listdir(path)
    folders = [item for item in items if os.path.isdir(os.path.join(path, item))]
    return len(folders)
    
def main(args):
    print('args:', args)
    device_list = [torch.device(f"cuda:{cuda_num+args.gpu_offset}") for cuda_num in range(args.gpu_num)]
    info_path = './vitpp_res/' + args.dataset
    if not os.path.exists(info_path):
        os.makedirs(info_path)
    dir_idx = count_folders(info_path)
    info_path += '/'
    info_path += str(dir_idx)
    if not os.path.exists(info_path):
        os.makedirs(info_path)
    with open(info_path+'/info', 'w') as f:
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"gpu_num: {args.gpu_num}\n")
    if os.path.exists("./whole/weights") is False:
        os.makedirs("./whole/weights")

    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if args.dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    elif args.dataset == 'SVHN':
        trainset = torchvision.datasets.SVHN(root='./data', split="train",
                                        download=True, transform=transform)
        testset = torchvision.datasets.SVHN(root='./data', split="test",
                                        download=True, transform=transform)
    elif args.dataset == 'STL10':
        trainset = torchvision.datasets.STL10(root='./data', split="train",
                                        download=True, transform=transform)
        testset = torchvision.datasets.STL10(root='./data', split="test",
                                        download=True, transform=transform)

# 创建数据加载器
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                            shuffle=True, num_workers=2)

    # 同样的方法应用于测试集
    
    val_loader = torch.utils.data.DataLoader(testset, batch_size=128,
                                            shuffle=False, num_workers=2)
    
    if args.dataset == 'STL10':
        imsize = 96
    else:
        imsize = 32

    model_list = []
    if args.gpu_num == 2:
        front=vit_base_patch16_224_base_front(img_size=imsize,patch_size=16,depth=6,num_classes=args.num_classes)
        back=vit_base_patch16_224_base_back(img_size=imsize,patch_size=16,depth=6,num_classes=args.num_classes)
        model_list.append(front)
        model_list.append(back)
    elif args.gpu_num == 4:
        front=vit_base_patch16_224_base_front(img_size=imsize,patch_size=16,depth=3,num_classes=args.num_classes)
        back=vit_base_patch16_224_base_back(img_size=imsize,patch_size=16,depth=3,num_classes=args.num_classes)
        model_list.append(front)
        model_list.append(vit_base_patch16_224_base_mid(img_size=imsize,patch_size=16,depth=3,num_classes=args.num_classes))
        model_list.append(vit_base_patch16_224_base_mid(img_size=imsize,patch_size=16,depth=3,num_classes=args.num_classes))
        model_list.append(back)
    
    # model = create_model(img_size=32,patch_size=4,num_classes=args.num_classes).to(device)
    
    model = CombinedModel(model_list, device_list)
    # debug
    # summary(model, input_size=(3,32,32))
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

    train_acc_list = []
    test_acc_list = []
    for epoch in range(args.epochs):
        # train
        model.train()
        accu_loss = torch.zeros(1).to(device_list[-1])  # 累计损失
        accu_num = torch.zeros(1).to(device_list[-1])   # 累计预测正确的样本数
        optimizer.zero_grad()

        sample_num = 0
        train_loader = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_loader):
            
            
            images, labels = data
            
            # print(f'images.shape: {images.shape}')
            sample_num += images.shape[0]

            
            
            pred = model(images.to(device_list[0]),labels.to(device_list[0]))
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device_list[-1])).sum()

            loss = loss_function(pred, labels.to(device_list[-1]))
            loss.backward()
            accu_loss += loss.detach()
            
            

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)

            optimizer.step()
            optimizer.zero_grad()

            # return accu_loss.item() / (step + 1), accu_num.item() / sample_num
        # train_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
        #                                                                                 accu_loss.item() / (step + 1),
        #                                                                                 accu_num.item() / sample_num)
        train_acc_list.append(accu_num.item() / sample_num)
        print(f"[train epoch {epoch}] loss: {accu_loss.item() / (step + 1):.3f}, acc: {accu_num.item() / sample_num:.3f}")
        # eval
        model.eval()
        sample_num = 0
        accu_num = torch.zeros(1).to(device_list[-1])
        test_loader = tqdm(val_loader, file=sys.stdout)
        for step, data in enumerate(test_loader):
            images, labels = data
            # print(f'images.shape: {images.shape}')
            sample_num += images.shape[0]
            pred = model(images.to(device_list[0]),labels.to(device_list[0]))
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device_list[-1])).sum()
        test_acc_list.append(accu_num.item() / sample_num)
        print(f"[train epoch {epoch}]: val acc: {accu_num.item() / sample_num:.3f}")

    with open(info_path+'/train_acc', 'w') as f:
        for i in train_acc_list:
            f.write(f'{i:.3f}\n')
    with open(info_path+'/test_acc', 'w') as f:
        for i in test_acc_list:
            f.write(f'{i:.3f}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--gpu_num', type=int, default=2)
    parser.add_argument('--gpu_offset', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='CIFAR10')

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