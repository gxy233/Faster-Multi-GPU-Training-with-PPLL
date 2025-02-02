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
from vit_model import vit_base_patch16_224 as create_model
from utils import train_one_epoch, evaluate
import torchvision


def main(args):
    batchsize=128
    imagesize=32
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./whole/weights") is False:
        os.makedirs("./whole/weights")

    tb_writer = SummaryWriter(log_dir='./whole/runs')

    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)

# # 创建数据加载器
#     train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
#                                             shuffle=True, num_workers=2)

#     # 同样的方法应用于测试集
#     testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                         download=True, transform=transform)
#     val_loader = torch.utils.data.DataLoader(testset, batch_size=128,
#                                             shuffle=False, num_workers=2)




    # trainset = torchvision.datasets.STL10(root='./data', split="train", transform=transform, download=True)
    # testset = torchvision.datasets.STL10(root='./data', split="test", transform=transform, download=True)

    trainset = torchvision.datasets.SVHN(root='./data', split="train", transform=transform, download=True)
    testset = torchvision.datasets.SVHN(root='./data', split="test", transform=transform, download=True)
 # 同样的方法应用于测试集
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                                shuffle=True, num_workers=2)
        
    val_loader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                                shuffle=False, num_workers=2)


    model = create_model(img_size=imagesize,patch_size=16,num_classes=args.num_classes).to(device)
    
    
 
    
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
# 假设 weights_dict 是你加载的预训练权重字典
        pos_embed = weights_dict['pos_embed']

        # 从 pos_embed 中移除类标记的位置
        pos_embed_without_cls = pos_embed[:, 1:, :]

        # 重塑 pos_embed 以形成二维网格
        pos_embed_reshaped = pos_embed_without_cls.reshape(1, 768, 14, 14)

        # 对位置嵌入进行插值以适应新的尺寸
        interpolated_pos_embed = F.interpolate(pos_embed_reshaped, size=(2, 2), mode='bilinear', align_corners=False)

        # 将类标记的位置添加回去
        # cls_token = pos_embed[:, :1, :].unsqueeze(2)  # 增加一个维度以匹配 interpolated_pos_embed
        cls_token = pos_embed[:, :1, :]
        
        # print(cls_token.shape)
        # interpolated_pos_embed = torch.cat([cls_token, interpolated_pos_embed.reshape(1, 768, 4)], dim=2)
        interpolated_pos_embed = torch.cat([cls_token.transpose(1,2), interpolated_pos_embed.reshape(1, 768, 4)], dim=2)
        

        # 更新 weights_dict 中的位置嵌入
        weights_dict['pos_embed'] = interpolated_pos_embed.reshape(1, 5, 768)


        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.module.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        # del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            # else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            if k in weights_dict:
                del weights_dict[k]
        
        # 加载模型时忽略不匹配的权重 
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

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

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./whole/weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--lrf', type=float, default=0.01)

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