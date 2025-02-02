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
import torch.multiprocessing as mp
import torch.distributed as dist
from vit_model import vit_base_patch16_224 as create_model
# from utils import train_one_epoch, evaluate
from tqdm import tqdm
import os
import sys
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'

def train_one_epoch(model, optimizer, data_loader, device, epoch, rank):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        
        # print(f'images.shape: {images.shape}')
        sample_num += images.shape[0]

        pred = model(images.to(device),labels.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()
        
        if rank == 0:
            data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                accu_loss.item() / (step + 1),
                                                                                accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, rank):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device),labels.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss
        if rank == 0:
            data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                accu_loss.item() / (step + 1),
                                                                                accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num











def main_worker(rank, world_size, args):
    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    
    # Set the device based on the rank (for each process to use a different GPU)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # Ensure output directories exist only once
    if rank == 0 and not os.path.exists("./whole/weights"):
        os.makedirs("./whole/weights")

    # Setup the tensorboard writer only for rank 0 (main process)
    if rank == 0:
        tb_writer = SummaryWriter(log_dir='./whole/runs')

    # Define the dataset and transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Dataset
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    valset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create DistributedSampler to split the data across GPUs
    train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(valset, num_replicas=world_size, rank=rank)

    # DataLoader
    train_loader = DataLoader(trainset, batch_size=args.batch_size, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(valset, batch_size=args.batch_size, sampler=val_sampler, num_workers=2)

    # Create the model and move it to the device
    model = create_model(img_size=32, patch_size=16, num_classes=args.num_classes).to(device)

    # Wrap the model with DistributedDataParallel
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    # Load weights if provided
    if args.weights != "":
        assert os.path.exists(args.weights), f"Weights file '{args.weights}' not found."
        weights_dict = torch.load(args.weights, map_location=device)
        # Custom weight loading logic (e.g., for positional embedding) here...

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        print(f"Rank {rank} - Missing keys: {missing_keys}")
        print(f"Rank {rank} - Unexpected keys: {unexpected_keys}")

    # Optimizer and scheduler
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    milestones = [80, 120]
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # Training loop
    for epoch in range(args.epochs):
        # Set the epoch for the sampler
        train_sampler.set_epoch(epoch)

        # Train for one epoch
        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device, epoch, rank)

        # Step the scheduler
        scheduler.step()

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, device, epoch, rank)

        if rank == 0:
            # Log to Tensorboard
            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

            # Save the model
            torch.save(model.state_dict(), f"./whole/weights/model-{epoch}.pth")

    # Cleanup
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--weights', type=str, default='', help='Path to initial weights')
    parser.add_argument('--device', default='cuda', help='Device (only used for setting CUDA_VISIBLE_DEVICES)')
    args = parser.parse_args()

    # Set the number of GPUs for DDP
    world_size = torch.cuda.device_count()

    # Launch a process for each GPU
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
