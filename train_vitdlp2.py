import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
import torchvision
from models.vit_model import ViTBasic
from models.vit_model import ViTWithEmbed
from models.vit_model import ViTWithClassifier
from models.vit_model import CombineModel

from tqdm import tqdm
import sys
import multiprocessing
import matplotlib.pyplot as plt
import yaml
import time

    
def train_first_module_one_epoch(model, optimizer, dataloader, device, out_cache):
    model.train()
    optimizer.zero_grad()
    data_loader = tqdm(dataloader, file=sys.stdout)
    for _, data in enumerate(data_loader):
        images, labels = data
        model(images.to(device), labels.to(device), out_cache)
        optimizer.step()
        optimizer.zero_grad()
    # finish training
    out_cache.put('END')

def train_mid_module_one_epoch(model, optimizer, device, in_cache, out_cache):
    model.train()
    optimizer.zero_grad()
    while True:
        if not in_cache.empty():
            data = in_cache.get()
            if data == 'END':
                out_cache.put('END')
                break
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.forward(inputs, labels, out_cache)
            optimizer.step()
            optimizer.zero_grad()

def train_last_moudle_one_epoch(model, optimizer, device, in_cache, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    step = 0
    while True:
        if not in_cache.empty():
            data = in_cache.get()
            if data == 'END':
                break
            inputs, labels = data
            inputs=inputs.to(device)
            labels=labels.to(device)
            step += 1
            sample_num += inputs.shape[0]
            pred = model.forward(inputs, labels)
            # print('pred:', pred.shape)
            pred_classes = torch.max(pred, dim=1)[1]
            # accu_num += torch.eq(pred_classes, labels).sum()
            
            loss = loss_function(pred, labels)
            loss.backward()
            accu_loss += loss.detach()
            if not torch.isfinite(loss):
                print('WARNING: infinite loss, ending training ', loss)
                sys.exit(1)
            optimizer.step()
            optimizer.zero_grad()
    print(f"-- finished: epoch {epoch}")
    print("[train epoch {}] loss: {:.3f} acc: {:.3f} ".format(epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num))
    return  accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def eval_first_layer(model, out_evalcache, device, test_loader):
    print('Start eval')
    model.eval()
    data_loader = tqdm(test_loader, file=sys.stdout)
    for _, data in enumerate(data_loader):
        images, labels = data
        model.forward_features(images.to(device), labels.to(device), out_evalcache)
    # finish eval
    # print('first layer eval finish')
    out_evalcache.put('END')
    # print('END put')

@torch.no_grad()
def eval_basic_layer(model, device, in_evalcache, out_evalcache, ln):
    model.eval()
    while True:
        if not in_evalcache.empty():
            try:
                data = in_evalcache.get()
            except Exception:
                pass
            if data == 'END':
                # print(f'basic{ln} end put')
                out_evalcache.put('END')
                break
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.forward_features(inputs, labels, out_evalcache)

@torch.no_grad()
def eval_last_layer(model, device, in_evalcache, epoch):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数

    sample_num = 0
    step = 0
    while True:
        if not in_evalcache.empty():
            try:
                data = in_evalcache.get()
            except Exception:
                break
            # print(data.type)
            if data == 'END':
                break
            inputs, labels = data
            inputs=inputs.to(device)
            labels=labels.to(device)
            step += 1
            sample_num += inputs.shape[0]
            pred = model.forward(inputs, None)
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels).sum()
            loss = loss_function(pred, labels)
            accu_loss += loss
    # finish eval
    print("[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch, accu_loss.item() / (step + 1),accu_num.item() / sample_num))
    print('finish eval')
    return  accu_loss.item() / (step + 1), accu_num.item() / sample_num


def train_first_module(model, out_cache, out_evalcache, device, train_loader, test_loader, lr, mm, wd, max_ep, path=None):
    # print('device:', device)
    # model = model.to(device)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=lr, momentum=mm, weight_decay=wd)
    
    total_time = 0
    for ep in range(max_ep):
        # start_time = time.time()
        train_first_module_one_epoch(model, optimizer, train_loader, device, out_cache)
        # total_time += time.time() - start_time
        # eval_first_layer(model, out_evalcache, device, test_loader)
    # with open(path, 'a') as f:
    #     f.write(f'Avg time for an epoch: {total_time / max_ep:.3f}s')

def train_basic_module(model, in_cache, in_evalcache, out_cache, out_evalcache, device, lr, mm, wd, max_ep, ln):
    # print('device:', device)
    # model = model.to(device)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=lr, momentum=mm, weight_decay=wd)
    for ep in range(max_ep):
        # start_time = time.time()
        train_mid_module_one_epoch(model, optimizer, device, in_cache, out_cache)
        # print(f'Epoch{ep} basic layer{layer_idx} time: {time.time() - start_time:.2f}')
        # eval_basic_layer(model, device, in_evalcache, out_evalcache, ln)


def record_res(ep_list, train_acc_list, test_acc_list, train_loss_list, test_loss_list, path):
    # plot accuracy curve
    # print('plot res:', path)
    # plt.figure()
    # plt.plot(ep_list, train_acc_list, '.-', color='r', label='train')
    # plt.plot(ep_list, test_acc_list, '.-', color='b', label='test')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy Curve')
    # plt.legend()
    # plt.savefig(path+'/acc_curve.png')
    # plot loss curve
    # plt.figure()
    # plt.plot(ep_list, train_loss_list, '.-', color='r', label='train')
    # plt.plot(ep_list, test_loss_list, '.-', color='b', label='test')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Loss Curve')
    # plt.legend()
    # plt.savefig(path+'/loss_curve.png')

    # record train_acc
    with open(path+'/train_acc', 'w') as f:
        for i in train_acc_list:
            f.write(f'{i:.3f}\n')
    # record test_acc
    with open(path+'/test_acc', 'w') as f:
        for i in test_acc_list:
            f.write(f'{i:.3f}\n')

def train_last_module(model, in_cache, in_evalcache, device, lr, mm, wd, max_ep, path=None):
    # print('device:', device)
    # model = model.to(device)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=lr, momentum=mm, weight_decay=wd)

    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []
    ep_list = [i+1 for i in range(max_ep)]
    for epoch in range(max_ep):
        # start_time = time.time()
        train_loss, train_acc =train_last_moudle_one_epoch(model, optimizer, device, in_cache, epoch)
        # print(f'Epoch{epoch} last layer time: {time.time() - start_time:.2f}')
        eval_loss, eval_acc = eval_last_layer(model, device, in_evalcache, epoch)
        test_acc_list.append(eval_acc)
        test_loss_list.append(eval_loss)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
    # record_res(ep_list, train_acc_list, test_acc_list, train_loss_list, test_loss_list, path)
        


def get_model(dataset, aug_depth, layer_num,device0,device1):
    
    first_module=CombineModel(part='front',numbasis=5,device=device0)
    last_module=CombineModel(part='back',numbasis=5,device=device1)
    
    return first_module,last_module
    # if dataset == 'CIFAR10':
    #     first_layer = ViTWithEmbed(img_size=32, num_classes=10, augdepth=aug_depth[0])
    #     last_layer = ViTWithClassifier(num_classes=10, augdepth=aug_depth[-1])
    #     mid_layers = [ViTBasic(num_classes=10, augdepth=aug_depth[i+1]) for i in range(layer_num - 2)]
    #     return first_layer, last_layer, mid_layers
    # elif dataset == 'STL10':
    #     first_layer = ViTWithEmbed(img_size=96, num_classes=10, augdepth=aug_depth[0])
    #     last_layer = ViTWithClassifier(num_classes=10, augdepth=aug_depth[-1])
    #     mid_layers = [ViTBasic(num_classes=10, augdepth=aug_depth[i+1]) for i in range(layer_num - 2)]
    #     return first_layer, last_layer, mid_layers
    # elif dataset == 'SVHN':
    #     first_layer = ViTWithEmbed(img_size=32, num_classes=10, augdepth=aug_depth[0])
    #     last_layer = ViTWithClassifier(num_classes=10, augdepth=aug_depth[-1])
    #     mid_layers = [ViTBasic(num_classes=10, augdepth=aug_depth[i+1]) for i in range(layer_num - 2)]
    #     return first_layer, last_layer, mid_layers
    # else:
    #     raise NotImplementedError("Only support dataset: CIFAR10, STL10, SVHN")

def count_folders(path):
    items = os.listdir(path)
    folders = [item for item in items if os.path.isdir(os.path.join(path, item))]
    return len(folders)

def train(cfg):
    # get config
    with open(cfg, 'r') as file1:
        cfg_data = yaml.load(file1, Loader=yaml.FullLoader)
    training_params = cfg_data['training_params']
    model_params = cfg_data['model_params']
    print('Start training', model_params['dataset'])
    # create folder for exp info and result
    # info_path = './vit_res/' + model_params['dataset']
    # if not os.path.exists(info_path):
    #     os.makedirs(info_path)
    # dir_idx = count_folders(info_path)
    # info_path += '/'
    # info_path += str(dir_idx)
    # if not os.path.exists(info_path):
    #     os.makedirs(info_path)
    # with open(info_path+'/info', 'w') as f:
    #     f.write(f"dataset: {model_params['dataset']}\n")
    #     f.write(f"layer num: {training_params['layer_num']}\n")
    #     f.write(f"gpu_num: {training_params['gpu_num']}\n")
    #     f.write(f"aug_depth: {model_params['aug_list']}\n")

    # # do some preprocessing
    # if os.path.exists("./whole/weights") is False:
    #     os.makedirs("./whole/weights")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # dataset
    ds_name = model_params['dataset']
    if ds_name == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif ds_name == 'STL10':
        trainset = torchvision.datasets.STL10(root='./data', split="train", transform=transform, download=True)
        testset = torchvision.datasets.STL10(root='./data', split="test", transform=transform, download=True)
    elif ds_name == 'SVHN':
        trainset = torchvision.datasets.SVHN(root='./data', split="train", transform=transform, download=True)
        testset = torchvision.datasets.SVHN(root='./data', split="test", transform=transform, download=True)
    else:
        raise NotImplementedError('Only support dataset: CIFAR10, STL10, SVHN')
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=training_params['train_bs'],
                                            shuffle=True, num_workers=2)
    
    test_loader = torch.utils.data.DataLoader(testset, batch_size=training_params['test_bs'],
                                              shuffle=False, num_workers=2)
    
    # caches
    cache_list = [multiprocessing.Queue() for _ in range(training_params['layer_num'] - 1)]
    evalcache_list = [multiprocessing.Queue() for _ in range(training_params['layer_num'] - 1)]
    # device
    device_list = [torch.device(f"cuda:{cuda_num+training_params['gpu_offset']}") for cuda_num in range(training_params['gpu_num'])]
    # aug_depth
    aug_depth = model_params['aug_list']

    # create models
    first_module, last_module= get_model(model_params['dataset'], aug_depth, training_params['layer_num'],device_list[0],device_list[-1])

    # multiprocessing
    process_first = multiprocessing.Process(target=train_first_module, 
                                            args=(first_module, cache_list[0], 
                                                  evalcache_list[0], device_list[0],
                                                  train_loader, test_loader,
                                                  training_params['lr'],
                                                  training_params['momentum'],
                                                  training_params['weight_decay'],
                                                  training_params['max_epoch'],
                                                  ))
    
    process_last = multiprocessing.Process(target=train_last_module,
                                           args=(last_module, cache_list[0], 
                                                 evalcache_list[-1], device_list[-1],
                                                 training_params['lr'],
                                                 training_params['momentum'],
                                                 training_params['weight_decay'],
                                                 training_params['max_epoch'],
                                                 ))
    # process_mid_list = []
    # for i in range(training_params['layer_num'] - 2):
    #     # how many layers per gpu
    #     layer_per_gpu = training_params['layer_num'] // training_params['gpu_num']
    #     # device index
    #     device_idx = (i+1) // layer_per_gpu
    #     process_mid_list.append(multiprocessing.Process(target=train_basic_module,
    #                                                     args=(mid_layers[i],
    #                                                           cache_list[i],
    #                                                           evalcache_list[i],
    #                                                           cache_list[i+1],
    #                                                           evalcache_list[i+1],
    #                                                           device_list[device_idx],
    #                                                           training_params['lr'],
    #                                                           training_params['momentum'],
    #                                                           training_params['weight_decay'],
    #                                                           training_params['max_epoch'],
    #                                                           i+1)))
        
    # start processes
    process_first.start()
    # for p in process_mid_list:
    #     p.start()
    process_last.start()

    # wait the processes to finish
    process_first.join()
    # for p in process_mid_list:
    #     p.join()
    process_last.join()

    # clear up the queues
    for q in cache_list:
        q.close()
        q.join_thread()
    for q in evalcache_list:
        q.close()
        q.join_thread()

    # All finish
    print('Finish training')

    
def test_dataset():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # test STL-10
    # trainset = torchvision.datasets.STL10(root='./data', split="train", transform=transform, download=True)
    # testset = torchvision.datasets.STL10(root='./data', split="test", transform=transform, download=True)
    # test svhn
    # trainset = torchvision.datasets.SVHN(root='./data', split="train", transform=transform, download=True)
    # testset = torchvision.datasets.SVHN(root='./data', split="test", transform=transform, download=True)
    # test imgnet
    # not publicly available, download need ~200G
    trainset = torchvision.datasets.ImageNet(root='./data', split='train', transform=transform, download=True)
    testset = torchvision.datasets.ImageNet(root='./data', split='val', transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                            shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128,
                                              shuffle=False, num_workers=2)
    print('train info:')
    print(len(trainset))
    for _, data in enumerate(train_loader):
        img, labels = data
        print(img.shape)
        # print(labels.min())
        break
    print('test info:')
    print(len(testset))
    for _, data in enumerate(test_loader):
        img, labels = data
        print(img.shape)
        # print(labels.max())
        break
    

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    cfg = 'configs/vit_dlp2.yaml'
    train(cfg)
    # test_dataset()