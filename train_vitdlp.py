import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
import torchvision
from models.vit_model import ViTBasic
from models.vit_model import ViTWithEmbed
from models.vit_model import ViTWithClassifier
from tqdm import tqdm
import sys
import multiprocessing
import matplotlib.pyplot as plt
import yaml
import time
torch.multiprocessing.set_sharing_strategy('file_system')


    
def train_first_layer_one_epoch(model, optimizer, dataloader, device, out_cache):
    model.train()
    # print('11')
    optimizer.zero_grad()
    # print('22')
    data_loader = tqdm(dataloader)
    # print('33')
    for _, data in enumerate(data_loader):
        # print('44')
        images, labels = data
        # print('start forward one ep')
        model.forward_features(images.to(device), labels.to(device), out_cache)
        # print('end forward one ep')
        optimizer.step()
        optimizer.zero_grad()
    # finish training
    out_cache.put('END')

def train_basic_layer_one_epoch(model, optimizer, device, in_cache, out_cache):
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
            model.forward_features(inputs, labels, out_cache)
            optimizer.step()
            optimizer.zero_grad()

def train_last_layer_one_epoch(model, optimizer, device, in_cache, epoch):
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
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels).sum()
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
    data_loader = tqdm(test_loader)
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


def train_first_layer(model, out_cache, out_evalcache, device, train_loader, test_loader, lr, mm, wd, max_ep, path):
    # print('device:', device)
    model = model.to(device)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=lr, momentum=mm, weight_decay=wd)
    for ep in range(max_ep):
        # start_time = time.time()
        # print('start forward')
        train_first_layer_one_epoch(model, optimizer, train_loader, device, out_cache)
        # print('end forward')
        # total_time += time.time() - start_time
        eval_first_layer(model, out_evalcache, device, test_loader)
    # with open(path, 'a') as f:
    #     f.write(f'Avg time for an epoch: {total_time / max_ep:.3f}s')

def train_basic_layer(model, in_cache, in_evalcache, out_cache, out_evalcache, device, lr, mm, wd, max_ep, ln):
    # print('device:', device)
    model = model.to(device)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=lr, momentum=mm, weight_decay=wd)
    for ep in range(max_ep):
        # start_time = time.time()
        train_basic_layer_one_epoch(model, optimizer, device, in_cache, out_cache)
        # print(f'Epoch{ep} basic layer{layer_idx} time: {time.time() - start_time:.2f}')
        eval_basic_layer(model, device, in_evalcache, out_evalcache, ln)


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

def record_one_ep(path, train_acc, test_acc):
    with open(path+'/info_train', 'a') as f:
        f.write(f'{train_acc:.3f}\n')
    with open(path+'/info_test', 'a') as f:
        f.write(f'{test_acc:.3f}\n')

def train_last_layer(model, in_cache, in_evalcache, device, lr, mm, wd, max_ep, path):
    # print('device:', device)
    model = model.to(device)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=lr, momentum=mm, weight_decay=wd)

    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []
    ep_list = [i+1 for i in range(max_ep)]
    for epoch in range(max_ep):
        # start_time = time.time()
        train_loss, train_acc =train_last_layer_one_epoch(model, optimizer, device, in_cache, epoch)
        # print(f'Epoch{epoch} last layer time: {time.time() - start_time:.2f}')
        eval_loss, eval_acc = eval_last_layer(model, device, in_evalcache, epoch)
        test_acc_list.append(eval_acc)
        test_loss_list.append(eval_loss)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        record_one_ep(path, train_acc, eval_acc)
    record_res(ep_list, train_acc_list, test_acc_list, train_loss_list, test_loss_list, path)
        


def get_model(dataset, aug_depth, layer_num):
    if dataset == 'CIFAR10':
        first_layer = ViTWithEmbed(img_size=32, num_classes=10, augdepth=aug_depth[0])
        last_layer = ViTWithClassifier(num_classes=10, augdepth=aug_depth[-1])
        mid_layers = [ViTBasic(num_classes=10, augdepth=aug_depth[i+1]) for i in range(layer_num - 2)]
        return first_layer, last_layer, mid_layers
    elif dataset == 'STL10':
        first_layer = ViTWithEmbed(img_size=96, num_classes=10, augdepth=aug_depth[0])
        last_layer = ViTWithClassifier(num_classes=10, augdepth=aug_depth[-1])
        mid_layers = [ViTBasic(num_classes=10, augdepth=aug_depth[i+1]) for i in range(layer_num - 2)]
        return first_layer, last_layer, mid_layers
    elif dataset == 'SVHN':
        first_layer = ViTWithEmbed(img_size=32, num_classes=10, augdepth=aug_depth[0])
        last_layer = ViTWithClassifier(num_classes=10, augdepth=aug_depth[-1])
        mid_layers = [ViTBasic(num_classes=10, augdepth=aug_depth[i+1]) for i in range(layer_num - 2)]
        return first_layer, last_layer, mid_layers
    elif dataset == 'Food101':
        first_layer = ViTWithEmbed(img_size=224, num_classes=101, augdepth=aug_depth[0])
        last_layer = ViTWithClassifier(num_classes=101, augdepth=aug_depth[-1])
        mid_layers = [ViTBasic(num_classes=101, augdepth=aug_depth[i+1]) for i in range(layer_num - 2)]
        return first_layer, last_layer, mid_layers
    elif dataset == 'ImageNet':
        first_layer = ViTWithEmbed(img_size=256, num_classes=1000, augdepth=aug_depth[0])
        last_layer = ViTWithClassifier(num_classes=1000, augdepth=aug_depth[-1])
        mid_layers = [ViTBasic(num_classes=1000, augdepth=aug_depth[i+1]) for i in range(layer_num - 2)]
        return first_layer, last_layer, mid_layers
    else:
        raise NotImplementedError("Only support dataset: CIFAR10, STL10, SVHN, Food101, ImageNet")

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
    info_path = './vit_res/' + model_params['dataset']
    if not os.path.exists(info_path):
        os.makedirs(info_path)
    dir_idx = count_folders(info_path)
    info_path += '/'
    info_path += str(dir_idx)
    if not os.path.exists(info_path):
        os.makedirs(info_path)
    with open(info_path+'/info', 'w') as f:
        f.write(f"dataset: {model_params['dataset']}\n")
        f.write(f"layer num: {training_params['layer_num']}\n")
        f.write(f"gpu_num: {training_params['gpu_num']}\n")
        f.write(f"aug_depth: {model_params['aug_list']}\n")

    # do some preprocessing
    if os.path.exists("./whole/weights") is False:
        os.makedirs("./whole/weights")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_food = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_imgnet = transform = transforms.Compose(
        [
            transforms.Resize((256,256)),
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
    elif ds_name == 'Food101':
        print('Support, but need to download first')
        exit()
        trainset = torchvision.datasets.Food101(root='./data', split='train', transform=transform_food, download=False)
        testset = torchvision.datasets.Food101(root='./data', split='test', transform=transform_food, download=False)
    elif ds_name == 'ImageNet':
        trainset = torchvision.datasets.ImageNet(root='./data/imagenet', split='train', transform=transform_imgnet)
        testset = torchvision.datasets.ImageNet(root='./data/imagenet', split='val', transform=transform_imgnet)
    else:
        raise NotImplementedError('Only support dataset: CIFAR10, STL10, SVHN, Food101')
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=training_params['train_bs'],
                                            shuffle=True, num_workers=8)
    
    test_loader = torch.utils.data.DataLoader(testset, batch_size=training_params['test_bs'],
                                              shuffle=False, num_workers=8)
    if ds_name == 'Food101':
        # preload data
        print('start loading dataset')
        stt = time.time()
        preloaded_train = [(img, label) for img, label in train_loader]
        preloaded_test = [(img, label) for img, label in test_loader]
        print('end loading dataset')
        print(f'load time:{time.time() - stt:.2f}s')
    
    # caches
    cache_list = [multiprocessing.Queue() for _ in range(training_params['layer_num'] - 1)]
    evalcache_list = [multiprocessing.Queue() for _ in range(training_params['layer_num'] - 1)]
    # device
    device_list = [torch.device(f"cuda:{cuda_num+training_params['gpu_offset']}") for cuda_num in range(training_params['gpu_num'])]
    # aug_depth
    aug_depth = model_params['aug_list']

    # create models
    first_layer, last_layer, mid_layers = get_model(model_params['dataset'], aug_depth, training_params['layer_num'])
    # multiprocessing
    if ds_name == 'Food101':
        process_first = multiprocessing.Process(target=train_first_layer, 
                                                args=(first_layer, cache_list[0], 
                                                    evalcache_list[0], device_list[0],
                                                    preloaded_train, preloaded_test,
                                                    training_params['lr'],
                                                    training_params['momentum'],
                                                    training_params['weight_decay'],
                                                    training_params['max_epoch'],
                                                    info_path+'/info'))
    else:
        process_first = multiprocessing.Process(target=train_first_layer, 
                                                args=(first_layer, cache_list[0], 
                                                    evalcache_list[0], device_list[0],
                                                    train_loader, test_loader,
                                                    training_params['lr'],
                                                    training_params['momentum'],
                                                    training_params['weight_decay'],
                                                    training_params['max_epoch'],
                                                    info_path+'/info'))
    
    process_last = multiprocessing.Process(target=train_last_layer,
                                           args=(last_layer, cache_list[-1], 
                                                 evalcache_list[-1], device_list[-1],
                                                 training_params['lr'],
                                                 training_params['momentum'],
                                                 training_params['weight_decay'],
                                                 training_params['max_epoch'],
                                                 info_path))
    process_mid_list = []
    for i in range(training_params['layer_num'] - 2):
        # how many layers per gpu
        layer_per_gpu = training_params['layer_num'] // training_params['gpu_num']
        # device index
        device_idx = (i+1) // layer_per_gpu
        process_mid_list.append(multiprocessing.Process(target=train_basic_layer,
                                                        args=(mid_layers[i],
                                                              cache_list[i],
                                                              evalcache_list[i],
                                                              cache_list[i+1],
                                                              evalcache_list[i+1],
                                                              device_list[device_idx],
                                                              training_params['lr'],
                                                              training_params['momentum'],
                                                              training_params['weight_decay'],
                                                              training_params['max_epoch'],
                                                              i+1)))
        
    # start processes
    process_first.start()
    for p in process_mid_list:
        p.start()
    process_last.start()
    # wait the processes to finish
    process_first.join()
    for p in process_mid_list:
        p.join()
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
            transforms.Resize((256,256)),
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
    # trainset = torchvision.datasets.ImageNet(root='./data', split='train', transform=transform, download=True)
    # testset = torchvision.datasets.ImageNet(root='./data', split='val', transform=transform, download=True)
    trainset = torchvision.datasets.ImageNet(root='./data/imagenet', split='train', transform=transform)
    testset = torchvision.datasets.ImageNet(root='./data/imagenet', split='val', transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                            shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=32,
                                              shuffle=False, num_workers=1)
    # print('start loading')
    # preloaded_train = []
    # for data in tqdm(train_loader):
    #     img, lbl = data
    #     preloaded_train.append((img, lbl))
    # preloaded_test = [(img, label) for img, label in test_loader]
    # print('end loading')
    print('train info:')
    print(len(trainset))
    model = torch.nn.Linear(256*256*3, 1000)
    loss_func = torch.nn.CrossEntropyLoss()
    # print(trainset[0][0].shape)
    cnt = 0
    loss_t = 0
    for data in tqdm(train_loader):
        img, labels = data
        bs = img.shape[0]
        preds = model(img.reshape(bs, -1))
        loss = loss_func(preds, labels)
        cnt += 1
        loss_t += loss.item()
        # print(labels.min())
    print('loss:', loss_t / cnt)
    # print('test info:')
    # print(len(testset))
    # for data in tqdm(test_loader):
    #     img, labels = data
    #     print(img.shape)
    #     print(labels)
    #     break
    

if __name__ == '__main__':
    cfg = 'configs/vit_dlp.yaml'
    # train(cfg)
    test_dataset()