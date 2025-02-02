import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

log_prefix = '/home/chengqixu/gxy/temp_name1/logs'
log_path = 'UNSET'


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch_front(model, optimizer, data_loader, device, cache, split_size, inc_input=False):
    
    model.train()
    optimizer.zero_grad()

    sample_num = 0

    global log_path
    if log_path == 'UNSET':
        log_path = log_prefix + '/' + model.__class__.__name__ + '-' + datetime.now().strftime('%Y-%m-%d:%H:%M:%S') + '.log'
    data_loader = tqdm(data_loader, file=sys.stdout)
    with open(log_path, 'a') as log_file:
        for step, data in enumerate(data_loader):
            
            images, labels = data
            
            sample_num += images.shape[0]
            image_splits = images.split(split_size, dim=0)
            label_splits = labels.split(split_size, dim=0)
            splits = zip(image_splits, label_splits)

            for img_split, label_split in splits:
            
                pred = model.forward_features_dp(img_split.to(device),label_split.to(device))
                
                if inc_input:
                    cache.put((img_split.cpu().numpy(),pred.detach().cpu().numpy(),label_split.detach().cpu().numpy()))
                    # cache.put((pred.detach().cpu().numpy(), labels.detach().cpu().numpy()))  # 放入数据
                else:
                    cache.put((pred.detach().cpu().numpy(), label_split.detach().cpu().numpy()))  # 放入数据
                    
            cache.put('BatchEND')  #一个mini-batch结束
            optimizer.step()
            optimizer.zero_grad()
            rate = data_loader.format_dict['rate']
            if rate is not None:
                log_file.write("{:.2f} it/s \n".format(rate))

        

        
    
    ###### Append 一个停止信号, 一个epoch结束
    # cache.append('END')
    cache.put('END')  # 一个epoch结束
    log_file.close()

   

    return pred

def train_one_epoch_back(model, optimizer, device, epoch, cache, inc_input=False):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    step = 0
    while True:
        # if len(cache)>0:
        if not cache.empty():
        
            # current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            # print(f'[{current_time}] in train_one_epoch_back cache volumn: {len(cache)}')
            # print(f'in train_one_epoch_back cache volumn: {len(cache)}')
            
            # data = cache.pop(0)
            data = cache.get()  # 从队列获取数据

            # print(data)
            #### 退出线程
            if data == 'END':
                break 
            if data == 'BatchEND':
                optimizer.step()
                optimizer.zero_grad()
                continue
            
            pred=None
            
            if inc_input:
                oriimg, inputs, labels = data
                oriimg=torch.tensor(oriimg)
                inputs=torch.tensor(inputs)
                labels=torch.tensor(labels)
                
                
                oriimg = oriimg.to(device)
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # print(f'\rget from cache -- data.shape:{inputs.shape} cache volumn(after): {len(cache)}',end='',flush=True)
                
                step += 1
                sample_num += inputs.shape[0]
        
                pred = model.forward(inputs,labels,oriimg)
            else:
                
                inputs, labels = data
                inputs=torch.tensor(inputs)
                labels=torch.tensor(labels)
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # print(f'\rget from cache -- data.shape:{inputs.shape} cache volumn(after): {len(cache)}',end='',flush=True)
                
                step += 1
                sample_num += inputs.shape[0]
    
                pred = model.forward(inputs,labels)
        
            
                
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels).sum()
            
            loss = loss_function(pred, labels)
            # print(f'pred.requires_grad :{pred.requires_grad} loss.requires_grad:{loss.requires_grad}')
            # exit(0)
            
            
            ### 
            loss.backward()
           
            accu_loss += loss.detach()


            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)

            
            # print("\r[train epoch {}] loss: {:.3f} acc: {:.3f} len(cache):{}".format(epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num, len(cache)),end='',flush=True)
            
    print(f"-- finished: epoch {epoch}")
    print("[train epoch {}] loss: {:.3f} acc: {:.3f} ".format(epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num))

    return  accu_loss.item() / (step + 1), accu_num.item() / sample_num

def train_one_epoch_mid(model, optimizer, device, in_cache, out_cache):
    model.train()
    optimizer.zero_grad()
    while True:
        # if len(in_cache)>0:
        if not in_cache.empty():
            # 获取当前时间并格式化，包含毫秒
            # current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

            # # 打印带毫秒时间戳的消息
            # print(f'[{current_time}] in train_one_epoch_mid cache volume: {len(in_cache)}')
            
            # print(f'cache volumn: {len(cache)}')
            
            # data = in_cache.pop(0)
            data = in_cache.get()
            # print(data)
            #### 退出线程
            if data == 'END':
                break 
            
            if data == 'BatchEND':
                optimizer.step()
                optimizer.zero_grad()
                out_cache.put('BatchEND')
                continue
            
            inputs, labels = data
            inputs=torch.tensor(inputs)
            labels=torch.tensor(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # print(f'\rget from cache -- data.shape:{inputs.shape} cache volumn(after): {len(cache)}',end='',flush=True)

            
            pred = model.forward_features(inputs,labels)
            
            # out_cache.append((pred.detach().cpu(),labels.detach().cpu()))
            out_cache.put((pred.detach().cpu().numpy(), labels.detach().cpu().numpy()))
            
            
    ###### Append 一个停止信号, 一个epoch结束
    # out_cache.append('END')
    out_cache.put('END')
    

    return pred

@torch.no_grad()
def evaluate_front(model, data_loader, device, evacache, inc_input=False):

    model.eval()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        
        pred = model.forward_features(images.to(device),labels.to(device))
  
        # evacache.append((pred.detach().cpu(),labels.detach().cpu()))
        evacache.put((pred.detach().cpu().numpy(), labels.detach().cpu().numpy()))
        
    # evacache.append('END')
    evacache.put('END')
    
    return pred


@torch.no_grad()
def evaluate_mid(model, device, in_evacache, out_evacache):
    model.eval()
    
    while True:
        # if len(in_evacache)>0:
        if not in_evacache.empty():
            
            # print(f'cache volumn: {len(cache)}')
            # data = in_evacache.pop(0)
            data = in_evacache.get()
            
            # print(data)
            #### 退出线程
            if data == 'END':
                break 
            inputs, labels = data
            inputs=torch.tensor(inputs)
            labels=torch.tensor(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # print(f'\rget from cache -- data.shape:{inputs.shape} cache volumn(after): {len(cache)}',end='',flush=True)
            
            pred = model.forward_features(inputs,labels)
            out_evacache.put((pred.detach().cpu().numpy(), labels.detach().cpu().numpy()))
            
            # out_evacache.append((pred.detach().cpu(),labels.detach().cpu()))
    
    ###### Append 一个停止信号, 一个epoch结束
    out_evacache.put('END')
    

    return pred


@torch.no_grad()
def evaluate_back(model, device, epoch, evacache, inc_input=False):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    step = 0
    while True:
        if not evacache.empty():
        
        # if len(evacache)>0:
            
            data = evacache.get()
            
            # data = evacache.pop(0)
            #### 退出线程
            if data == 'END':
                break 
            
                
            inputs, labels = data
            inputs=torch.tensor(inputs)
            labels=torch.tensor(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            
            step += 1
            sample_num += inputs.shape[0]
            
            #### 推理阶段 target=None
            pred = model(inputs,target=None)
                
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels).sum()

            loss = loss_function(pred, labels)
            accu_loss += loss

    print("[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch, accu_loss.item() / (step + 1),accu_num.item() / sample_num))
    return  accu_loss.item() / (step + 1), accu_num.item() / sample_num









@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
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

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num