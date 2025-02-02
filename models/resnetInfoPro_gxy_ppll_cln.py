import copy
import pdb
import time
import argparse

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import sys
# from .config import InfoPro, InfoPro_balanced_memory
# from .auxiliary_nets_1 import Decoder, AuxClassifier

from config import InfoPro, InfoPro_balanced_memory
from auxiliary_nets_1 import Decoder, AuxClassifier
import os
from torch.utils.tensorboard import SummaryWriter


"""
1. 扩展性更好的框架，加入了seg_para 用于划分模型以及辅助网络，实现的是man的整体
2. 能work
"""





def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class InfoProResNet_basic(nn.Module):

    def __init__(self, block, arch, partname, local_module_num, infopro_config, batch_size, seg_para, image_size=32,
                 inplanes=None, balanced_memory=False, dataset='cifar10', class_num=10, useema=True,
                 wide_list=(16, 16, 32, 64), dropout_rate=0, aux_net_config='1c2f', local_loss_mode='contrast',
                 aux_net_widen=1, aux_net_feature_dim=128, momentum=0.9, device=None):
        super(InfoProResNet_basic, self).__init__()

        self.enc_indexs = seg_para['enc']
        self.dec_indexs = seg_para['dec']
       
        
        # self.layers = seg_para['layers']
        self.layers=[5,5,5]
        
        self.partname = partname
        
        self.widelist = wide_list
        self.inplanes = inplanes
        self.inplanes_ori = inplanes
        
        
        self.dropout_rate = dropout_rate
        self.class_num = class_num
        self.local_module_num = local_module_num
        self.momentum = momentum
        self.image_size=image_size
        self.infopro_config = infopro_config
        
      


        # 调整层数
        selflayer=self._initialize_layers(block)
        
        # print(selflayer)
        # exit(0)
        self._initialize_auxiliary_nets(aux_net_config, local_loss_mode, aux_net_widen, aux_net_feature_dim)
        self.Encoder_Net = self._make_Encoder_Aux_Net(selflayer)
        
        # print(self.Encoder_Net)
        # exit(0)
        
        for module in self.Encoder_Net:
            for param in module.parameters():
                param.requires_grad = True


            

        #### 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        
          # 配置均值和方差
        self.mask_train_mean, self.mask_train_std = self._configure_normalization(dataset, batch_size, image_size, device)
    
        
        
        for net in self.Encoder_Net:
            net = net.cuda()

        # for net1, net2 in zip(self.LB, self.EMA_Net):
        #     net1 = net1.cuda()
        #     net2 = net2.cuda()
            
            
            
    def _initialize_layers(self, block):
        
     
        selflayers={}
        for i in range(1,4):
            stride = 2 if i > 1 else 1
            selflayers[f'layer{i}']=self._make_layer(block, self.widelist[i], self.layers[i -1], stride=stride)
        
        
        return selflayers
    
        
    def _make_Encoder_Aux_Net(self,selflayer):
        Encoder_Net = nn.ModuleList([])

        Encoder_temp = nn.ModuleList([])
          
        local_block_index = 0
          
        for index in self.enc_indexs:
            blocks,layers=index[0],index[1]
            Encoder_temp.append(copy.deepcopy(selflayer[f'layer{blocks}'][layers]))
                         
            if blocks  == self.infopro_config[local_block_index][0] \
                    and layers == self.infopro_config[local_block_index][1]:
                        

                Encoder_Net.append(nn.Sequential(*Encoder_temp))

                Encoder_temp = nn.ModuleList([])
                local_block_index += 1
        return Encoder_Net

            
    def _make_decoder_Aux_Net(self,selflayer):
        Decoder_Net = nn.ModuleList([])

        Decoder_temp = nn.ModuleList([])
          
        local_block_index = 0
          
        for index in self.dec_indexs:
            blocks,layers=index[0],index[1]
            Decoder_temp.append(copy.deepcopy(selflayer[f'layer{blocks}'][layers]))
                         
            if blocks  == self.infopro_config[local_block_index][0] \
                    and layers == self.infopro_config[local_block_index][1]:
                        

                Decoder_Net.append(nn.Sequential(*Decoder_temp))

                Decoder_temp = nn.ModuleList([])
                local_block_index += 1
        return Decoder_Net
    
    
    
    def _configure_normalization(self, dataset, batch_size, image_size, device):
        if 'CIFAR' in dataset:
            mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
            std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
        else:
            mean = [127.5 / 255.0] * 3
            std = [127.5 / 255.0] * 3

        mean = torch.Tensor(mean).view(1, 3, 1, 1).expand(batch_size, 3, image_size, image_size).to(device)
        std = torch.Tensor(std).view(1, 3, 1, 1).expand(batch_size, 3, image_size, image_size).to(device)
        return mean, std

    
    def _image_restore(self, normalized_image):
        return normalized_image.mul(self.mask_train_std[:normalized_image.size(0)]) + self.mask_train_mean[:normalized_image.size(0)]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = [block(self.inplanes, planes, stride, downsample, dropout_rate=self.dropout_rate)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes, dropout_rate=self.dropout_rate) for _ in range(1, blocks))

        
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False
        return nn.Sequential(*layers)
    
    
    
class InfoProResNet_front(InfoProResNet_basic):

    def __init__(self, *args, **kwargs):
        super(InfoProResNet_front, self).__init__(*args, **kwargs)

     
  
        
        self.conv1 = nn.Conv2d(3, self.inplanes_ori, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes_ori)
        self.relu = nn.ReLU(inplace=True)
        self.criterion_ce = nn.CrossEntropyLoss()
    


    def forward_features(self, img, target=None,ixx_1=5, ixy_1=0.5,
                ixx_2=0, ixy_2=0):
        if self.training:
            
            x = self.conv1(img)
            x = self.bn1(x)
            x = self.relu(x)

            for i in range(len(self.Encoder_Net)):
                
                
                x = self.Encoder_Net[i](x)

                # print(f'x.shape:{x.shape}  y.shape:{y.shape}')
                local_index,layer_index = self.infopro_config[i]
                lo,la = local_index,layer_index
                # if layer_index == 4:
                #     lo,la = local_index + 1,0
                # else:
                #     lo,la = local_index,layer_index
                    
                # print(f'lo {lo} la {la}  y.shape:{y.shape}')
                # print('self.decoder_' + str(lo) + '_' + str(la))
                # print('self.aux_classifier_' + str(lo) + '_' + str(la))    
                # print(f'self.layerslist[{lo}][{la}]')
                        
                    
                
                ratio = lo / (self.local_module_num - 2) if self.local_module_num > 2 else 0
                ixx_r = ixx_1 * (1 - ratio) + ixx_2 * ratio
                ixy_r = ixy_1 * (1 - ratio) + ixy_2 * ratio

                
                # print(f'lo {lo}  la{la}')
                loss_ixx = eval('self.decoder_' + str(lo) + '_' + str(la))(x,self._image_restore(img))
                loss_ixy = eval('self.aux_classifier_' + str(lo) + '_' + str(la))(x,target)
                loss = ixx_r * loss_ixx + ixy_r * loss_ixy
                loss.backward()
                # print(f'ratio {ratio}')
                # print(f'loss : {loss}')
                # print(f'loss_ixx : {loss_ixx}')
                # print(f'loss_ixy : {loss_ixy}')
                
                
                                
                x = x.detach()

               
        
 
            return x

        else:
            x = self.conv1(img)
            x = self.bn1(x)
            x = self.relu(x)

            for i in range(len(self.Encoder_Net)):
                x = self.Encoder_Net[i](x)

            return x


class InfoProResNet_mid(InfoProResNet_basic):

    

        
    def __init__(self, *args, **kwargs):
        super(InfoProResNet_mid, self).__init__(*args, **kwargs)

     
    
    def forward_features(self, x, target=None, oriimg=None, ixx_1=5, ixy_1=0.5,
                ixx_2=0, ixy_2=0):
        if self.training:

            
            for i in range(len(self.Encoder_Net)):
                
                
                
                x = self.Encoder_Net[i](x)
                y = self.LB[i](x) + self.EMA_Net[i](x)

                    
                local_index,layer_index = self.infopro_config[i]
                if layer_index == 4:
                    lo,la = local_index + 1,0
                else:
                    lo,la = local_index,layer_index + 1
          
           
     
                ratio = lo / (self.local_module_num - 2) if self.local_module_num > 2 else 0
                ixx_r = ixx_1 * (1 - ratio) + ixx_2 * ratio
                ixy_r = ixy_1 * (1 - ratio) + ixy_2 * ratio
        
                loss_ixx = eval('self.decoder_' + str(lo) + '_' + str(la))(y,self._image_restore(oriimg))
                loss_ixy = eval('self.aux_classifier_' + str(lo) + '_' + str(la))(y,target)
                loss = ixx_r * loss_ixx + ixy_r * loss_ixy
                loss.backward()

                x = x.detach()
           

            return x

        else:


            for i in range(len(self.Encoder_Net)):
                x = self.Encoder_Net[i](x)

            return x
        

class InfoProResNet_back(InfoProResNet_basic):

    

        
    def __init__(self, *args, **kwargs):
        super(InfoProResNet_back, self).__init__(*args, **kwargs)

     
        self.feature_num=self.widelist[-1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.feature_num, self.class_num)
        self.head = nn.Linear(self.feature_num, self.class_num)
        
        self.Flatten = nn.Flatten()

        self.criterion_ce = nn.CrossEntropyLoss()

 
    def forward(self, x, target=None, oriimg=None, ixx_1=5, ixy_1=0.5,
                ixx_2=0, ixy_2=0):
        if self.training:

            
            for i in range(len(self.Encoder_Net) - 1):
                
                
                
                x = self.Encoder_Net[i](x)

                    
                local_index,layer_index = self.infopro_config[i]
                # if layer_index == 4:
                #     lo,la = local_index + 1,0
                # else:
                #     lo,la = local_index,layer_index
          
                lo,la = local_index,layer_index
           
     
                ratio = lo / (self.local_module_num - 2) if self.local_module_num > 2 else 0
                ixx_r = ixx_1 * (1 - ratio) + ixx_2 * ratio
                ixy_r = ixy_1 * (1 - ratio) + ixy_2 * ratio
                loss_ixx = eval('self.decoder_' + str(lo) + '_' + str(la))(x,self._image_restore(oriimg))
                loss_ixy = eval('self.aux_classifier_' + str(lo) + '_' + str(la))(x,target)
                loss = ixx_r * loss_ixx + ixy_r * loss_ixy
                loss.backward()

                x = x.detach()
      


            # last local module
            x = self.Encoder_Net[-1](x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

            logits = self.head(x)
            
            loss = self.criterion_ce(logits, target)
            loss.backward()
            
            return logits,loss
        
            loss = self.criterion_ce(logits, target)
            loss.backward()
            return logits,loss
         

        else:
 

            for i in range(len(self.Encoder_Net)):
                x = self.Encoder_Net[i](x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            logits = self.head(x)
        
            loss = self.criterion_ce(logits, target)
            return logits, loss
            




def resnet32_front(**kwargs):
    # layers=kwargs['layers']
    # infopro_config=kwargs['infopro_config']
    # wide_list=kwargs['wide_list']
    model = InfoProResNet_front(BasicBlock, arch='resnet32', **kwargs)
    
    return model

def resnet32_mid(**kwargs):
    model = InfoProResNet_mid(BasicBlock, arch='resnet32', **kwargs)
    
    return model

def resnet32_back(**kwargs):
    model = InfoProResNet_back(BasicBlock, arch='resnet32', **kwargs)
    return model




import yaml



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count





training_configurations = {
    'resnet': {
        'epochs': 400,
        'batch_size': 1024 ,
        'initial_learning_rate': 0.8 ,
        'changing_lr': [300],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
    }
}



def adjust_learning_rate(model, optimizer, epoch, cos_lr=True):
    """Sets the learning rate"""
    if not cos_lr:
        if epoch in training_configurations[model]['changing_lr']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= training_configurations[model]['lr_decay_rate']
        print('lr:')
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

    else:
        for param_group in optimizer.param_groups:
            if epoch <= 10:
                param_group['lr'] = 0.5 * training_configurations[model]['initial_learning_rate']\
                                * (1 + math.cos(math.pi * epoch / training_configurations[model]['epochs'])) * (epoch - 1) / 10 + 0.01 * (11 - epoch) / 10
            else:
                param_group['lr'] = 0.5 * training_configurations[model]['initial_learning_rate']\
                                    * (1 + math.cos(math.pi * epoch / training_configurations[model]['epochs']))
        print('lr:')
        for param_group in optimizer.param_groups:
            print(param_group['lr'])


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.constant_(m.weight, 0.5)  # 例如，将卷积层的权重初始化为0.5
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.1)  # 将偏置初始化为0.1
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)  # BatchNorm权重初始化为1
            torch.nn.init.constant_(m.bias, 0)    # 偏置初始化为0
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.constant_(m.weight, 0.1)  # 全连接层的权重初始化为0.1
            torch.nn.init.constant_(m.bias, 0)      # 偏置初始化为0


def count_trainable_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:  # 检查是否需要梯度
            print(f"{name}: para: {param.numel()}, shape: {param.shape}")
            total_params += param.numel()  # 累加参数数量
    print(f"Total trainable parameters: {total_params}")
    return total_params


def get_para(partname1,cfg_data,partname2=None):
    if partname2!=None:
        infrop_s=cfg_data[partname2][partname1]['infrop_s']
        infrop_e=cfg_data[partname2][partname1]['infrop_e']
        inplanes=cfg_data[partname2][partname1]['inplanes']
        seg_para=cfg_data[partname2][partname1]['seg_para']
        device=cfg_data[partname2][partname1]['device']
        
    else:
        
        infrop_s=cfg_data[partname1]['infrop_s']
        infrop_e=cfg_data[partname1]['infrop_e']
        inplanes=cfg_data[partname1]['inplanes']
        seg_para=cfg_data[partname1]['seg_para']
        device=cfg_data[partname1]['device']
        
    
    
    return infrop_s,infrop_e,inplanes,seg_para,device

######### 分成4个模块但是不并行训练
def main_k4():
    # torch.manual_seed(42)
    
    
    writer = SummaryWriter(log_dir="./runs/InfoProResNet_ori")
    
    epoches=400
    cfg='configs/k4_resnet32_gxy_modify.yaml'
    
    with open(cfg, 'r') as file1:
        cfg_data=  yaml.load(file1,Loader=yaml.FullLoader)
    
    
    ## front 参数：
    finfrop_s,finfrop_e,f_inplanes,seg_paraf,devicef = get_para('front',cfg_data,None)
    

    infopro_config_f=InfoPro[cfg_data['common']['InfoPro']][cfg_data['common']['local_module_num']][finfrop_s:finfrop_e]
  
    
    
    ######## mid1 参数
    m1infrop_s,m1infrop_e,m1_inplanes,seg_param1,devicem1= get_para('part1',cfg_data,'mid')
    infopro_config_m1=InfoPro[cfg_data['common']['InfoPro']][cfg_data['common']['local_module_num']][m1infrop_s:m1infrop_e]
    
    
    
    ######## mid2 参数
    m2infrop_s,m2infrop_e,m2_inplanes,seg_param2,devicem2 = get_para('part2',cfg_data,'mid')
    infopro_config_m2=InfoPro[cfg_data['common']['InfoPro']][cfg_data['common']['local_module_num']][m2infrop_s:m2infrop_e]
    
    
    
    ######## back 参数
    binfrop_s,binfrop_e,b_inplanes,seg_parab,deviceb = get_para('back',cfg_data,None)
    
 
    infopro_config_b=InfoPro[cfg_data['common']['InfoPro']][cfg_data['common']['local_module_num']][binfrop_s:binfrop_e]
 
    
    
    batch_size=cfg_data['common']['batch_size']
    image_size=cfg_data['common']['image_size']
    dropout_rate=cfg_data['common']['dropout_rate']
    class_num=cfg_data['common']['class_num']
    aux_net_config=cfg_data['common']['aux_net_config']
    local_loss_mode=cfg_data['common']['local_loss_mode']
    aux_net_widen=cfg_data['common']['aux_net_widen']
    aux_net_feature_dim=cfg_data['common']['aux_net_feature_dim']
    local_module_num=cfg_data['common']['local_module_num']
    wide_list=cfg_data['common']['wide_list']
    
    front= resnet32_front(infopro_config=infopro_config_f, local_module_num=local_module_num, partname='front', seg_para=seg_paraf, inplanes=f_inplanes, batch_size=1024, image_size=32,
                   dataset='cifar10', class_num=10, wide_list=wide_list,
                    dropout_rate=0,
                   aux_net_config=aux_net_config, local_loss_mode=local_loss_mode,
                   aux_net_widen=1, aux_net_feature_dim=128, device=devicef)
    front = front.to(devicef)
    
    
    # count_trainable_parameters(front)

    # initialize_weights(front)
    
    # print(f'front:{front}')
  
    
    
    
    mid1= resnet32_mid(infopro_config=infopro_config_m1, local_module_num=local_module_num, partname='mid1', seg_para=seg_param1, inplanes=m1_inplanes, batch_size=1024, image_size=32,
                   dataset='cifar10', class_num=10, wide_list=wide_list,
                    dropout_rate=0,
                   aux_net_config=aux_net_config, local_loss_mode=local_loss_mode,
                   aux_net_widen=1, aux_net_feature_dim=128, device=devicem1)
    mid1 = mid1.to(devicem1)
    
    # count_trainable_parameters(mid1)
    
    
    mid2= resnet32_mid(infopro_config=infopro_config_m2, local_module_num=local_module_num, partname='mid2', seg_para=seg_param2, inplanes=m2_inplanes, batch_size=1024, image_size=32,
                   dataset='cifar10', class_num=10, wide_list=wide_list,
                    dropout_rate=0,
                   aux_net_config=aux_net_config, local_loss_mode=local_loss_mode,
                   aux_net_widen=1, aux_net_feature_dim=128, device=devicem2)
    mid2 = mid2.to(devicem2)
    
    
    # count_trainable_parameters(mid2)

    
    
    back= resnet32_back(infopro_config=infopro_config_b, local_module_num=local_module_num, partname='back', seg_para=seg_parab, inplanes=b_inplanes, batch_size=1024, image_size=32,
                   dataset='cifar10', class_num=10, wide_list=wide_list,
                   dropout_rate=0,
                   aux_net_config=aux_net_config, local_loss_mode=local_loss_mode,
                   aux_net_widen=1, aux_net_feature_dim=128,device=deviceb)
    
    # print(f'back:{back}')

    
    # count_trainable_parameters(back)
    
    # exit(0)
    back = back.to(deviceb)
    # initialize_weights(back)
    

    optimizef = torch.optim.SGD(front.parameters(),
                                lr=0.1,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=0.0001)
    
    optimizem1 = torch.optim.SGD(mid1.parameters(),
                                lr=0.1,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=0.0001)
    optimizem2 = torch.optim.SGD(mid2.parameters(),
                                lr=0.1,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=0.0001)
    
    optimizeb = torch.optim.SGD(back.parameters(),
                                lr=0.1,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=0.0001)
    
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    kwargs_dataset_train = {'train': True}
    kwargs_dataset_test = {'train': False}
    
    kwargs = {'num_workers': 8, 'pin_memory': False}
    

    
    transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                  (4, 4, 4, 4), mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])
    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__['CIFAR10']('./data', download=True, transform=transform_train,
                                                **kwargs_dataset_train),
        batch_size=1024, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__['CIFAR10']('./data', transform=transform_test,
                                                **kwargs_dataset_test),
        batch_size=1024, shuffle=False, **kwargs)

    
    



    # switch to train mode
    front.train()
    mid1.train()
    mid2.train()
    back.train()
    
    
    for epoch in range(epoches):
        # train(train_loader, front,back, optimizef, optimizeb, epoch)
            
        # adjust_learning_rate('resnet',optimizef, epoch + 1,False)
        # adjust_learning_rate('resnet',optimizeb, epoch + 1,False)
        
        adjust_learning_rate('resnet',optimizef, epoch + 1)
        adjust_learning_rate('resnet',optimizem1, epoch + 1)
        adjust_learning_rate('resnet',optimizem2, epoch + 1)
        adjust_learning_rate('resnet',optimizeb, epoch + 1)
        
        # for param_groupf,param_groupb in zip(optimizef.param_groups,optimizeb.param_groups):
        #     print(f"Lr check Epoch {epoch}: Current learning rate for optimizef:{param_groupf['lr']}   optimizeb:{param_groupb['lr']}")
        # prec1 = validate(val_loader, front,back, epoch)
    
        train_prec1, train_speed = train(train_loader, front, mid1, mid2, back, optimizef, optimizem1,optimizem2,optimizeb, epoch, writer)
        
        val_prec1 = validate(val_loader, front,  mid1, mid2, back, epoch, writer)
            
        # 记录 epoch 结束时的平均精度
        writer.add_scalar('Prec@1/train', train_prec1, epoch)
        writer.add_scalar('Prec@1/val', val_prec1, epoch)
        writer.add_scalar('Speed/train', train_speed, epoch)
        
    writer.close()  # 关闭记录器  

######### 分成2个模块但是不并行训练
def main_k2():
    
    
    
    epoches=400
    cfg='configs/k2_resnet32_gxy_ppll_k16_noema.yaml'
    
    with open(cfg, 'r') as file1:
        cfg_data=  yaml.load(file1,Loader=yaml.FullLoader)
    
    
    ## front 参数：
    finfrop_s,finfrop_e,f_inplanes,seg_paraf,devicef = get_para('front',cfg_data,None)
    

    infopro_config_f=InfoPro[cfg_data['common']['InfoPro']][cfg_data['common']['local_module_num']][finfrop_s:finfrop_e]

    
    
    ######## back 参数
    binfrop_s,binfrop_e,b_inplanes,seg_parab,deviceb = get_para('back',cfg_data,None)
    
 
    infopro_config_b=InfoPro[cfg_data['common']['InfoPro']][cfg_data['common']['local_module_num']][binfrop_s:binfrop_e]
 
    
    
    batch_size=cfg_data['common']['batch_size']
    image_size=cfg_data['common']['image_size']
    dropout_rate=cfg_data['common']['dropout_rate']
    class_num=cfg_data['common']['class_num']
    aux_net_config=cfg_data['common']['aux_net_config']
    local_loss_mode=cfg_data['common']['local_loss_mode']
    aux_net_widen=cfg_data['common']['aux_net_widen']
    aux_net_feature_dim=cfg_data['common']['aux_net_feature_dim']
    local_module_num=cfg_data['common']['local_module_num']
    wide_list=cfg_data['common']['wide_list']
    
    front= resnet32_front(infopro_config=infopro_config_f, local_module_num=local_module_num, partname='front', seg_para=seg_paraf, inplanes=f_inplanes, batch_size=1024, image_size=32,
                   dataset='cifar10', class_num=10, wide_list=wide_list,
                    dropout_rate=0,
                   aux_net_config=aux_net_config, local_loss_mode=local_loss_mode,
                   aux_net_widen=1, aux_net_feature_dim=128, device='cuda:0')
    front = front.to('cuda:0')
    
    
    # count_trainable_parameters(front)

    # initialize_weights(front)
    
    # print(f'front:{front}')


    
    
    back= resnet32_back(infopro_config=infopro_config_b, local_module_num=local_module_num, partname='back', seg_para=seg_parab, inplanes=b_inplanes, batch_size=1024, image_size=32,
                   dataset='cifar10', class_num=10, wide_list=wide_list,
                   dropout_rate=0,
                   aux_net_config=aux_net_config, local_loss_mode=local_loss_mode,
                   aux_net_widen=1, aux_net_feature_dim=128,device='cuda:0')
    
    # print(f'back:{back}')

    
    # count_trainable_parameters(back)
    
    # exit(0)
    back = back.to('cuda:0')
    # initialize_weights(back)
    

    optimizef = torch.optim.SGD(front.parameters(),
                                lr=0.1,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=0.0001)
    
 
    
    optimizeb = torch.optim.SGD(back.parameters(),
                                lr=0.1,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=0.0001)
    
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    kwargs_dataset_train = {'train': True}
    kwargs_dataset_test = {'train': False}
    
    kwargs = {'num_workers': 8, 'pin_memory': False}
    

    
    transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                  (4, 4, 4, 4), mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])
    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__['CIFAR10']('./data', download=True, transform=transform_train,
                                                **kwargs_dataset_train),
        batch_size=1024, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__['CIFAR10']('./data', transform=transform_test,
                                                **kwargs_dataset_test),
        batch_size=1024, shuffle=False, **kwargs)



    # switch to train mode
    front.train()

    back.train()
    
    
    for epoch in range(epoches):
        # train(train_loader, front,back, optimizef, optimizeb, epoch)
            
        # adjust_learning_rate('resnet',optimizef, epoch + 1,False)
        # adjust_learning_rate('resnet',optimizeb, epoch + 1,False)
        
        adjust_learning_rate('resnet',optimizef, epoch + 1)

        adjust_learning_rate('resnet',optimizeb, epoch + 1)
        
        # for param_groupf,param_groupb in zip(optimizef.param_groups,optimizeb.param_groups):
        #     print(f"Lr check Epoch {epoch}: Current learning rate for optimizef:{param_groupf['lr']}   optimizeb:{param_groupb['lr']}")
        # prec1 = validate(val_loader, front,back, epoch)
    
        train_prec1, train_speed = traink2(train_loader, front, back, optimizef,optimizeb, epoch)
        
        val_prec1 = validatek2(val_loader, front,back, epoch)
            




from datetime import datetime
def train_one_epoch_front(train_loader, front, device, optimizer,out_cache):
# def train(train_loader, front, back, optimizef, epoch, writer):
    
    
    front.train()
    optimizer.zero_grad()
    # dict1={'forward':0,'张量转移到cpu':0,'放入cache':0,'update':0}
    
    for i, (x, target) in enumerate(tqdm(train_loader)):
        
        

        # current_time = datetime.now()
        # print(f'[开始一次循环] in train_one_epoch_front')
        target = target.to(device)
        x = x.to(device)
            
        
        
        frontout = front.forward_features(img=x, target=target, ixx_1=5, ixy_1=0.5, ixx_2=0, ixy_2=0)
        
        # current_time1 = datetime.now()
        # dict1['forward']+=(current_time1-current_time).total_seconds() * 1000
        # print(f'[forward] 用时{(current_time1-current_time).total_seconds() * 1000 } ')
        
        
        frontout1=frontout.detach()
        target1=target.detach()
        x1=x.detach()
    
        
        
        # current_time2 = datetime.now()
        # dict1['张量转移到cpu']+=(current_time2-current_time1).total_seconds() * 1000
        # print(f'[张量转移到cpu] 用时{(current_time2-current_time1).total_seconds() * 1000 } ')
        
        out_cache.put((frontout1, target1, x1))
        
        
        # current_time3 = datetime.now()
        # dict1['放入cache']+=(current_time3-current_time2).total_seconds() * 1000

        # print(f'[放入cache] 用时{(current_time3-current_time2).total_seconds() * 1000 } ')
        
        
        optimizer.step()
        optimizer.zero_grad()

        # current_time4 = datetime.now()
        # dict1['update']+=(current_time4-current_time3).total_seconds() * 1000
        
        # print(f'[update] 用时{(current_time4-current_time3).total_seconds() * 1000 }')
      
    out_cache.put('END')  # 放入数据
    
    # num_batches = len(train_loader)
    # for key,value in dict1.items():
    #      print(f'{key} 平均耗时: {value / num_batches:.2f} ms')
    
    
    return frontout



def train_one_epoch_mid(mid, device, optimizer, in_cache, out_cache):
    mid.train()
    optimizer.zero_grad()
    while True:
        # if len(in_cache)>0:
        if not in_cache.empty():
       
            data = in_cache.get()
            # print(data)
            
            #### 退出线程
            if data == 'END':
                break 
            inputs, labels, oriimg = data
            # inputs=torch.tensor(inputs)
            # labels=torch.tensor(labels)
            # oriimg=torch.tensor(oriimg)
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            oriimg = oriimg.to(device)
            
            
            # print(f'\rget from cache -- data.shape:{inputs.shape} cache volumn(after): {len(cache)}',end='',flush=True)

            
            pred = mid.forward_features(x=inputs, target=labels, oriimg=oriimg, ixx_1=5, ixy_1=0.5, ixx_2=0, ixy_2=0)
            
            out_cache.put((pred.detach(), labels.detach(), oriimg.detach()))
            
            optimizer.step()
            optimizer.zero_grad()
            
            
    ###### Append 一个停止信号, 一个epoch结束

    out_cache.put('END')
    

    return pred


def train_one_epoch_back(back, optimizer, device, epoch, in_cache):
    back.train()

    optimizer.zero_grad()

    sample_num = 0
    step = 0
    
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    i=0
    train_batches_num=49
    
    while True:
        # if len(cache)>0:
        if not in_cache.empty():
            # print("back队列中元素数量:", in_cache.qsize())
        
            
            # data = cache.pop(0)
            data = in_cache.get()  # 从队列获取数据
            # current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            # print(f'[back model {current_time}] get one')
            
            # print(data)
            #### 退出线程
            if data == 'END':
                break 
            
            
            pred=None
        
            inputs, labels, oriimg = data
 
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            oriimg = oriimg.to(device)
            
            # print(f'\rget from cache -- data.shape:{inputs.shape} cache volumn(after): {len(cache)}',end='',flush=True)
            
            step += 1
            sample_num += inputs.shape[0]
    
            pred,loss = back(x=inputs, target=labels, oriimg=oriimg, ixx_1=5, ixy_1=0.5, ixx_2=0, ixy_2=0)


                   # 计算 Prec@1 和更新指标
            prec1 = accuracy(pred.data, labels, topk=(1,))[0]
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % 10 == 0:
                # TensorBoard 中记录损失和精度
                # writer.add_scalar('Loss/train', losses.ave, epoch * train_batches_num + i)
                # writer.add_scalar('Prec@1/train_batch', top1.ave, epoch * train_batches_num + i)
            
            
                string = ('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                        'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                        'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                        epoch, i+1, train_batches_num, batch_time=batch_time,
                        loss=losses, top1=top1))

                print(string)
            ### 
 
            optimizer.step()
            optimizer.zero_grad()
            i+=1
            
            
    return losses.value,top1.value





@torch.no_grad()
def evaluate_front(model, data_loader, device, evacache):

    model.eval()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        
        pred = model.forward_features(images.to(device),None)
  
        evacache.put((pred.detach().cpu(),labels.detach().cpu()))
      

        # evacache.put((pred.detach().cpu().numpy(), labels.detach().cpu().numpy()))
        
    # evacache.append('END')
    evacache.put('END')
    
    return pred


@torch.no_grad()
def evaluate_mid(model, device, in_evacache, out_evacache):
    model.eval()
    
    while True:
        # if len(in_evacache)>0:
        if not in_evacache.empty():
            
      
            data = in_evacache.get()
            
            # print(data)
            #### 退出线程
            if data == 'END':
                break 
            inputs, labels = data
            # inputs=torch.tensor(inputs)
            # labels=torch.tensor(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # print(f'\rget from cache -- data.shape:{inputs.shape} cache volumn(after): {len(cache)}',end='',flush=True)
            
            pred = model.forward_features(inputs,None)
            out_evacache.put((pred.detach().cpu(), labels.detach().cpu()))
            
    
    ###### Append 一个停止信号, 一个epoch结束
    out_evacache.put('END')
    

    return pred


@torch.no_grad()
def evaluate_back(model, device, epoch, evacache):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    # accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    # accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    step = 0
    val_batches_num=10
    i=0
    end = time.time()
    
    while True:
        if not evacache.empty():
        
            data = evacache.get()
            
            # data = evacache.pop(0)
            #### 退出线程
            if data == 'END':
                break 
            
            i+=1
            
            inputs, labels = data
            # inputs=torch.tensor(inputs)
            # labels=torch.tensor(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            
            step += 1
            sample_num += inputs.shape[0]
            
            
            #### 推理阶段 target=None
            pred,loss = model(inputs,target=labels)
       
            prec1 = accuracy(pred.data, labels, topk=(1,))[0]
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
       
    string = ('Test: [{0}][{1}/{2}]\t'
              'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
              'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
              'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
        epoch, (i), val_batches_num, batch_time=batch_time,
        loss=losses, top1=top1))
    
    print(string)


    return  losses.value,top1.value



def train_front(train_loader,val_loader, cfg_data, cache, evacache):

     ## front 参数：
    finfrop_s,finfrop_e,f_inplanes,seg_paraf,devicef = get_para('front',cfg_data,None)
    
    infopro_config_f=InfoPro[cfg_data['common']['InfoPro']][cfg_data['common']['local_module_num']][finfrop_s:finfrop_e]
    aux_net_config=cfg_data['common']['aux_net_config']
    local_loss_mode=cfg_data['common']['local_loss_mode']

    local_module_num=cfg_data['common']['local_module_num']
    wide_list=cfg_data['common']['wide_list']
    epochs=cfg_data['common']['epochs']
    
    ### setting lr
    lr=cfg_data['common']['lr']
    momentum=cfg_data['common']['momentum']
    weight_decay=cfg_data['common']['weight_decay']
    use_coslr=cfg_data['common']['cos_lr']
    
#### 初始化front模型
    
    front= resnet32_front(infopro_config=infopro_config_f, local_module_num=local_module_num, partname='front', seg_para=seg_paraf, inplanes=f_inplanes, batch_size=1024, image_size=32,
                   dataset='cifar10', class_num=10, wide_list=wide_list,
                    dropout_rate=0,
                   aux_net_config=aux_net_config, local_loss_mode=local_loss_mode,
                   aux_net_widen=1, aux_net_feature_dim=128, device=devicef)
    
    front = front.to(devicef)
    
    # count_trainable_parameters(front)
    
    # exit(0)
    optimizef = torch.optim.SGD(front.parameters(),
                                lr=lr,
                                momentum=momentum,
                                nesterov=True,
                                weight_decay=weight_decay)
    
            

    for epoch in range(epochs):
        
        
        adjust_learning_rate('resnet', optimizer=optimizef, epoch=epoch + 1,cos_lr=use_coslr)
        
       
        # train
        pred = train_one_epoch_front(
                                    train_loader=train_loader,
                                    front=front,
                                    device=devicef,
                                    optimizer=optimizef,
                                    out_cache=cache,
                                    )
 
        print(f'train_one_epoch_front finish')
        
        # print("front outcache 队列中元素数量:", cache.qsize())
        
        print(f'start evaluate front')
        pred = evaluate_front(
                            data_loader=val_loader,
                            model=front,
                            device=devicef,
                            evacache=evacache,
                            )
        print(f'evaluate_front finish')


#### 中间模块
def train_mid(partname, cfg_data, in_cache, out_cache, in_evacache, out_evacache):
    
    ## front 参数：
    minfrop_s,minfrop_e,m_inplanes,seg_param,devicem = get_para(partname,cfg_data,'mid')
    
    infopro_config_m=InfoPro[cfg_data['common']['InfoPro']][cfg_data['common']['local_module_num']][minfrop_s:minfrop_e]
    aux_net_config=cfg_data['common']['aux_net_config']
    local_loss_mode=cfg_data['common']['local_loss_mode']

    local_module_num=cfg_data['common']['local_module_num']
    wide_list=cfg_data['common']['wide_list']
    epochs=cfg_data['common']['epochs']

    ### setting lr
    lr=cfg_data['common']['lr']
    momentum=cfg_data['common']['momentum']
    weight_decay=cfg_data['common']['weight_decay']
    use_coslr=cfg_data['common']['cos_lr']
    
#### 初始化mid模型
    mid= resnet32_mid(infopro_config=infopro_config_m, local_module_num=local_module_num, partname='mid1', seg_para=seg_param, inplanes=m_inplanes, batch_size=1024, image_size=32,
                   dataset='cifar10', class_num=10, wide_list=wide_list,
                    dropout_rate=0,
                   aux_net_config=aux_net_config, local_loss_mode=local_loss_mode,
                   aux_net_widen=1, aux_net_feature_dim=128, device=devicem)
    mid = mid.to(devicem)
    
    # model = create_model(model_name=comargs.model_name, part='mid', args=args, comargs=comargs)
    optimizem = torch.optim.SGD(mid.parameters(),
                                lr=lr,
                                momentum=momentum,
                                nesterov=True,
                                weight_decay=weight_decay)

    for epoch in range(epochs):
        
        adjust_learning_rate('resnet', optimizer=optimizem, epoch=epoch + 1,cos_lr=use_coslr)
        
        # train
        pred = train_one_epoch_mid(mid=mid,
                                    device=devicem,
                                    optimizer=optimizem,
                                    in_cache=in_cache,
                                    out_cache=out_cache)


          # validate
        pred = evaluate_mid(model=mid,
                            device=devicem,
                            in_evacache=in_evacache,
                            out_evacache=out_evacache)


#### 最后一个模块
def train_back(cfg_data, cache, evacache):

    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    

    
    dirname= cfg_data['common']['exp_name']
    log_dir=f'./runs/{dirname}'
    os.makedirs(log_dir,exist_ok=True)
    
    tb_writer = SummaryWriter(log_dir=log_dir)

    ## back 参数：
    binfrop_s,binfrop_e,b_inplanes,seg_parab,deviceb = get_para('back',cfg_data,None)
    
    infopro_config_b=InfoPro[cfg_data['common']['InfoPro']][cfg_data['common']['local_module_num']][binfrop_s:binfrop_e]
    aux_net_config=cfg_data['common']['aux_net_config']
    local_loss_mode=cfg_data['common']['local_loss_mode']

    local_module_num=cfg_data['common']['local_module_num']
    wide_list=cfg_data['common']['wide_list']
    epochs=cfg_data['common']['epochs']

    ### setting lr
    lr=cfg_data['common']['lr']
    momentum=cfg_data['common']['momentum']
    weight_decay=cfg_data['common']['weight_decay']
    use_coslr=cfg_data['common']['cos_lr']
    
#### 初始化back模型

    back= resnet32_back(infopro_config=infopro_config_b, local_module_num=local_module_num, partname='back', seg_para=seg_parab, inplanes=b_inplanes, batch_size=1024, image_size=32,
                   dataset='cifar10', class_num=10, wide_list=wide_list,
                   dropout_rate=0,
                   aux_net_config=aux_net_config, local_loss_mode=local_loss_mode,
                   aux_net_widen=1, aux_net_feature_dim=128,device=deviceb)

    back = back.to(deviceb)
    
    # count_trainable_parameters(back)
    
    # exit(0)
    optimizeb = torch.optim.SGD(back.parameters(),
                                lr=lr,
                                momentum=momentum,
                                nesterov=True,
                                weight_decay=weight_decay)
   
    


    for epoch in range(epochs):
        
        # print(f'train_back cache len: {len(cache)}')
        adjust_learning_rate('resnet', optimizeb, epoch=epoch + 1,cos_lr=use_coslr)

            
        # train
        train_loss,train_prec1=train_one_epoch_back(back=back,
                            optimizer=optimizeb,
                            device=deviceb,
                            epoch=epoch,
                            in_cache=cache
                            )

        
        
        
        print(f'train_one_epoch_back finish')
        
        print(f'start evaluate back')
        # validate
        val_loss,val_prec1=evaluate_back(model=back,
                    device=deviceb,
                    epoch=epoch,
                    evacache=evacache,
                    )
        
        print(f'evaluate_back finish')
        # exit(0)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_prec1, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_prec1, epoch)

        # print(f'Train:\nloss:{train_loss}\nacc:{train_acc}\n\nVal:\nloss:{val_loss}\nacc:{val_acc}')


        # torch.save(model.state_dict(), f"./weights/{comargs.exp_name}/{args.partname}/model-{epoch}.pth")


import multiprocessing
def training(cfg=None):
    """
    训练函数，初始化模型、优化器并启动训练线程。

    参数:
    config (dict): 配置字典，包括设备、训练参数等
    criterion (nn.Module): 损失函数
    optimizerA (optim.Optimizer): 网络A的优化器
    optimizerB (optim.Optimizer): 网络B的优化器
    """
    

    # cfg='configs/k2_resnet32_gxy_ppll_k16_noema.yaml'
    cfg='configs/k2_resnet32_gxy_ppll_k8_noema.yaml'
    
    
    
    torch.manual_seed(42)
    
    with open(cfg, 'r') as file1:
        cfg_data=  yaml.load(file1,Loader=yaml.FullLoader)
    
    
    
    
    
    # 划分块数
    k = cfg_data['k']
    
    
    
    dataset=cfg_data['common']['dataset']
    ########  数据加载和预处理
    if dataset=='CIFAR10':
    
        
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        kwargs_dataset_train = {'train': True}
        kwargs_dataset_test = {'train': False}
        
        kwargs = {'num_workers': 8, 'pin_memory': False}
        

        
        transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                    (4, 4, 4, 4), mode='reflect').squeeze()),
                    transforms.ToPILImage(),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ])
        train_loader = torch.utils.data.DataLoader(
            datasets.__dict__['CIFAR10']('./data', download=True, transform=transform_train,
                                                    **kwargs_dataset_train),
            batch_size=1024, shuffle=False, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.__dict__['CIFAR10']('./data', transform=transform_test,
                                                    **kwargs_dataset_test),
            batch_size=1024, shuffle=False, **kwargs)

        # first_batch = next(iter(train_loader))
        # print("Images: ", first_batch[0])  # Check shape of images
        # print("Labels: ", first_batch[1])
        # exit(0)

        
    elif dataset=='SVHN':

        normalize = transforms.Normalize(mean=[x / 255 for x in [127.5, 127.5, 127.5]],
                                         std=[x / 255 for x in [127.5, 127.5, 127.5]])
        kwargs_dataset_train = {'split': 'train'}
        kwargs_dataset_test = {'split': 'test'}
    

    
        transform_train = transforms.Compose(
                [transforms.RandomCrop(32, padding=2),
                 transforms.ToTensor(),
                 normalize])
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])
        train_loader = torch.utils.data.DataLoader(
        datasets.__dict__['SVHN']('./data', download=True, transform=transform_train,
                                                **kwargs_dataset_train),
        batch_size=1024, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
        datasets.__dict__['SVHN']('./data', transform=transform_test,
                                                **kwargs_dataset_test),
        batch_size=1024, shuffle=False, **kwargs)
    
        
    
    # k=2 则没有mid层
    if k == 2:
       
        # 创建队列用于进程间通信
        cache = multiprocessing.Queue()
        evacache = multiprocessing.Queue()
        
        # front_args = argparse.Namespace(**front_args)
        # back_args = argparse.Namespace(**back_args)
        # comargs = argparse.Namespace(**comargs)
        

        # 创建进程来并行训练
        process_front = multiprocessing.Process(target=train_front, args=(train_loader, val_loader, cfg_data, cache, evacache))
        process_back = multiprocessing.Process(target=train_back, args=(cfg_data, cache, evacache))
        # process_rd = multiprocessing.Process(target=log_gpu_memory, args=(rd_txt,))
        # process_rd.daemon = True  # 设置为守护进程

        # 启动进程
        process_front.start()
        process_back.start()
        # process_rd.start()

        # 等待进程完成
        process_front.join()
        process_back.join()

    # 网络由front，mid，back组成
    else:

        # 创建队列用于进程间通信
        startcache = multiprocessing.Queue()
        mid_cache = [multiprocessing.Queue() for _ in range(k - 2)]  # 为每个mid创建独立的队列
        startevacache = multiprocessing.Queue()
        midevacache = [multiprocessing.Queue() for _ in range(k - 2)]
        
        # front_args = argparse.Namespace(**front_args)
        # back_args = argparse.Namespace(**back_args)
        # comargs = argparse.Namespace(**comargs)
        
        # rd_txt = f'rd/{comargs.exp_name}.txt'

        # 创建进程来并行训练
        process_front = multiprocessing.Process(target=train_front, args=(train_loader, val_loader, cfg_data, startcache, startevacache))
        process_mid_pool = []

        for i in range(k - 2):
            # mid_args_ = argparse.Namespace(**mid_args[f'part{i+1}'])

            if i == 0:
                process_mid = multiprocessing.Process(target=train_mid, args=(f'part{i+1}', cfg_data, startcache, mid_cache[i], startevacache, midevacache[i]))
                process_mid_pool.append(process_mid)
            else:
                process_mid = multiprocessing.Process(target=train_mid, args=(f'part{i+1}', cfg_data, mid_cache[i-1], mid_cache[i], midevacache[i-1], midevacache[i]))
                process_mid_pool.append(process_mid)

        process_back = multiprocessing.Process(target=train_back, args=(cfg_data, mid_cache[-1], midevacache[-1]))
        # process_rd = multiprocessing.Process(target=log_gpu_memory, args=(rd_txt,))
        # process_rd.daemon = True  # 设置为守护进程

        # 启动进程
        process_front.start()
        for process_mid in process_mid_pool:
            process_mid.start()
        process_back.start()
        # process_rd.start()

        # 等待进程完成
        process_front.join()
        for process_mid in process_mid_pool:
            process_mid.join()
        process_back.join()

    print('Training completed.')




def train(train_loader, front, mid1, mid2, back, optimizef, optimizem1, optimizem2, optimizeb, epoch, writer):
# def train(train_loader, front, back, optimizef, epoch, writer):
    
    
    front.train()
    mid1.train()
    mid2.train()
    back.train()
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    train_batches_num = len(train_loader)
    end = time.time()

    for i, (x, target) in enumerate(tqdm(train_loader)):
        
        # shape=(1024, 3, 32, 32)
        # tensor = torch.empty(shape)
        # for i in range(shape[0]):
        #     for j in range(shape[1]):
        #         for k in range(shape[2]):
        #             for l in range(shape[3]):
        #                 tensor[i, j, k, l] = np.sin(i * 0.1 + j * 0.2 + k * 0.3 + l * 0.4)
        # x=tensor
        # # print(f'x: {x}')
        # target = torch.arange(1024) % 10
        
        target = target.cuda()
        oriimg = x.clone().detach().cuda()
        x = x.cuda()
     
        
        
        optimizef.zero_grad()
        optimizem1.zero_grad()
        optimizem2.zero_grad()
        optimizeb.zero_grad()
        frontout = front.forward_features(img=x, target=target, ixx_1=5, ixy_1=0.5, ixx_2=0, ixy_2=0)
        mid1out = mid1.forward_features(x=frontout, target=target, oriimg=oriimg, ixx_1=5, ixy_1=0.5, ixx_2=0, ixy_2=0)
        mid2out = mid2.forward_features(x=mid1out, target=target, oriimg=oriimg, ixx_1=5, ixy_1=0.5, ixx_2=0, ixy_2=0)
        
        output, loss = back(x=mid2out, target=target, oriimg=oriimg, ixx_1=5, ixy_1=0.5, ixx_2=0, ixy_2=0)

        
        # print(f'output: {output}')
        # exit(0)
        optimizef.step()
        optimizem1.step()
        optimizem2.step()
        optimizeb.step()

        # 计算 Prec@1 和更新指标
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), x.size(0))
        top1.update(prec1.item(), x.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % 10 == 0:
            # TensorBoard 中记录损失和精度
            writer.add_scalar('Loss/train', losses.ave, epoch * train_batches_num + i)
            writer.add_scalar('Prec@1/train_batch', top1.ave, epoch * train_batches_num + i)
        
        
            string = ('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                    'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                    'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                    epoch, i+1, train_batches_num, batch_time=batch_time,
                    loss=losses, top1=top1))

            print(string)
            
        if i == train_batches_num - 1:  # 最后一个 batch 记录速度
            speed = x.size(0) / batch_time.ave
            writer.add_scalar('Speed/train_batch', speed, epoch)

        

    return top1.ave, speed  # 返回训练精度和训练速度

        
    
    
def validate(val_loader, front,  mid1, mid2, back, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    
    train_batches_num = len(val_loader)
    front.eval()
    mid1.eval()
    mid2.eval()
    back.eval()
    end = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(tqdm(val_loader)):
            target = target.cuda()
            input = input.cuda()

            frontout = front.forward_features(img=input, target=target, ixx_1=5, ixy_1=0.5, ixx_2=0, ixy_2=0)
            mid1out = mid1.forward_features(x=frontout, target=target, oriimg=input, ixx_1=5, ixy_1=0.5, ixx_2=0, ixy_2=0)
            mid2out = mid2.forward_features(x=mid1out, target=target, oriimg=input, ixx_1=5, ixy_1=0.5, ixx_2=0, ixy_2=0)
            
            output, loss = back(x=mid2out, target=target, oriimg=input, ixx_1=5, ixy_1=0.5, ixx_2=0, ixy_2=0)

            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            
            
    string = ('Test: [{0}][{1}/{2}]\t'
              'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
              'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
              'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
        epoch, (i + 1), train_batches_num, batch_time=batch_time,
        loss=losses, top1=top1))
    
    print(string)
    # 在 TensorBoard 中记录验证集精度
    writer.add_scalar('Loss/val', losses.ave, epoch)
    writer.add_scalar('Prec@1/val', top1.ave, epoch)

    return top1.ave








def traink2(train_loader, front,back, optimizef, optimizeb, epoch):
# def train(train_loader, front, back, optimizef, epoch, writer):
    
    
    front.train()
    back.train()
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    train_batches_num = len(train_loader)
    end = time.time()

    for i, (x, target) in enumerate(tqdm(train_loader)):
 
        target = target.cuda()
        oriimg = x.clone().detach().cuda()
        x = x.cuda()
        optimizef.zero_grad()

        optimizeb.zero_grad()
        frontout = front.forward_features(img=x, target=target, ixx_1=5, ixy_1=0.5, ixx_2=0, ixy_2=0)
        
        output, loss = back(x=frontout, target=target, oriimg=oriimg, ixx_1=5, ixy_1=0.5, ixx_2=0, ixy_2=0)

        
      
        optimizef.step()
        optimizeb.step()

        # 计算 Prec@1 和更新指标
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), x.size(0))
        top1.update(prec1.item(), x.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % 10 == 0:
         
            string = ('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                    'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                    'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                    epoch, i+1, train_batches_num, batch_time=batch_time,
                    loss=losses, top1=top1))

            print(string)
            
        if i == train_batches_num - 1:  # 最后一个 batch 记录速度
            speed = x.size(0) / batch_time.ave

        

    return top1.ave, speed  # 返回训练精度和训练速度

        
    
    
def validatek2(val_loader, front, back, epoch, writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    
    train_batches_num = len(val_loader)
    front.eval()
    back.eval()
    end = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(tqdm(val_loader)):
            target = target.cuda()
            input = input.cuda()

            frontout = front.forward_features(img=input, target=target, ixx_1=5, ixy_1=0.5, ixx_2=0, ixy_2=0)
     
            output, loss = back(x=frontout, target=target, oriimg=input, ixx_1=5, ixy_1=0.5, ixx_2=0, ixy_2=0)

            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            
            
    string = ('Test: [{0}][{1}/{2}]\t'
              'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
              'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
              'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
        epoch, (i + 1), train_batches_num, batch_time=batch_time,
        loss=losses, top1=top1))
    
    print(string)
    # # 在 TensorBoard 中记录验证集精度
    # writer.add_scalar('Loss/val', losses.ave, epoch)
    # writer.add_scalar('Prec@1/val', top1.ave, epoch)

    return top1.ave










if __name__ == "__main__":
    main_k2()
    # training()