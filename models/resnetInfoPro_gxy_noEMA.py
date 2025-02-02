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

# from .config import InfoPro, InfoPro_balanced_memory
# from .auxiliary_nets_1 import Decoder, AuxClassifier

from config import InfoPro, InfoPro_balanced_memory
from auxiliary_nets_1 import Decoder, AuxClassifier

from torch.utils.tensorboard import SummaryWriter


"""
1. 更接近源代码的分割框架，很多地方写死了，扩展性差，复用了resnetInfoPro_gxy.py的框架
2. 完全不使用ema和lb
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


class InfoProResNet_front(nn.Module):

    def __init__(self, block, layers, arch, local_module_num, infopro_config, batch_size, image_size=32, inplanes=None,
                 balanced_memory=False, dataset='cifar10', class_num=10,
                 wide_list=(16, 16, 32, 64), dropout_rate=0,
                 aux_net_config='1c2f', local_loss_mode='contrast',
                 aux_net_widen=1, aux_net_feature_dim=128,momentum = 0.999, device=None):
        super(InfoProResNet_front, self).__init__()

     
        
        assert arch in ['resnet32', 'resnet110'], "This repo supports resnet32 and resnet110 currently. " \
                                                  "For other networks, please set network configs in .configs."
        
        self.widelist = wide_list
        self.inplanes = wide_list[0]
        self.dropout_rate = dropout_rate
        self.class_num = class_num
        self.local_module_num = local_module_num
        self.layers = layers
        self.momentum = momentum
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, wide_list[1], layers[0])
        self.layer2 = self._make_layer(block, wide_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, wide_list[3], layers[2], stride=2)
        
         # 假设 block 是构建每层的基本模块，wide_list 是每层的宽度列表，layers 是每层的深度
        self.layerslist = nn.ModuleList()  # 用于存储所有层的 ModuleList

        
        self.infopro_config=infopro_config
        
        
        
        for module_index in range(1,4):
            for layer_index in range(len(self.layer1)):

                exec('self.decoder_' + str(module_index) + '_' + str(layer_index) +
                     '= Decoder(wide_list[module_index], image_size, widen=aux_net_widen)')

                exec('self.aux_classifier_' + str(module_index) + '_' + str(layer_index) +
                     '= AuxClassifier(wide_list[module_index], net_config=aux_net_config, '
                     'loss_mode=local_loss_mode, class_num=class_num, '
                     'widen=aux_net_widen, feature_dim=aux_net_feature_dim)')
                
                
        # self.LB = nn.ModuleList([])
        # self.EMA_Net = nn.ModuleList([])

        # for item in self.infopro_config:
        #     module_index, layer_index = item
        #     if layer_index == len(self.layer1) - 1:
        #         mo,la = module_index + 1,0
        #     else:
        #         mo,la = module_index,layer_index + 1
        #     self.LB.append(copy.deepcopy(eval('self.layer' + str(mo))[la]))
        #     self.EMA_Net.append(copy.deepcopy(eval('self.layer' + str(mo))[la]))
            

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        
        
        
        if 'CIFAR' in dataset:
            self.mask_train_mean = torch.Tensor([x / 255.0 for x in [125.3, 123.0, 113.9]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).to(device)
            self.mask_train_std = torch.Tensor([x / 255.0 for x in [63.0, 62.1, 66.7]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).to(device)
        else:
            self.mask_train_mean = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).to(device)
            self.mask_train_std = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).to(device)

        self.Encoder_Net = self._make_Encoder_Aux_Net()

        
        
        
        for net in self.Encoder_Net:
            net = net.cuda()

        # for net1, net2 in zip(self.LB, self.EMA_Net):
        #     net1 = net1.cuda()
        #     net2 = net2.cuda()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.infopro_config = infopro_config
        self.criterion_ce = nn.CrossEntropyLoss()


 
        

        # print(self.Encoder_Net)
    #反归一化
    def _image_restore(self, normalized_image):
        return normalized_image.mul(self.mask_train_std[:normalized_image.size(0)]) \
               + self.mask_train_mean[:normalized_image.size(0)]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout_rate=self.dropout_rate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout_rate=self.dropout_rate))

        return nn.Sequential(*layers)

    def _make_Encoder_Aux_Net(self):
        Encoder_Net = nn.ModuleList([])

        Encoder_temp = nn.ModuleList([])

        local_block_index = 0

        # Build Encoder_Net 3个block 每个block有5层，
        for blocks in range(2):
            for layers in range(self.layers[blocks]):
                Encoder_temp.append(eval('self.layer' + str(blocks + 1))[layers])
                if blocks + 1 == self.infopro_config[local_block_index][0] \
                        and layers == self.infopro_config[local_block_index][1]:
                    Encoder_Net.append(nn.Sequential(*Encoder_temp))

                    Encoder_temp = nn.ModuleList([])
                    local_block_index += 1
        return Encoder_Net

   
    def forward_features(self, img, target=None,ixx_1=5, ixy_1=0.5,
                ixx_2=0, ixy_2=0):
        if self.training:
            # print(f'img  {img}')
            # exit(0)
            
            x = self.conv1(img)
            # print(f'x1 shape {x.shape}')
            x = self.bn1(x)
            x = self.relu(x)

            for i in range(len(self.Encoder_Net)):
                
                
                # print(f'x.shape {x.shape}')
                x = self.Encoder_Net[i](x)
                
                # y = self.LB[i](x) + self.EMA_Net[i](x)
                y=x
                    
                local_index,layer_index = self.infopro_config[i]
                lo,la = local_index,layer_index 
                # if layer_index == len(self.layer1) - 1:
                #     lo,la = local_index + 1,0
                # else:
                #     lo,la = local_index,layer_index 
                    
                    # lo,la = local_index,layer_index + 1
                    
                # print(f'lo {lo} la {la}  y.shape:{y.shape}')
                # print('self.decoder_' + str(lo) + '_' + str(la))
                # print('self.aux_classifier_' + str(lo) + '_' + str(la))    
                # print(f'self.layerslist[{lo}][{la}]')
                        
                    
                
                ratio = lo / (self.local_module_num - 2) if self.local_module_num > 2 else 0
                ixx_r = ixx_1 * (1 - ratio) + ixx_2 * ratio
                ixy_r = ixy_1 * (1 - ratio) + ixy_2 * ratio
 
                loss_ixx = eval('self.decoder_' + str(lo) + '_' + str(la))(y,self._image_restore(img))
                loss_ixy = eval('self.aux_classifier_' + str(lo) + '_' + str(la))(y,target)
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



class InfoProResNet_back(nn.Module):

    

        
    def __init__(self, block, layers, arch, local_module_num, infopro_config, batch_size, ini_idx=None, image_size=32, inplanes=None,
                 balanced_memory=False, dataset='cifar10', class_num=10,
                 wide_list=(16, 16, 32, 64), dropout_rate=0,
                 aux_net_config='1c2f', local_loss_mode='contrast',
                 aux_net_widen=1, aux_net_feature_dim=128,momentum = 0.999, device=None):
        super(InfoProResNet_back, self).__init__()

     
        
        assert arch in ['resnet32', 'resnet110'], "This repo supports resnet32 and resnet110 currently. " \
                                                  "For other networks, please set network configs in .configs."
        self.widelist = wide_list
        self.inplanes = inplanes
        self.dropout_rate = dropout_rate
        
        self.layer3 = self._make_layer(block, wide_list[3], layers[0], stride=2)
        
        self.feature_num = wide_list[-1]
        self.class_num = class_num
        self.local_module_num = local_module_num
        self.layers = layers
        self.momentum = momentum


    
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.feature_num, self.class_num)
        self.head = nn.Linear(self.feature_num, self.class_num)
        
        self.Flatten = nn.Flatten()

        self.infopro_config = infopro_config
        self.criterion_ce = nn.CrossEntropyLoss()

    
        for module_index in range(3,4):
            for layer_index in range(len(self.layer3)):

                exec('self.decoder_' + str(module_index) + '_' + str(layer_index) +
                     '= Decoder(wide_list[module_index], image_size, widen=aux_net_widen)')

                exec('self.aux_classifier_' + str(module_index) + '_' + str(layer_index) +
                     '= AuxClassifier(wide_list[module_index], net_config=aux_net_config, '
                     'loss_mode=local_loss_mode, class_num=class_num, '
                     'widen=aux_net_widen, feature_dim=aux_net_feature_dim)')

        
        # 创建LB,EMA_Net
        # self.LB = nn.ModuleList([])
        # self.EMA_Net = nn.ModuleList([])

        
        # for item in self.infopro_config[:-1]:
        #     module_index, layer_index = item
        #     if layer_index == len(self.layer3) - 1:
        #         mo,la = module_index + 1,0
        #     else:
        #         mo,la = module_index,layer_index + 1
        #     self.LB.append(copy.deepcopy(eval('self.layer' + str(mo))[la]))
        #     self.EMA_Net.append(copy.deepcopy(eval('self.layer' + str(mo))[la]))

                

        # for i in range(len(self.EMA_Net)):
        #     for param in self.EMA_Net[i].parameters():
        #         param.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        
        
        if 'CIFAR' in dataset:
            # print(f'device {device}')
            self.mask_train_mean = torch.Tensor([x / 255.0 for x in [125.3, 123.0, 113.9]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).to(device)
            self.mask_train_std = torch.Tensor([x / 255.0 for x in [63.0, 62.1, 66.7]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).to(device)
        else:
            self.mask_train_mean = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).to(device)
            self.mask_train_std = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).to(device)

        self.Encoder_Net = self._make_Encoder_Aux_Net()

        for net in self.Encoder_Net:
            net = net.cuda()

        # for net1, net2 in zip(self.LB, self.EMA_Net):
        #     net1 = net1.cuda()
        #     net2 = net2.cuda()
            
            
        

    def _image_restore(self, normalized_image):
        
        # print(f'normalized_image.shape: {normalized_image.shape}')
        return normalized_image.mul(self.mask_train_std[:normalized_image.size(0)]) \
               + self.mask_train_mean[:normalized_image.size(0)]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout_rate=self.dropout_rate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout_rate=self.dropout_rate))

        return nn.Sequential(*layers)

    def _make_Encoder_Aux_Net(self):
        Encoder_Net = nn.ModuleList([])

        Encoder_temp = nn.ModuleList([])

        local_block_index = 0

        # Build Encoder_Net 第3个block
        for blocks in range(2,2+len(self.layers)):
            for layers in range(self.layers[blocks-2]):
                
                Encoder_temp.append(eval('self.layer' + str(blocks + 1))[layers])
                if blocks + 1 == self.infopro_config[local_block_index][0] \
                        and layers == self.infopro_config[local_block_index][1]:
                    Encoder_Net.append(nn.Sequential(*Encoder_temp))

                    Encoder_temp = nn.ModuleList([])
                    local_block_index += 1
                    
                    
        return Encoder_Net

    def forward(self, x, target=None, oriimg=None, ixx_1=5, ixy_1=0.5,
                ixx_2=0, ixy_2=0):
        if self.training:

            
            for i in range(len(self.Encoder_Net)-1):
                
                
                
                x = self.Encoder_Net[i](x)
                # y = self.LB[i](x) + self.EMA_Net[i](x)
                y=x

                    
                local_index,layer_index = self.infopro_config[i]
                lo,la = local_index,layer_index 
                # if layer_index == len(self.layer3) - 1:
                #     lo,la = local_index + 1,0
                # else:
                #     lo,la = local_index,layer_index 
          
           
     
                ratio = lo / (self.local_module_num - 2) if self.local_module_num > 2 else 0
                ixx_r = ixx_1 * (1 - ratio) + ixx_2 * ratio
                ixy_r = ixy_1 * (1 - ratio) + ixy_2 * ratio
                loss_ixx = eval('self.decoder_' + str(lo) + '_' + str(la))(y,self._image_restore(oriimg))
                loss_ixy = eval('self.aux_classifier_' + str(lo) + '_' + str(la))(y,target)
                loss = ixx_r * loss_ixx + ixy_r * loss_ixy
                loss.backward()

                x = x.detach()
                # cur_layer = eval(f'self.layerslist[{i+1}]')
                
                # for paramEncoder, paramEMA in zip(cur_layer.parameters(),self.EMA_Net[i].parameters()):
                    
                #     paramEMA.data = paramEMA.data * self.momentum + paramEncoder.data * (1 - self.momentum)
                
            

            # last local module
            x = self.Encoder_Net[-1](x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

            logits = self.head(x)
            
            # return logits
            loss = self.criterion_ce(logits, target)
            loss.backward()
            return logits,loss
            
            # local_index, layer_index = self.infopro_config[-1]
            # cur_layer = eval(f'self.layerslist[{local_index-1}]')[layer_index]
            # return x,loss

        else:
 

            for i in range(len(self.Encoder_Net)):
                x = self.Encoder_Net[i](x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            logits = self.head(x)
            loss = self.criterion_ce(logits, target)
            return logits, loss
            
            return logits




def resnet32_front(**kwargs):
    # layers=kwargs['layers']
    # infopro_config=kwargs['infopro_config']
    # wide_list=kwargs['wide_list']
    print(f"pos1 {kwargs['local_module_num']}")
    model = InfoProResNet_front(BasicBlock, arch='resnet32', **kwargs)
    
    return model

def resnet32_mid(**kwargs):
    return 

def resnet32_back(**kwargs):
    model = InfoProResNet_back(BasicBlock, arch='resnet32', **kwargs)
    return model


# def resnet110(**kwargs):
#     model = InfoProResNet(BasicBlock, [18, 18, 18], arch='resnet110', **kwargs)
#     return model

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
        'changing_lr': [80, 120],
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

def main():
    torch.manual_seed(42)
    
    
    writer = SummaryWriter(log_dir="./runs/InfoProResNet_ori")
    
    epoches=400
    cfg='configs/k2_resnet32_gxy.yaml'
    
    with open(cfg, 'r') as file1:
        cfg_data=  yaml.load(file1,Loader=yaml.FullLoader)
    
    layers_f=cfg_data['front']['layers']
    
        
    infopro_config_f=InfoPro[cfg_data['common']['InfoPro']][cfg_data['common']['local_module_num']][:10]
    f_inplanes=cfg_data['front']['inplanes']
    
    layers_b=cfg_data['back']['layers']
    
    
    
    infopro_config_b=InfoPro[cfg_data['common']['InfoPro']][cfg_data['common']['local_module_num']][10:]
    b_inplanes=cfg_data['back']['inplanes']
    
    
    wide_listf=cfg_data['front']['wide_list']
    wide_listb=cfg_data['back']['wide_list']
    
    
    batch_size=cfg_data['common']['batch_size']
    image_size=cfg_data['common']['image_size']
    dropout_rate=cfg_data['common']['dropout_rate']
    class_num=cfg_data['common']['class_num']
    aux_net_config=cfg_data['common']['aux_net_config']
    local_loss_mode=cfg_data['common']['local_loss_mode']
    aux_net_widen=cfg_data['common']['aux_net_widen']
    aux_net_feature_dim=cfg_data['common']['aux_net_feature_dim']
    local_module_num=cfg_data['common']['local_module_num']
    
  
    
    front= resnet32_front(layers=layers_f ,infopro_config=infopro_config_f, local_module_num=local_module_num, inplanes=f_inplanes, batch_size=1024, image_size=32,
                   dataset='cifar10', class_num=10, wide_list=wide_listf,
                    dropout_rate=0,
                   aux_net_config=aux_net_config, local_loss_mode=local_loss_mode,
                   aux_net_widen=1, aux_net_feature_dim=128, device='cuda:0')
    front = front.cuda()
    print(front)
    # exit(0)
    # initialize_weights(front)
    
    # print(front)
    front_para=0
    modules_n=['decoder' ,'aux_classifier'  ,'LB' ,'EMA_Net' ,'Encoder_Net']
    for name, param in front.named_parameters():
        for it in modules_n:
            if it in name:
                front_para+=param.numel()
                break

    # print(f'front_para: {front_para}')

    back= resnet32_back(layers=layers_b ,infopro_config=infopro_config_b, local_module_num=local_module_num, inplanes=b_inplanes, batch_size=1024, image_size=32,
                   dataset='cifar10', class_num=10, wide_list=wide_listb,
                   dropout_rate=0,
                   aux_net_config=aux_net_config, local_loss_mode=local_loss_mode,
                   aux_net_widen=1, aux_net_feature_dim=128,device='cuda:0')
    
    # print(back)
    back_para=0
    for name, param in back.named_parameters():
        for it in modules_n:
            if it in name:
                back_para+=param.numel()
                break
    # print(f'back_para: {back_para}')
    
    # exit(0)
    back = back.cuda()
    # initialize_weights(back)
    
    
    
    
    cache=[]
   
                                   
    # optimizef = torch.optim.SGD(list(front.parameters()) + list(back.parameters()),
    #                         lr=0.01,
    #                         momentum=0.995,
    #                         nesterov=True,
    #                         weight_decay=0.0001)

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

    
    
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(train_loader)

    # switch to train mode
    front.train()
    back.train()
    
    
    print(f'train_batches_num {train_batches_num}')
    end = time.time()
    for epoch in range(epoches):
        # train(train_loader, front,back, optimizef, optimizeb, epoch)
            
        adjust_learning_rate('resnet',optimizef, epoch + 1,False)
        adjust_learning_rate('resnet',optimizeb, epoch + 1,False)
        # for param_groupf,param_groupb in zip(optimizef.param_groups,optimizeb.param_groups):
        #     print(f"Lr check Epoch {epoch}: Current learning rate for optimizef:{param_groupf['lr']}   optimizeb:{param_groupb['lr']}")
        # prec1 = validate(val_loader, front,back, epoch)
    
        train_prec1, train_speed = train(train_loader, front, back, optimizef, optimizeb, epoch, writer)
        
        val_prec1 = validate(val_loader, front, back, epoch, writer)
            
        # 记录 epoch 结束时的平均精度
        writer.add_scalar('Prec@1/train', train_prec1, epoch)
        writer.add_scalar('Prec@1/val', val_prec1, epoch)
        writer.add_scalar('Speed/train', train_speed, epoch)
        
    writer.close()  # 关闭记录器  





def train(train_loader, front, back, optimizef, optimizeb, epoch, writer):
# def train(train_loader, front, back, optimizef, epoch, writer):
    
    
    front.train()
    back.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    train_batches_num = len(train_loader)
    end = time.time()

    import numpy as np
    for i, (x, target) in enumerate(tqdm(train_loader)):
        # # print(f'target.shape :{target.shape}')
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
        optimizeb.zero_grad()
        frontout = front.forward_features(img=x, target=target, ixx_1=5, ixy_1=0.5, ixx_2=0, ixy_2=0)
        output, loss = back(x=frontout, target=target, oriimg=oriimg, ixx_1=5, ixy_1=0.5, ixx_2=0, ixy_2=0)
       
        optimizef.step()
        optimizeb.step()

        # print("Front model weights after step:")
        # for name, param in front.named_parameters():
        #     if param.requires_grad and "aux_classifier" in name:
        #         print(f"Layer: {name} - Weights shape: {param.shape}")
        #         print(param.data)

        # 打印后模块的权重
        # print("\nBack model weights after step:")
        # for name, param in back.named_parameters():
        #     if param.requires_grad and "aux_classifier" in name:
        #         print(f"Layer: {name} - Weights shape: {param.shape}")
        #         print(param.data)
        
        # exit(0)
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

        
    
    
def validate(val_loader, front, back, epoch, writer):
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
    # 在 TensorBoard 中记录验证集精度
    writer.add_scalar('Loss/val', losses.ave, epoch)
    writer.add_scalar('Prec@1/val', top1.ave, epoch)

    return top1.ave


if __name__ == "__main__":
    main()